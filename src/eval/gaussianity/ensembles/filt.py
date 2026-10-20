from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from src.eval.gaussianity.ensembles import common as C


class _PerMemberMu0:

    def __init__(self, v):
        self.v = v

    def view(self, *s):
        return self

    def to(self, *a, **k):
        return self

    def unsqueeze(self, d):
        return self

    def squeeze(self, d):
        return self.v

    def repeat(self, n, *a):
        return self.v


def _load_dbf(ds: str, density: int, dev):
    from src.eval.benchmark import dbf_load

    dbf, ckp = dbf_load(ds, density, dev)
    for n in ("f_network", "G_network", "h_network"):
        setattr(dbf, n, getattr(dbf, n).to(dev).eval())
    return dbf, ckp


def run(
    ds: str,
    *,
    members: int,
    seed: int,
    density: int,
    chunk: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict]:
    from src.assimilation.observation import MaskSampler
    from src.eval.benchmark import DBF_MASK_FIXED

    dev = C.device()
    cfg, data, runs, _sampler, geom = C.load_protocol(ds, density)
    sampler = MaskSampler(
        cfg["image_size"],
        float(cfg["obs_ratio"]),
        cfg.get("mask_kind", "random"),
        int(cfg.get("mask_seed", 0)),
        fixed=DBF_MASK_FIXED[ds],
    )
    dbf, ckp = _load_dbf(ds, density, dev)
    dbf.bench_obs_every = int(cfg.get("obs_every", 1))
    enc = lambda x: dbf.f_network(x.to(dev).reshape(x.shape[0], -1)).float()

    zdim = dbf.z_dim
    V0 = dbf.V_init.to(dev).double()
    Bmat = torch.block_diag(*V0.reshape(-1, 2, 2)) if V0.dim() == 4 else V0
    Bchol = torch.linalg.cholesky(Bmat).float()
    Bnp = Bmat.cpu().numpy()

    draw = C.Draw(seed)
    spin, Wn, rl = geom["spinup"], geom["window"], geom["run_length"]
    orig_mu0 = dbf.mu_0
    out, refs = [], []
    try:
        for si in range(len(runs)):
            gt = [runs[si][t].unsqueeze(0).to(dev) for t in range(rl + 1)]
            c0 = min(spin, rl)
            n_step = min(Wn, rl - c0) + 1
            with torch.no_grad():
                z_true = enc(gt[c0])[0]
            eps = draw.bg(members, z_true.shape[0]).to(dev) @ Bchol.T
            mu0 = z_true.unsqueeze(0) + eps
            obs1 = torch.stack(
                [
                    (gt[c0 + k] * sampler.mask(si, c0 + k).to(dev)).reshape(1, -1)
                    for k in range(n_step)
                ],
                dim=1,
            ).float()
            Za = []
            for a0 in range(0, members, chunk):
                sl = slice(a0, min(a0 + chunk, members))
                mm = mu0[sl]
                nb = mm.shape[0]
                v = mm.reshape(nb, zdim // 2, 2) if dbf.sigma_block_diag else mm
                dbf.mu_0 = _PerMemberMu0(v)
                with torch.no_grad():
                    _, _, mu_t, _ = dbf._compute_mu_t_sigma_t(
                        obs1.expand(nb, -1, -1).contiguous(), n_step, nb, 1
                    )
                Za.append(mu_t[:, 0].reshape(nb, -1).detach())
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            Z0 = torch.cat(Za)
            out.append(
                (
                    eps.cpu().numpy().astype(np.float64),
                    (Z0 - z_true.unsqueeze(0)).cpu().numpy().astype(np.float64),
                )
            )
            refs.append(float(z_true.pow(2).mean().sqrt()))
    finally:
        dbf.mu_0 = orig_mu0

    meta = {
        "ckpt": ckp,
        "mask_fixed": bool(DBF_MASK_FIXED[ds]),
        "reference_rms": float(np.mean(refs)),
        "geometry": geom,
        "latent_dim": int(zdim),
    }
    return out, Bnp, meta
