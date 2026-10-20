from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from src.assimilation.variational import nmc_background, solve_window
from src.assimilation.models import build_model
from src.eval.gaussianity.ensembles import common as C


def run(
    ds: str,
    model_name: str,
    *,
    members: int,
    seed: int,
    density: int,
    chunk: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict]:
    dev = C.device()
    from src.eval.benchmark import CKPT

    mkey = model_name.lower().replace("-", "_")
    sr, sq, sig_prov = C.sigma_for(ds, model_name, density)
    cfg, data, runs, sampler, geom = C.load_protocol(ds, density)

    model = build_model(mkey, CKPT[(ds, mkey)], device=dev, compile=False)
    samples = data.nmc_samples(
        int(cfg["nmc_lead"]),
        int(cfg["skip"]),
        int(cfg["nmc_samples"]),
        seed=int(cfg.get("mask_seed", 0)),
    )
    Bi = nmc_background(
        model,
        data.frame_batched,
        samples,
        int(cfg["nmc_lead"]),
        int(cfg["skip"]),
        diagonal=bool(cfg.get("b_diagonal", False)),
        ridge=float(cfg["b_ridge"]),
    ).to(dev)
    Bchol, Bnp = C.chol_from_precision(Bi, dev)

    draw = C.Draw(seed)
    W, spin, oev = geom["window"], geom["spinup"], geom["obs_every"]
    rl = geom["run_length"]
    out, ref_rms = [], []
    for si in range(len(runs)):
        gt = [runs[si][t].unsqueeze(0).to(dev) for t in range(rl + 1)]
        cursor = min(spin, rl)
        with torch.no_grad():
            z_true = model.encode(gt[cursor]).float()[0]
        D = z_true.shape[0]
        eps = draw.bg(members, D).to(dev) @ Bchol.T
        z_b = z_true.unsqueeze(0) + eps

        Wn = min(W, rl - cursor)
        obs_idx = [k for k in range(Wn + 1) if (cursor + k) % oev == 0]
        if obs_idx:
            m1 = torch.stack([sampler.mask(si, cursor + k) for k in obs_idx]).to(dev)
            masks = m1.unsqueeze(0).expand(members, *m1.shape).contiguous()
            o1 = torch.stack([gt[cursor + k][0] for k in obs_idx])
            clean = o1.unsqueeze(0).expand(members, *o1.shape)
            obs = (clean * masks).contiguous()
        else:
            masks = obs = None

        Zs = []
        for a0 in range(0, members, chunk):
            sl = slice(a0, min(a0 + chunk, members))
            Zs.append(
                solve_window(
                    model,
                    z_b[sl],
                    obs_idx,
                    None if masks is None else masks[sl],
                    None if obs is None else obs[sl],
                    Bi,
                    sq,
                    sr,
                    Wn + 1,
                    iters=int(cfg["iters"]),
                    lr=float(cfg["lr"]),
                    constraint=cfg.get("constraint", "weak"),
                    warm_start=bool(cfg.get("warm_start", False)),
                    converged=True,
                ).detach()
            )
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        Z = torch.cat(Zs)
        e_b = eps.cpu().numpy().astype(np.float64)
        e_a = (Z[:, 0] - z_true.unsqueeze(0)).cpu().numpy().astype(np.float64)
        out.append((e_b, e_a))
        ref_rms.append(float(z_true.pow(2).mean().sqrt()))

    meta = {
        "sigma_r": sr,
        "sigma_q": sq,
        "sigma_source": sig_prov,
        "reference_rms": float(np.mean(ref_rms)),
        "geometry": geom,
        "latent_dim": int(Bnp.shape[0]),
    }
    return out, Bnp, meta
