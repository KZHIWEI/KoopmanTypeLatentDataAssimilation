from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from src.config import build_ae_config
from src.assimilation.variational import _rrmse_batch
from src.assimilation.observation import MaskSampler
from src.model.tensorvar import TensorVarForward, TensorVarInverse


def snapshot_shared_ae(fwd):
    return {
        k: v.detach().clone()
        for k, v in fwd.state_dict().items()
        if k.startswith(("phi_S.", "phi_inv_S."))
    }


def assert_shared_ae_unmoved(fwd, before, inv_ckpt, fwd_ckpt):
    after = fwd.state_dict()
    moved = [k for k, v in before.items() if not torch.equal(v, after[k])]
    if moved:
        raise RuntimeError(inv_ckpt, fwd_ckpt, moved[0])


class TVDA:

    def __init__(self, fwd_ckpt: str, inv_ckpt: str | None, device: torch.device):
        ck_f = torch.load(fwd_ckpt, map_location="cpu", weights_only=False)
        cfg = ck_f["cfg"]
        fwd = TensorVarForward(build_ae_config(cfg))
        fwd.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in ck_f["state_dict"].items()}
        )
        if inv_ckpt is None:
            inv = None
        else:
            ck_i = torch.load(inv_ckpt, map_location="cpu", weights_only=False)
            icfg = ck_i["cfg"]
            inv = TensorVarInverse(
                build_ae_config(icfg),
                fwd,
                history_len=int(icfg.get("history_len", 10)),
                obs_ratio=float(icfg.get("obs_ratio", 0.15)),
                mask_aware=bool(icfg.get("mask_aware", False)),
            )
            before = snapshot_shared_ae(fwd)
            inv.load_state_dict(
                {k.replace("_orig_mod.", ""): v for k, v in ck_i["state_dict"].items()}
            )
            assert_shared_ae_unmoved(fwd, before, inv_ckpt, fwd_ckpt)
            for p in inv.parameters():
                p.requires_grad_(False)
        for p in fwd.parameters():
            p.requires_grad_(False)
        self.fwd = fwd.to(device).eval()
        self.inv = inv.to(device).eval() if inv is not None else None
        self.device = device
        self.train_cfg = cfg
        self.history_len = inv.history_len if inv is not None else None
        self.C = fwd.C_forward.detach().t().to(device)
        self.D = self.C.shape[0]

    def encode(self, x):
        return self.fwd.encode(x.to(self.device))

    def decode(self, z):
        return self.fwd.decode(z)

    def step(self, z):
        return self.fwd.step(z)


@torch.no_grad()
def feature_obs(
    model: TVDA,
    gt: List[torch.Tensor],
    sampler: MaskSampler,
    run_length: int,
    obs_every: int = 1,
) -> torch.Tensor:
    B = gt[0].shape[0]
    C, H, W = gt[0].shape[1:]
    m = model.history_len
    obs = []
    msk = []
    for t in range(run_length + 1):
        if t % obs_every == 0:
            masks = torch.stack([sampler.mask(b, t) for b in range(B)]).to(model.device)
            obs.append(gt[t] * masks)
            msk.append(masks)
        else:
            obs.append(torch.zeros_like(gt[t]))
            msk.append(torch.zeros(B, 1, H, W, device=model.device))
    zero = torch.zeros_like(obs[0])
    zero1 = torch.zeros_like(msk[0])
    ma = bool(getattr(model.inv, "mask_aware", False))
    ys = []
    for t in range(run_length + 1):
        hist = torch.cat(
            [obs[t - k] if t - k >= 0 else zero for k in range(m - 1, -1, -1)], dim=1
        )
        if ma:
            mh = torch.cat(
                [msk[t - k] if t - k >= 0 else zero1 for k in range(m - 1, -1, -1)],
                dim=1,
            )
            hist = torch.cat([hist, mh], dim=1)
        ys.append(model.inv.observe(hist).float())
    return torch.stack(ys, dim=1)


@torch.no_grad()
def estimate_QR(
    model: TVDA,
    cfg,
    sampler: MaskSampler,
    obs_every: int,
    n_windows: int = 400,
    ridge: float = 0.1,
    batch: int = 64,
    phase: str = "all",
):
    import numpy as np
    from src.data import load_split, train_norm

    tr = load_split(cfg["data_path"])
    mean, std = train_norm(cfg)
    skip = int(cfg.get("skip", 1))
    m = model.history_len
    ntraj, T = tr.shape[0], tr.shape[1]
    rng = np.random.RandomState(0)
    D = model.D
    ma = bool(getattr(model.inv, "mask_aware", False))
    Ct = model.C.t().float()
    eQ, eR = [], []
    per = max(1, n_windows // ntraj)
    jobs_q = [
        (ti, int(t0))
        for ti in range(ntraj)
        for t0 in rng.randint(m * skip, T - skip - 1, size=per)
    ]

    def _snap(t0):
        q, r = divmod(t0, skip)
        return max(m * skip, (q // obs_every) * obs_every * skip + r)

    jobs_r = [(ti, _snap(t0)) for ti, t0 in jobs_q] if phase == "obs" else jobs_q
    bs = int(max(1, batch))

    def _windows(jobs):
        for i0 in range(0, len(jobs), bs):
            chunk = jobs[i0 : i0 + bs]
            frames = np.stack(
                [
                    np.asarray(
                        tr[
                            ti,
                            [t0 - k * skip for k in range(m - 1, -1, -1)] + [t0 + skip],
                        ]
                    )
                    for ti, t0 in chunk
                ]
            ).astype(np.float32)
            frames = (frames - mean.reshape(1, 1, -1, 1, 1)) / std.reshape(
                1, 1, -1, 1, 1
            )
            w = torch.tensor(frames, device=model.device)
            yield chunk, w, model.encode(w[:, m - 1]).float()

    for _, w, z_t in _windows(jobs_q):
        eQ.append(model.encode(w[:, m]).float() - z_t @ Ct)
    for chunk, w, z_t in _windows(jobs_r):
        hist, mhist = [], []
        for j, (_, t0) in enumerate(chunk):
            hj, mj = [], []
            for k in range(m):
                t_abs = t0 - (m - 1 - k) * skip
                if (t_abs // skip) % obs_every == 0:
                    mask = sampler.mask(0, int(t_abs)).to(model.device)
                    hj.append(w[j, k] * mask)
                    mj.append(mask.reshape(1, *mask.shape[-2:]))
                else:
                    hj.append(torch.zeros_like(w[j, k]))
                    mj.append(torch.zeros(1, *w[j, k].shape[-2:], device=w.device))
            hist.append(torch.cat(hj, dim=0))
            mhist.append(torch.cat(mj, dim=0))
        _h = torch.stack(hist)
        if ma:
            _h = torch.cat([_h, torch.stack(mhist)], dim=1)
        eR.append(model.inv.observe(_h).float() - z_t)
    EQ = torch.cat(eQ).double()
    ER = torch.cat(eR).double()

    def inv_cov(E):
        cov = (E.t() @ E) / E.shape[0]
        return torch.linalg.inv(
            cov
            + ridge
            * cov.diagonal().mean()
            * torch.eye(D, device=E.device, dtype=torch.float64)
        ).float()

    Qi, Ri = inv_cov(EQ), inv_cov(ER)
    print(
        f"[QR] estimated from {EQ.shape[0]} samples: "
        f"trQ={float((EQ.var(0)).mean()):.4g} trR={float((ER.var(0)).mean()):.4g}",
        flush=True,
    )
    return Qi.to(model.device), Ri.to(model.device)


def _bi_matrix(Bi: torch.Tensor, D: int) -> torch.Tensor:
    return torch.diag(Bi) if Bi.dim() == 1 else Bi


def solve_window_closed(
    C: torch.Tensor,
    z_b: torch.Tensor,
    y: torch.Tensor,
    obs_idx: List[int],
    Bi: torch.Tensor,
    sigma_q: float,
    sigma_r: float,
    horizon: int,
    constraint: str = "weak",
    Qi: Optional[torch.Tensor] = None,
    Ri: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dev = z_b.device
    B, D = z_b.shape
    W = horizon - 1
    q, r = 1.0 / sigma_q**2, 1.0 / sigma_r**2
    C64 = C.double()
    I = torch.eye(D, device=dev, dtype=torch.float64)
    QiM = q * (Qi.double() if Qi is not None else I)
    RiM = r * (Ri.double() if Ri is not None else I)
    BiM = _bi_matrix(Bi, D).double()
    zb = z_b.double()
    obs_set = set(obs_idx)

    if constraint == "strong":
        A = BiM.clone()
        rhs = zb @ BiM.t()
        G = I.clone()
        for t in range(horizon):
            if t in obs_set:
                A = A + G.t() @ RiM @ G
                rhs = rhs + (y[:, t].double() @ RiM) @ G
            G = C64 @ G
        z0 = torch.linalg.solve(A, rhs.t()).t()
        Z = [z0]
        for _ in range(W):
            Z.append(Z[-1] @ C64.t())
        return torch.stack(Z, dim=1).float()

    CtQC = C64.t() @ QiM @ C64
    U = -(C64.t() @ QiM)
    L = -(QiM @ C64)
    A_diag, rhs = [], []
    for t in range(horizon):
        A_t = torch.zeros_like(I)
        r_t = torch.zeros(B, D, device=dev, dtype=torch.float64)
        if t == 0:
            A_t = A_t + BiM
            r_t = r_t + zb @ BiM.t()
        if t < W:
            A_t = A_t + CtQC
        if t > 0:
            A_t = A_t + QiM
        if t in obs_set:
            A_t = A_t + RiM
            r_t = r_t + y[:, t].double() @ RiM.t()
        A_diag.append(A_t)
        rhs.append(r_t)

    S = [A_diag[0]]
    G = [rhs[0]]
    for t in range(1, horizon):
        Wt = torch.linalg.solve(S[t - 1].t(), L.t()).t()
        S.append(A_diag[t] - Wt @ U)
        G.append(rhs[t] - G[t - 1] @ Wt.t())
    Z = [None] * horizon
    Z[W] = torch.linalg.solve(S[W], G[W].t()).t()
    for t in range(W - 1, -1, -1):
        Z[t] = torch.linalg.solve(S[t], (G[t] - Z[t + 1] @ U.t()).t()).t()
    return torch.stack(Z, dim=1).float()


@torch.no_grad()
def assimilate_runs_tv(
    model: TVDA,
    runs,
    sampler,
    Bi,
    sigma_q,
    sigma_r,
    *,
    window_size: int,
    spinup: int,
    run_length: int,
    obs_every: int,
    constraint: str = "weak",
    Qi: Optional[torch.Tensor] = None,
    Ri: Optional[torch.Tensor] = None,
    return_analysis: bool = False,
):
    dev = model.device
    B = len(runs)
    gt = [
        torch.stack([runs[b][t] for b in range(B)]).to(dev)
        for t in range(run_length + 1)
    ]
    if spinup >= run_length:
        y = torch.zeros(B, run_length + 1, model.D, device=dev)
    else:
        y = feature_obs(model, gt, sampler, run_length, obs_every)

    z = model.encode(gt[0]).float()
    analysis = torch.zeros(B, run_length + 1, model.D, device=dev)
    for t in range(min(spinup, run_length + 1)):
        analysis[:, t] = z
        z = model.step(z).float()
    z_b = z

    cursor = min(spinup, run_length)
    while cursor < run_length:
        Wn = min(window_size, run_length - cursor)
        obs_idx = [k for k in range(Wn + 1) if (cursor + k) % obs_every == 0]
        Z = solve_window_closed(
            model.C,
            z_b,
            y[:, cursor : cursor + Wn + 1],
            obs_idx,
            Bi,
            sigma_q,
            sigma_r,
            Wn + 1,
            constraint,
            Qi=Qi,
            Ri=Ri,
        )
        analysis[:, cursor : cursor + Wn + 1] = Z
        z_b = Z[:, Wn]
        cursor += Wn

    errs = []
    for t in range(run_length):
        errs.append(_rrmse_batch(model.decode(analysis[:, t]), gt[t]).cpu().numpy())
    if return_analysis:
        return np.stack(errs, axis=1), analysis.detach().cpu().numpy()
    return np.stack(errs, axis=1)


def score(model, runs, sampler, Bi, sq, sr, **kw):
    arr = assimilate_runs_tv(model, runs, sampler, Bi, sq, sr, **kw)
    spin = int(kw.get("spinup", 0))
    return float(arr[:, spin:].mean()), arr.mean(0)
