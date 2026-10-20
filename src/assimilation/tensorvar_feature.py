from __future__ import annotations


import numpy as np
import torch

from src.assimilation.variational import _rrmse_batch
from src.assimilation.tensorvar import TVDA, feature_obs


@torch.no_grad()
def collect_stats(
    model: TVDA,
    cfg,
    obs_ratio: float,
    mask_kind: str,
    n_seg: int,
    seg_len: int,
    lag: int,
    seed: int = 12345,
    batch: int = 512,
):
    from src.data import load_split, train_norm

    tr = load_split(cfg["data_path"])
    mean, std = train_norm(cfg)
    mean, std = mean.reshape(1, -1, 1, 1), std.reshape(1, -1, 1, 1)
    ntraj, T, Cc, H, W = tr.shape
    skip = int(cfg.get("skip", 1))
    m = model.history_len
    dev = model.device
    ma = bool(getattr(model.inv, "mask_aware", False))
    seg_len = min(seg_len, (T - 1 - (m - 1) * skip) // skip - 1)
    rng = np.random.RandomState(seed)
    npix = H * W
    nobs = int(round(obs_ratio * npix))

    Ys, Zs, EQs, SEGY, SEGZ = [], [], [], [], []
    n_fr = m - 1 + seg_len + 1
    chunk = max(1, batch // max(1, n_fr // 8))
    s = 0
    while s < n_seg:
        b = min(chunk, n_seg - s)
        s += b
        ti = rng.randint(ntraj, size=b)
        t0 = rng.randint((m - 1) * skip, T - seg_len * skip - 1, size=b)
        idx = np.stack([[t + (k - (m - 1)) * skip for k in range(n_fr)] for t in t0])
        w = np.stack([np.asarray(tr[int(a), list(r)]) for a, r in zip(ti, idx)])
        w = torch.from_numpy(((w - mean) / std).astype(np.float32)).to(dev)
        r = np.argsort(rng.rand(b, n_fr, npix), axis=-1)
        mk = np.zeros((b, n_fr, npix), dtype=np.float32)
        np.put_along_axis(mk, r[:, :, :nobs], 1.0, axis=-1)
        masks = torch.from_numpy(mk).to(dev).view(b, n_fr, 1, H, W)
        obs = w * masks
        hs = []
        for j in range(seg_len + 1):
            i = (m - 1) + j
            h = obs[:, i - m + 1 : i + 1].reshape(b, -1, H, W)
            if ma:
                h = torch.cat(
                    [h, masks[:, i - m + 1 : i + 1].reshape(b, -1, H, W)], dim=1
                )
            hs.append(h)
        hs = torch.stack(hs, 1)
        flat_h = hs.reshape(b * (seg_len + 1), *hs.shape[2:])
        flat_x = w[:, m - 1 :].reshape(-1, Cc, H, W)
        y = torch.cat(
            [
                model.inv.observe(flat_h[i : i + 64]).float()
                for i in range(0, len(flat_h), 64)
            ]
        ).reshape(b, seg_len + 1, -1)
        z = torch.cat(
            [
                model.encode(flat_x[i : i + 64]).float()
                for i in range(0, len(flat_x), 64)
            ]
        ).reshape(b, seg_len + 1, -1)
        Ys.append(y.reshape(-1, y.shape[-1]))
        Zs.append(z.reshape(-1, z.shape[-1]))
        SEGY.extend(list(y))
        SEGZ.extend(list(z))
        EQs.append(
            (z[:, 1:] - z[:, :-1] @ model.C.t().float()).reshape(-1, z.shape[-1])
        )
    Y = torch.cat(Ys)
    Z = torch.cat(Zs)
    EQ = torch.cat(EQs)
    return dict(Y=Y, Z=Z, EQ=EQ, SEGY=SEGY, SEGZ=SEGZ)


def _lsq(X: torch.Tensor, Tg: torch.Tensor, ridge: float):
    n = X.shape[1]
    G = X.t() @ X
    G = G + ridge * G.diagonal().mean() * torch.eye(n, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(G, X.t() @ Tg)


def fit_calib(Y: torch.Tensor, Z: torch.Tensor, mode: str, ridge: float = 1e-3):
    D = Y.shape[1]
    dev = Y.device
    I = torch.eye(D, device=dev)
    Yd, Zd = Y.double(), Z.double()
    ones = torch.ones(Yd.shape[0], 1, device=dev, dtype=torch.float64)
    if mode == "none":
        return I, torch.zeros(D, device=dev), I
    if mode == "shift":
        return I, (Zd.mean(0) - Yd.mean(0)).float(), I
    if mode == "inv":
        Wm = _lsq(torch.cat([Yd, ones], 1), Zd, ridge)
        return Wm[:D].t().contiguous().float(), Wm[D].float(), I
    if mode == "fwd":
        Wm = _lsq(torch.cat([Zd, ones], 1), Yd, ridge)
        H = Wm[:D].t().contiguous().float()
        c = Wm[D].float()
        return I, -c, H
    raise ValueError(mode)


def cov_inv(E: torch.Tensor, ridge: float):
    D = E.shape[1]
    cov = (E.double().t() @ E.double()) / E.shape[0]
    return (
        torch.linalg.inv(
            cov
            + ridge
            * cov.diagonal().mean()
            * torch.eye(D, device=E.device, dtype=torch.float64)
        ).float(),
        cov.float(),
    )


def solve_window_dense(
    C,
    z_b,
    yhat,
    obs_idx,
    Bi,
    sigma_q,
    sigma_r,
    horizon,
    constraint,
    Qi,
    Sinv_full: dict,
    H: torch.Tensor,
):
    dev = z_b.device
    B, D = z_b.shape
    n = horizon
    q, r = 1.0 / sigma_q**2, 1.0 / sigma_r**2
    C64 = C.double()
    I = torch.eye(D, device=dev, dtype=torch.float64)
    QiM = q * (Qi.double() if Qi is not None else I)
    BiM = (torch.diag(Bi) if Bi.dim() == 1 else Bi).double()
    zb = z_b.double()
    Sinv = r * Sinv_full[n].double()
    H64 = H.double()
    _obs = set(int(k) for k in obs_idx)
    Hb = torch.block_diag(
        *[H64 if t in _obs else torch.zeros_like(H64) for t in range(n)]
    )

    Yh = yhat[:, :n].reshape(B, n * D).double()
    HtS = Hb.t() @ Sinv
    A = HtS @ Hb
    rhs = Yh @ Sinv @ Hb

    A[:D, :D] += BiM
    rhs[:, :D] += zb @ BiM.t()

    if constraint == "strong":
        G = torch.zeros(n * D, D, device=dev, dtype=torch.float64)
        P = I.clone()
        for t in range(n):
            G[t * D : (t + 1) * D] = P
            P = C64 @ P
        z0 = torch.linalg.solve(G.t() @ A @ G, (rhs @ G).t()).t()
        Z = [z0]
        for _ in range(n - 1):
            Z.append(Z[-1] @ C64.t())
        return torch.stack(Z, 1).float()

    for t in range(n - 1):
        A[t * D : (t + 1) * D, t * D : (t + 1) * D] += C64.t() @ QiM @ C64
        A[(t + 1) * D : (t + 2) * D, (t + 1) * D : (t + 2) * D] += QiM
        A[t * D : (t + 1) * D, (t + 1) * D : (t + 2) * D] += -(C64.t() @ QiM)
        A[(t + 1) * D : (t + 2) * D, t * D : (t + 1) * D] += -(QiM @ C64)
    Zf = torch.linalg.solve(A, rhs.t()).t()
    return Zf.reshape(B, n, D).float()


_DECODE_ELEM_BUDGET = 32_000_000


@torch.no_grad()
def _score(model, analysis, gt, run_length):
    B = analysis.shape[0]
    per_t = B * int(np.prod(gt[0].shape[1:]))
    n_t = max(1, min(run_length, _DECODE_ELEM_BUDGET // max(per_t, 1)))
    E = torch.empty(B, run_length, device=analysis.device)
    for s in range(0, run_length, n_t):
        e = min(s + n_t, run_length)
        dec = model.decode(analysis[:, s:e].reshape(B * (e - s), -1))
        ref = torch.stack(gt[s:e], 1).reshape(B * (e - s), *gt[0].shape[1:])
        E[:, s:e] = _rrmse_batch(dec, ref).reshape(B, e - s)
    return list(E.t().cpu().numpy())


@torch.no_grad()
def assimilate(
    model,
    runs,
    sampler,
    Bi,
    sq,
    sr,
    *,
    window_size,
    spinup,
    run_length,
    obs_every,
    constraint,
    Qi,
    Ri,
    A,
    b,
    H,
    Sinv_full=None,
    return_analysis: bool = False,
):
    dev = model.device
    B = len(runs)
    gt = [
        torch.stack([runs[i][t] for i in range(B)]).to(dev)
        for t in range(run_length + 1)
    ]
    if spinup >= run_length:
        yhat = torch.zeros(B, run_length + 1, model.D, device=dev)
    else:
        y = feature_obs(model, gt, sampler, run_length, obs_every)
        yhat = y @ A.t() + b

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
        yw = yhat[:, cursor : cursor + Wn + 1]
        Z = solve_window_dense(
            model.C, z_b, yw, obs_idx, Bi, sq, sr, Wn + 1, constraint, Qi, Sinv_full, H
        )
        analysis[:, cursor : cursor + Wn + 1] = Z
        z_b = Z[:, Wn]
        cursor += Wn
    if spinup >= run_length:
        errs = _score(model, analysis, gt, run_length)
    else:
        errs = [
            _rrmse_batch(model.decode(analysis[:, t]), gt[t]).cpu().numpy()
            for t in range(run_length)
        ]
    if return_analysis:
        return np.stack(errs, 1), yhat, gt, analysis.detach().cpu().numpy()
    return np.stack(errs, 1), yhat, gt
