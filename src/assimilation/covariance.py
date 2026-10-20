import numpy as np
import torch


@torch.no_grad()
def collect_QR(
    model,
    cfg,
    sampler,
    obs_every,
    n_samples=8000,
    yshift=0,
    gap=0,
    batch=32,
    per_cap=25,
):
    from src.data import load_split, train_norm

    tr = load_split(cfg["data_path"])
    mean, std = train_norm(cfg)
    mean = mean.reshape(1, -1, 1, 1)
    std = std.reshape(1, -1, 1, 1)
    skip = int(cfg.get("skip", 1))
    m = model.history_len
    ntraj, T = tr.shape[0], tr.shape[1]
    rng = np.random.RandomState(0)
    ma = bool(getattr(model.inv, "mask_aware", False))

    lo = (m - 1) * skip
    hi = T - skip * (1 + yshift) - 1
    per = max(1, per_cap) if per_cap else max(1, n_samples // ntraj)
    ntraj_use = min(ntraj, max(1, int(np.ceil(n_samples / per))))
    eQ, eR = [], []
    for ti in range(ntraj_use):
        t0s = rng.randint(lo, hi, size=per)
        t0s = t0s - ((t0s // skip) % obs_every - gap) * skip
        t0s = np.clip(t0s, lo, hi)
        need = sorted(
            {
                int(np.clip(t0 - (m - 1 - k) * skip, 0, T - 1))
                for t0 in t0s
                for k in range(m)
            }
            | {
                int(np.clip(t0 + d, 0, T - 1))
                for t0 in t0s
                for d in (0, skip, skip * yshift)
            }
        )
        idx = {t: i for i, t in enumerate(need)}
        traj = np.asarray(tr[ti, need]).astype(np.float32)
        w = torch.tensor((traj - mean) / std, device=model.device)
        hists, zt, zn, ztgt = [], [], [], []
        for t0 in t0s:
            frames, masks = [], []
            for k in range(m):
                t_abs = t0 - (m - 1 - k) * skip
                if (t_abs // skip) % obs_every == 0:
                    msk = sampler.mask(0, int(t_abs)).to(model.device)
                    frames.append(w[idx[t_abs]] * msk)
                    masks.append(msk if msk.dim() == 3 else msk.unsqueeze(0))
                else:
                    frames.append(torch.zeros_like(w[idx[t_abs]]))
                    masks.append(torch.zeros(1, *w.shape[-2:], device=w.device))
            h = torch.cat(frames, 0)
            if ma:
                h = torch.cat([h, torch.cat(masks, 0)], 0)
            hists.append(h)
            zt.append(idx[t0])
            zn.append(idx[t0 + skip])
            ztgt.append(idx[t0 + skip * yshift])
        H = torch.stack(hists)
        for i in range(0, len(H), batch):
            y = model.inv.observe(H[i : i + batch]).float()
            z_t = model.encode(w[zt[i : i + batch]]).float()
            z_n = model.encode(w[zn[i : i + batch]]).float()
            z_g = model.encode(w[ztgt[i : i + batch]]).float()
            eQ.append((z_n - z_t @ model.C.t().float()).double().cpu())
            eR.append((y - z_g).double().cpu())
    EQ = torch.cat(eQ)
    ER = torch.cat(eR)
    print(
        f"[QR] collected n={EQ.shape[0]} yshift={yshift} gap={gap} "
        f"varQ={float(EQ.var(0).mean()):.4g} varR={float(ER.var(0).mean()):.4g}",
        flush=True,
    )
    return EQ, ER


def build_inv(E, ridge=0.1, mode="full"):
    E = (E.cuda() if torch.cuda.is_available() else E).double()
    n, D = E.shape
    if mode == "diag":
        v = E.var(0) + 1e-12
        return torch.diag(1.0 / (v + ridge * v.mean())).float()
    cov = (E.t() @ E) / n
    d = cov.diagonal().mean()
    return torch.linalg.inv(
        cov + ridge * d * torch.eye(D, device=E.device, dtype=torch.float64)
    ).float()
