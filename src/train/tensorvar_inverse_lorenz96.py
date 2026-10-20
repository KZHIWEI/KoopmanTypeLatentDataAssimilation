from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch

from src.config import load_config, build_ae_config
from src.assimilation.variational import nmc_background
from src.model.tensorvar import TensorVarForward, TensorVarInverse


class L96Seq:

    def __init__(self, cfg, path_key="data_path"):
        from src.data import _load_trajs, channel_norm

        tr = np.asarray(_load_trajs(cfg["data_path"]))
        mean, std = channel_norm(tr[:, :: max(1, tr.shape[1] // 200)], cfg)
        src = np.asarray(_load_trajs(cfg[path_key]))
        self.x = (
            (src - mean.reshape(1, 1, -1, 1, 1)) / std.reshape(1, 1, -1, 1, 1)
        ).astype(np.float32)
        self.ntraj, self.T = self.x.shape[0], self.x.shape[1]
        self.shape = self.x.shape[2:]

    def batch(self, rng, n, length, dev):
        ti = rng.randint(0, self.ntraj, size=n)
        t0 = rng.randint(0, self.T - length, size=n)
        out = np.stack([self.x[a, b : b + length] for a, b in zip(ti, t0)])
        return torch.from_numpy(out).to(dev)


def rand_masks(rng, n, nfr, H, W, nobs, dev):
    npix = H * W
    m = np.zeros((n, nfr, npix), dtype=np.float32)
    for i in range(n):
        for j in range(nfr):
            m[i, j, rng.choice(npix, size=nobs, replace=False)] = 1.0
    return torch.from_numpy(m).to(dev).view(n, nfr, 1, H, W)


def build_hist(obs, masks, m, steps, mask_aware):
    B, nfr = obs.shape[:2]
    hs = []
    for j in range(steps):
        i = nfr - steps + j
        h = obs[:, i - m + 1 : i + 1].reshape(B, -1, *obs.shape[-2:])
        if mask_aware:
            h = torch.cat(
                [h, masks[:, i - m + 1 : i + 1].reshape(B, -1, *obs.shape[-2:])], 1
            )
        hs.append(h)
    return torch.stack(hs, 1)


def build_A(C, Bi, Qi, Ri, H, n, sq, sr):
    D = C.shape[0]
    dev = C.device
    A = torch.zeros(n * D, n * D, device=dev, dtype=torch.float64)
    C64 = C.double()
    QiM = (1.0 / sq**2) * Qi.double()
    RiM = (1.0 / sr**2) * Ri.double()
    BiM = (torch.diag(Bi) if Bi.dim() == 1 else Bi).double()
    H64 = H.double()
    HtR = H64.t() @ RiM
    for t in range(n):
        A[t * D : (t + 1) * D, t * D : (t + 1) * D] += HtR @ H64
    A[:D, :D] += BiM
    for t in range(n - 1):
        A[t * D : (t + 1) * D, t * D : (t + 1) * D] += C64.t() @ QiM @ C64
        A[(t + 1) * D : (t + 2) * D, (t + 1) * D : (t + 2) * D] += QiM
        A[t * D : (t + 1) * D, (t + 1) * D : (t + 2) * D] += -(C64.t() @ QiM)
        A[(t + 1) * D : (t + 2) * D, t * D : (t + 1) * D] += -(QiM @ C64)
    return A, HtR, BiM


def solve_batch(A_lu, HtR, BiM, y, c, z_b, D, n):
    B = y.shape[0]
    rhs = ((y.double().reshape(B, n, D) - c.double()) @ HtR.t()).reshape(B, n * D)
    rhs = rhs.clone()
    rhs[:, :D] = rhs[:, :D] + z_b.double() @ BiM.t()
    Z = torch.linalg.lu_solve(*A_lu, rhs.t().contiguous()).t()
    return Z.reshape(B, n, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lorenz96_obs30")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config("config/da.yaml", args.dataset)
    tvcfg = load_config("config/tensorvar.yaml", args.dataset)
    ksteps = int(tvcfg.get("koopman_steps", 5))
    dev = torch.device("cuda")
    from src.config import ckpt_path

    root = os.path.dirname(ckpt_path(tvcfg, "tensorvar", "_"))

    ckf = torch.load(
        os.path.join(root, "forward.pt"), map_location="cpu", weights_only=False
    )
    fcfg = ckf["cfg"]
    fwd = TensorVarForward(build_ae_config(fcfg))
    fwd.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ckf["state_dict"].items()}
    )
    fwd = fwd.to(dev).eval()
    for p in fwd.parameters():
        p.requires_grad_(False)
    C = fwd.C_forward.detach().t().to(dev)
    D = C.shape[0]

    icfg = dict(fcfg)
    icfg["obs_ratio"] = float(cfg["obs_ratio"])
    icfg["mask_aware"] = True
    icfg["history_len"] = int(fcfg.get("history_len", 10))
    inv = TensorVarInverse(
        build_ae_config(icfg),
        fwd,
        history_len=icfg["history_len"],
        obs_ratio=icfg["obs_ratio"],
        mask_aware=True,
    ).to(dev)
    cki = torch.load(
        os.path.join(root, "inverse_encoder_pretrained.pt"),
        map_location="cpu",
        weights_only=False,
    )
    sd = {k.replace("_orig_mod.", ""): v for k, v in cki["state_dict"].items()}
    inv.load_state_dict(sd, strict=False)
    for p in inv.K_O.parameters():
        p.requires_grad_(True)
    m = inv.history_len
    dens = round(float(cfg["obs_ratio"]) * 100)
    keep = 3000 if dens >= 30 else 6000
    out = os.path.join(root, f"inverse_encoder@{dens}.pt")

    ds = L96Seq(cfg)
    H, W = ds.shape[-2:]
    nobs = int(round(float(cfg["obs_ratio"]) * H * W))
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    print(
        f"[data] train {ds.x.shape}  nobs={nobs}/{H*W}  D={D}  ksteps={ksteps}",
        flush=True,
    )

    class _M:
        device = dev

        def encode(self, x):
            return fwd.encode(x.to(dev))

        def decode(self, z):
            return fwd.decode(z)

        def step(self, z):
            return fwd.step(z)

    skip = int(cfg["skip"])
    lead = int(cfg["nmc_lead"])
    _r = np.random.RandomState(int(cfg.get("mask_seed", 0)))
    lo = 2 * lead * skip
    samples = [
        (int(_r.randint(0, ds.ntraj)), int(_r.randint(lo, ds.T)))
        for _ in range(int(cfg["nmc_samples"]))
    ]
    frames_at = lambda t, f: torch.from_numpy(ds.x[t, f]).unsqueeze(0)
    Bi = nmc_background(
        _M(),
        frames_at,
        samples,
        lead,
        skip,
        diagonal=bool(cfg.get("b_diagonal", False)),
        ridge=float(cfg["b_ridge"]),
    ).to(dev)

    @torch.no_grad()
    def est_Q(nb=40, bs=256, L=8):
        E = []
        for _ in range(nb):
            w = ds.batch(rng, bs, L, dev)
            z = fwd.encode(w.reshape(-1, *ds.shape)).reshape(bs, L, D).float()
            E.append((z[:, 1:] - z[:, :-1] @ C.t()).reshape(-1, D))
        E = torch.cat(E).double()
        cov = (E.t() @ E) / E.shape[0]
        return torch.linalg.inv(
            cov
            + 0.01
            * cov.diagonal().mean()
            * torch.eye(D, device=dev, dtype=torch.float64)
        ).float()

    @torch.no_grad()
    def est_HR(nb=24, bs=192):
        Ys, Zs = [], []
        for _ in range(nb):
            w = ds.batch(rng, bs, m + 1, dev)
            mk = rand_masks(rng, bs, m, H, W, nobs, dev)
            obs = w[:, :m] * mk
            h = build_hist(obs, mk, m, 1, True)[:, 0]
            Ys.append(inv.observe(h).float())
            Zs.append(fwd.encode(w[:, m - 1]).float())
        Y = torch.cat(Ys).double()
        Z = torch.cat(Zs).double()
        X = torch.cat(
            [Z, torch.ones(Z.shape[0], 1, device=dev, dtype=torch.float64)], 1
        )
        G = X.t() @ X
        G = G + 1e-3 * G.diagonal().mean() * torch.eye(
            D + 1, device=dev, dtype=torch.float64
        )
        Wm = torch.linalg.solve(G, X.t() @ Y)
        Hm = Wm[:D].t().contiguous()
        c = Wm[D]
        E = Y - Z @ Hm.t() - c
        cov = (E.t() @ E) / E.shape[0]
        Ri = torch.linalg.inv(
            cov
            + 0.01
            * cov.diagonal().mean()
            * torch.eye(D, device=dev, dtype=torch.float64)
        ).float()
        return Ri, Hm.float(), c.float()

    Qi = est_Q()
    Ri, Hm, c = est_HR()
    n = ksteps + 1
    print(f"[cov] n={n}", flush=True)

    opt = torch.optim.Adam(inv.K_O.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6000, eta_min=1e-6)
    A, HtR, BiM = build_A(C, Bi, Qi, Ri, Hm, n, 1.0, 1.0)
    A_lu = torch.linalg.lu_factor(A)

    Lmax = 8
    seg = Lmax + (m - 1) + n
    t0 = time.time()
    for it in range(1, 6000 + 1):
        if it % 400 == 0:
            inv.eval()
            Ri, Hm, c = est_HR()
            inv.train()
            A, HtR, BiM = build_A(C, Bi, Qi, Ri, Hm, n, 1.0, 1.0)
            A_lu = torch.linalg.lu_factor(A)
        w = ds.batch(rng, 48, seg, dev)
        B = w.shape[0]
        mk = rand_masks(rng, B, seg, H, W, nobs, dev)
        obs = w * mk
        hist = build_hist(obs, mk, m, n, True)
        y = inv.observe(hist.reshape(B * n, *hist.shape[2:])).reshape(B, n, D).float()
        with torch.no_grad():
            lead = int(rng.randint(4, 8 + 1))
            i_a = seg - n
            z_b = fwd.encode(w[:, i_a - lead]).float()
            for _ in range(lead):
                z_b = z_b @ C.t()
            s = w[:, i_a:].reshape(B * n, *ds.shape)
        Z = solve_batch(A_lu, HtR, BiM, y, c, z_b, D, n).float()
        dec = fwd.decode(Z.reshape(B * n, D))
        loss = ((dec - s) ** 2).mean() / (s**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inv.K_O.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 100 == 0:
            print(
                f"  it {it:5d}  loss {float(loss):.4f}  lr {sched.get_last_lr()[0]:.2e} "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
        if it == keep:
            torch.save({"state_dict": inv.state_dict(), "cfg": icfg}, out)
            print(f"  [save] {out}", flush=True)


if __name__ == "__main__":
    main()
