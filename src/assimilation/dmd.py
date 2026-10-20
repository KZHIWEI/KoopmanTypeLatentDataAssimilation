from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from src.config import load_config, build_ae_config
from src.assimilation.variational import _rrmse_batch
from src.assimilation.observation import MaskSampler
from src.assimilation.tensorvar import solve_window_closed

_FP64_COPY_BUDGET = 2 * 2**30


class DMDDA:

    def __init__(
        self, cfg, dataset: str, device: torch.device, r: int = 0, max_pairs: int = 6000
    ):
        from src.data import _load_trajs, channel_norm

        if r <= 0:
            r = build_ae_config(load_config("config/kae.yaml", dataset)).latent_dim
        p = cfg["data_path"]
        tr = _LazyH5(p) if p.endswith(".h5") else _load_trajs(p)
        r = min(r, int(np.prod(tr.shape[2:])))
        mean, std = channel_norm(np.asarray(tr[:, :: max(1, tr.shape[1] // 200)]), cfg)
        skip = int(cfg.get("skip", 1))
        ntraj, T = tr.shape[0], tr.shape[1]
        per = max(2, max_pairs // ntraj)
        ts = np.linspace(0, T - skip - 1, per).astype(int)

        def prep(frames):
            A = (frames.astype(np.float32) - mean.reshape(1, -1, 1, 1)) / std.reshape(
                1, -1, 1, 1
            )
            return torch.tensor(A.reshape(A.shape[0], -1).T, device=device)

        X = torch.cat([prep(np.asarray(tr[i, ts])) for i in range(ntraj)], dim=1)
        Xp = torch.cat(
            [prep(np.asarray(tr[i, ts + skip])) for i in range(ntraj)], dim=1
        )
        _need = X.shape[0] * X.shape[1] * 8
        _chunked = _need > _FP64_COPY_BUDGET

        def _gram(M):
            if not _chunked:
                return M.t().double() @ M.double()
            out = torch.zeros(
                M.shape[1], M.shape[1], dtype=torch.float64, device=M.device
            )
            step = max(1, _FP64_COPY_BUDGET // (M.shape[1] * 8))
            for i in range(0, M.shape[0], step):
                b = M[i : i + step].double()
                out += b.t() @ b
                del b
            return out

        if _chunked:
            print(f"[dmd] chunked gram {_need / 2**30:.2f} GiB", flush=True)
        G = _gram(X)
        evals, V = torch.linalg.eigh(G)
        self.evals_all = evals.detach().sort(descending=True).values.clone()
        idx = torch.argsort(evals, descending=True)[:r]
        S = evals[idx].clamp(min=1e-10).sqrt()
        V = V[:, idx]
        if not _chunked:
            U = (X.double() @ V) / S
            A = U.t() @ Xp.double() @ V / S
        else:
            step = max(1, _FP64_COPY_BUDGET // (V.shape[1] * 8))
            U = torch.empty(
                X.shape[0], V.shape[1], dtype=torch.float32, device=X.device
            )
            for i in range(0, X.shape[0], step):
                U[i : i + step] = ((X[i : i + step].double() @ V) / S).float()
            del X
            torch.cuda.empty_cache()
            A = torch.zeros(
                V.shape[1], V.shape[1], dtype=torch.float64, device=U.device
            )
            for i in range(0, Xp.shape[0], step):
                A += U[i : i + step].t().double() @ (Xp[i : i + step].double() @ V)
            A /= S
        self.U = U.float().contiguous()
        del U
        self.C = A.float().contiguous()
        self.D = r
        self.device = device
        self.shape = None
        del Xp, G
        if not _chunked:
            del X
        torch.cuda.empty_cache()
        spec = float(torch.linalg.eigvals(self.C.double().cpu()).abs().max())
        print(f"[dmd] fit: r={r} pairs={V.shape[0]} spec_radius={spec:.4f}", flush=True)

    def encode(self, x):
        self.shape = tuple(x.shape[1:])
        return x.to(self.device).reshape(x.shape[0], -1) @ self.U

    def decode(self, z):
        return (z @ self.U.t()).reshape(z.shape[0], *self.shape)

    def step(self, z):
        return z @ self.C.t()


@torch.no_grad()
def proj_frame(
    model: DMDDA, x: torch.Tensor, mask: torch.Tensor, lam: float = 1e-3
) -> torch.Tensor:
    C = x.shape[0]
    HW = model.U.shape[0] // C
    pix = mask.reshape(-1).bool()
    rows = torch.cat([pix.nonzero(as_tuple=True)[0] + c * HW for c in range(C)])
    Uo = model.U[rows].double()
    xo = x.reshape(-1)[rows].double()
    H = Uo.t() @ Uo
    H += (
        lam
        * H.diagonal().mean()
        * torch.eye(model.D, device=H.device, dtype=torch.float64)
    )
    return torch.linalg.solve(H, Uo.t() @ xo).float()


def hist_indices(t: int, span: int, obs_every: int) -> List[int]:
    idx = [t - k * obs_every for k in range(-(-span // obs_every))]
    return [max(0, i) for i in reversed(idx)]


@torch.no_grad()
def feature_obs_dmd(
    model: DMDDA,
    gt: List[torch.Tensor],
    sampler: MaskSampler,
    run_length: int,
    obs_every: int,
    lam: float = 1e-3,
    hist_m: int = 1,
    hist_w: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B = gt[0].shape[0]
    r = model.D
    dev = model.device
    ys = torch.zeros(B, run_length + 1, r, device=dev)
    for t in range(run_length + 1):
        if t % obs_every != 0:
            continue
        for b in range(B):
            if hist_w is None:
                ys[b, t] = proj_frame(
                    model, gt[t][b].to(dev), sampler.mask(b, t).to(dev), lam
                )
                continue
            idx = hist_indices(t, hist_m, obs_every)
            f = torch.stack(
                [
                    proj_frame(model, gt[i][b].to(dev), sampler.mask(b, i).to(dev), lam)
                    for i in idx
                ],
                dim=1,
            )
            ys[b, t] = (f * hist_w).sum(dim=1)
    return ys


class _LazyH5:

    def __init__(self, path):
        import h5py
        from src.data import _abspath

        self._f = h5py.File(_abspath(path), "r")
        self._d = self._f["data"]
        T, H, W, C = self._d.shape
        self.shape = (1, T, C, H, W)

    @staticmethod
    def _cf(d):
        return np.transpose(d, (0, 3, 1, 2))

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError(key)
        a, b = key
        if isinstance(a, slice) and isinstance(b, slice):
            return self._cf(self._d[b])[None]
        if isinstance(a, (int, np.integer)):
            if a != 0:
                raise IndexError(a)
            idx = list(b) if not isinstance(b, (int, np.integer)) else [int(b)]
            if any(y <= x for x, y in zip(idx, idx[1:])):
                raise IndexError(idx)
            return self._cf(self._d[idx])
        raise TypeError(key)

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass


@torch.no_grad()
def estimate_QR_dmd(
    model: DMDDA, cfg, sampler: MaskSampler, *, n_windows: int = 400, ridge: float = 0.1
):
    from src.data import load_split, train_norm

    tr = load_split(cfg["data_path"])
    mean, std = train_norm(cfg)
    skip = int(cfg.get("skip", 1))
    ntraj, T = tr.shape[0], tr.shape[1]
    rng = np.random.RandomState(0)
    D = model.D
    eQ, eR = [], []
    per = max(1, n_windows // ntraj)
    for ti in range(ntraj):
        for t0 in rng.randint(skip, T - skip - 1, size=per):
            w = (
                np.asarray(tr[ti, [t0, t0 + skip]]).astype(np.float32)
                - mean.reshape(1, -1, 1, 1)
            ) / std.reshape(1, -1, 1, 1)
            w = torch.tensor(w, device=model.device)
            z_t = model.encode(w[0:1]).float()
            z_n = model.encode(w[1:2]).float()
            eQ.append((z_n - model.step(z_t)).squeeze(0))
            y = feature_obs_dmd(model, [w[0:1]], sampler, 0, 1)[0, 0]
            eR.append(y - z_t.squeeze(0))
    EQ = torch.stack(eQ).double()
    ER = torch.stack(eR).double()
    if EQ.shape[0] < D:
        print(f"[QR] rank {EQ.shape[0]} < {D}", flush=True)

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
        f"trQ={float(EQ.var(0).mean()):.4g} trR={float(ER.var(0).mean()):.4g}",
        flush=True,
    )
    return Qi.to(model.device), Ri.to(model.device)


@torch.no_grad()
def assimilate_runs_dmd(
    model: DMDDA,
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
    hist_m: int = 1,
    hist_w: Optional[torch.Tensor] = None,
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
        y = feature_obs_dmd(
            model, gt, sampler, run_length, obs_every, hist_m=hist_m, hist_w=hist_w
        )

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
    arr = assimilate_runs_dmd(model, runs, sampler, Bi, sq, sr, **kw)
    spin = int(kw.get("spinup", 0))
    return float(arr[:, spin:].mean()), arr.mean(0)
