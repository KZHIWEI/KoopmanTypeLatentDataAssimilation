from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

_AMP = dict(device_type="cuda", dtype=torch.bfloat16)


class LatentDAModel(ABC):

    device: torch.device

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def step(self, z: torch.Tensor) -> torch.Tensor: ...

    def linear_operator(self) -> Optional[torch.Tensor]:
        return None


def rrmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred = pred.detach().float()
    gt = gt.detach().float()
    return float(
        torch.sqrt(((pred - gt) ** 2).mean() / (gt**2).mean().clamp(min=1e-12))
    )


def _rrmse_batch(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    gt = gt.float()
    num = ((pred - gt) ** 2).mean(dim=(1, 2, 3))
    den = (gt**2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
    return torch.sqrt(num / den)


def inv_reg(cov: torch.Tensor, ridge: float = 0.05) -> torch.Tensor:
    d = cov.shape[0]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    return torch.linalg.inv(cov + ridge * torch.diag(cov).mean() * eye)


def _b_quadratic(e: torch.Tensor, Bi: torch.Tensor) -> torch.Tensor:
    e = e.float()
    if Bi.dim() == 1:
        return (e**2 * Bi).sum()
    return ((e @ Bi) * e).sum()


def _step_seq(model: LatentDAModel, Zseq: torch.Tensor) -> torch.Tensor:
    b, l, d = Zseq.shape
    return model.step(Zseq.reshape(b * l, d)).reshape(b, l, d)


def _rollout_b(model: LatentDAModel, z0: torch.Tensor, horizon: int) -> torch.Tensor:
    zs = [z0]
    for _ in range(horizon - 1):
        zs.append(model.step(zs[-1]))
    return torch.stack(zs, dim=1)


def nmc_background(
    model: LatentDAModel,
    frames_at: Callable[[int, int], torch.Tensor],
    samples: Sequence[Tuple[int, int]],
    lead: int,
    skip: int,
    diagonal: bool = False,
    ridge: float = 0.05,
    center: bool = False,
) -> torch.Tensor:
    diffs = []
    with torch.no_grad(), torch.autocast(**_AMP):
        for traj, frame in tqdm(samples, desc="NMC background", leave=False):
            z_short = model.encode(frames_at(traj, frame - lead * skip))
            for _ in range(lead):
                z_short = model.step(z_short)
            z_long = model.encode(frames_at(traj, frame - 2 * lead * skip))
            for _ in range(2 * lead):
                z_long = model.step(z_long)
            diffs.append((z_long - z_short)[0].float())
    D = torch.stack(diffs).double()
    if center:
        D = D - D.mean(0)
    if diagonal:
        var = D.var(dim=0) + ridge * D.var(dim=0).mean()
        return (1.0 / var).float()
    cov = (D.t() @ D) / D.shape[0]
    return inv_reg(cov, ridge).float()


def solve_window(
    model: LatentDAModel,
    z_b: torch.Tensor,
    obs_idx: List[int],
    masks: Optional[torch.Tensor],
    obs: Optional[torch.Tensor],
    Bi: torch.Tensor,
    sigma_q: float,
    sigma_r: float,
    horizon: int,
    iters: int = 400,
    lr: float = 0.05,
    constraint: str = "weak",
    warm_start: bool = False,
    converged: bool = False,
    conv_iters: int = 1600,
) -> torch.Tensor:
    if converged:
        warm_start = True

    def obs_cost(Z: torch.Tensor) -> torch.Tensor:
        if not obs_idx:
            return Z.new_zeros(())
        b, n = Z.shape[0], len(obs_idx)
        dec = model.decode(Z[:, obs_idx].reshape(b * n, -1)).reshape(
            b, n, *obs.shape[2:]
        )
        return ((masks * dec - obs) ** 2).sum() / sigma_r**2

    def solve_strong() -> torch.Tensor:
        z0 = z_b.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([z0], lr=lr)
        for _ in range(iters):
            with torch.autocast(**_AMP):
                Z = _rollout_b(model, z0, horizon)
                cost = _b_quadratic(z0 - z_b, Bi) + obs_cost(Z)
            opt.zero_grad()
            cost.backward()
            opt.step()
        with torch.no_grad(), torch.autocast(**_AMP):
            return _rollout_b(model, z0, horizon).detach()

    if constraint == "strong":
        return solve_strong()

    if warm_start:
        Z0 = solve_strong()
    else:
        with torch.no_grad(), torch.autocast(**_AMP):
            Z0 = _rollout_b(model, z_b, horizon)
    Z = Z0.float().clone().detach().requires_grad_(True)
    n_it = conv_iters if converged else iters
    opt = torch.optim.Adam([Z], lr=lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_it, eta_min=0.0)
        if converged
        else None
    )
    amp = dict(device_type=_AMP["device_type"], enabled=False) if converged else _AMP
    for _ in range(n_it):
        with torch.autocast(**amp):
            model_err = (
                (Z[:, 1:] - _step_seq(model, Z[:, :-1])) ** 2
            ).sum() / sigma_q**2
            cost = _b_quadratic(Z[:, 0] - z_b, Bi) + model_err + obs_cost(Z)
        opt.zero_grad()
        cost.backward()
        opt.step()
        if sched is not None:
            sched.step()
    return Z.detach()


def assimilate_runs(
    model: LatentDAModel,
    runs: Sequence[List[torch.Tensor]],
    sampler,
    Bi: torch.Tensor,
    sigma_q: float,
    sigma_r: float,
    *,
    window_size: int,
    spinup: int,
    run_length: int,
    obs_every: int,
    iters: int = 400,
    lr: float = 0.05,
    constraint: str = "weak",
    warm_start: bool = False,
    converged: bool = False,
    conv_iters: int = 1600,
    channelwise: bool = False,
    return_analysis: bool = False,
) -> np.ndarray:
    dev = model.device
    B = len(runs)
    gt = [
        torch.stack([runs[b][t] for b in range(B)]).to(dev)
        for t in range(run_length + 1)
    ]

    with torch.no_grad(), torch.autocast(**_AMP):
        z = model.encode(gt[0]).float()
    D = z.shape[1]
    analysis = torch.zeros(B, run_length + 1, D, device=dev)

    with torch.no_grad(), torch.autocast(**_AMP):
        for t in range(min(spinup, run_length + 1)):
            analysis[:, t] = z
            z = model.step(z).float()
    z_b = z.detach()

    cursor = min(spinup, run_length)
    while cursor < run_length:
        W = min(window_size, run_length - cursor)
        obs_idx = [k for k in range(W + 1) if (cursor + k) % obs_every == 0]
        if obs_idx:
            masks = torch.stack(
                [
                    torch.stack([sampler.mask(b, cursor + k) for k in obs_idx])
                    for b in range(B)
                ]
            ).to(dev)
            obs = torch.stack([gt[cursor + k] for k in obs_idx], dim=1) * masks
        else:
            masks = obs = None
        Z = solve_window(
            model,
            z_b,
            obs_idx,
            masks,
            obs,
            Bi,
            sigma_q,
            sigma_r,
            W + 1,
            iters=iters,
            lr=lr,
            constraint=constraint,
            warm_start=warm_start,
            converged=converged,
            conv_iters=conv_iters,
        )
        analysis[:, cursor : cursor + W] = Z[:, :W]
        analysis[:, cursor + W] = Z[:, W]
        z_b = Z[:, W].detach()
        cursor += W

    errs = []
    with torch.no_grad(), torch.autocast(**_AMP):
        for t in range(run_length):
            dec = model.decode(analysis[:, t])
            if channelwise:
                num = ((dec.float() - gt[t]) ** 2).mean(dim=(2, 3))
                den = (gt[t] ** 2).mean(dim=(2, 3)).clamp(min=1e-12)
                errs.append(torch.sqrt(num / den).cpu().numpy())
            else:
                errs.append(_rrmse_batch(dec, gt[t]).cpu().numpy())
    if return_analysis:
        return np.stack(errs, axis=1), analysis.detach().cpu().numpy()
    return np.stack(errs, axis=1)


def score_runs(
    model: LatentDAModel,
    runs: Sequence[List[torch.Tensor]],
    sampler,
    Bi: torch.Tensor,
    sigma_q: float,
    sigma_r: float,
    **run_kw,
) -> Tuple[float, np.ndarray]:
    arr = assimilate_runs(model, runs, sampler, Bi, sigma_q, sigma_r, **run_kw)
    spinup = int(run_kw.get("spinup", 0))
    return float(arr[:, spinup:].mean()), arr.mean(0)


def run_sweep(
    model: LatentDAModel,
    runs: Sequence[List[torch.Tensor]],
    sampler,
    Bi: torch.Tensor,
    *,
    var: str,
    grid: Sequence[float],
    sr_fixed: float = 0.05,
    sq_fixed: float = 1.0,
    **run_kw,
) -> Dict:
    rows = []
    for val in tqdm(grid, desc=f"{var} sweep"):
        sr, sq = (val, sq_fixed) if var == "sigma_r" else (sr_fixed, val)
        score, _ = score_runs(model, runs, sampler, Bi, sq, sr, **run_kw)
        rows.append((sr, sq, score))
        tqdm.write(f"  {var}={val:<10g} -> RRMSE {score:.4f}")
    best = min(rows, key=lambda r: r[2])
    return {
        "var": var,
        "grid": rows,
        "best_sr": best[0],
        "best_sq": best[1],
        "best": best[2],
    }


def write_sweep_csv(path: str, result: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma_r", "sigma_q", "rrmse"])
        for sr, sq, v in result["grid"]:
            w.writerow([sr, sq, f"{v:.6f}"])
        w.writerow([])
        w.writerow(
            [
                f"best_{result['var']}",
                result["best_sr"] if result["var"] == "sigma_r" else result["best_sq"],
                f"{result['best']:.6f}",
            ]
        )
