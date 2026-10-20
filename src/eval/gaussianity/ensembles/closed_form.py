from __future__ import annotations

import os

import numpy as np
import torch

from src.assimilation.variational import nmc_background
from src.eval.gaussianity.ensembles import common as C


def _tv(ds, members, draw, density, cfg, data, runs, sampler, geom, dev):
    from src.assimilation.tensorvar import feature_obs, solve_window_closed
    from src.assimilation.tensorvar_feature import solve_window_dense
    from src.eval.benchmark import TV_DRIVER, tv_build, tv_checkpoint_paths

    D = TV_DRIVER[ds]
    feature_space = D["kind"] == "feature_space"
    paths = tv_checkpoint_paths(ds, density)
    sr, sq = D["sig"][density]
    W, spin, oev, rl = (
        geom["window"],
        geom["spinup"],
        geom["obs_every"],
        geom["run_length"],
    )
    samples = data.nmc_samples(
        int(cfg["nmc_lead"]),
        int(cfg["skip"]),
        int(cfg["nmc_samples"]),
        seed=int(cfg.get("mask_seed", 0)),
    )
    model, Bi, _, cov = tv_build(
        D, None, density, cfg, data, sampler, samples, oev, W, dev, paths=paths
    )
    Qi, Ri = cov["Qi"], cov["Ri"]
    Bchol, Bnp = C.chol_from_precision(Bi, dev)
    out, refs = [], []
    for si in range(len(runs)):
        gt = [runs[si][t].unsqueeze(0).to(dev) for t in range(rl + 1)]
        with torch.no_grad():
            y1 = feature_obs(model, gt, sampler, rl, oev)
            if feature_space:
                y1 = y1 @ cov["A"].t() + cov["b"]
            z_true = model.encode(gt[min(spin, rl)]).float()[0]
        eps = draw.bg(members, z_true.shape[0]).to(dev) @ Bchol.T
        z_b = z_true.unsqueeze(0) + eps
        cursor = min(spin, rl)
        Wn = min(W, rl - cursor)
        obs_idx = [k for k in range(Wn + 1) if (cursor + k) % oev == 0]
        yw = y1[:, cursor : cursor + Wn + 1].expand(members, -1, -1).contiguous()
        constraint = cfg.get("constraint", "weak")
        with torch.no_grad():
            if feature_space:
                Z = solve_window_dense(
                    model.C,
                    z_b,
                    yw,
                    obs_idx,
                    Bi,
                    sq,
                    sr,
                    Wn + 1,
                    constraint,
                    Qi,
                    cov["Sinv_full"],
                    cov["H"],
                )
            else:
                Z = solve_window_closed(
                    model.C,
                    z_b,
                    yw,
                    obs_idx,
                    Bi,
                    sq,
                    sr,
                    Wn + 1,
                    constraint,
                    Qi=Qi,
                    Ri=Ri,
                )
        out.append(
            (
                eps.cpu().numpy().astype(np.float64),
                (Z[:, 0] - z_true.unsqueeze(0)).cpu().numpy().astype(np.float64),
            )
        )
        refs.append(float(z_true.pow(2).mean().sqrt()))
    meta = {
        "tv_kind": D["kind"],
        "tv_fwd": D["fwd"],
        "tv_inv": D["inv"].format(d=density),
        "sigma_r": float(sr),
        "sigma_q": float(sq),
        "reference_rms": float(np.mean(refs)),
    }
    return out, Bnp, meta


def run_tensorvar(ds, *, members, seed, density, **_):
    dev = C.device()
    cfg, data, runs, sampler, geom = C.load_protocol(ds, density)
    draw = C.Draw(seed)
    out, Bnp, meta = _tv(
        ds, members, draw, density, cfg, data, runs, sampler, geom, dev
    )
    meta.update(geometry=geom, latent_dim=int(Bnp.shape[0]))
    return out, Bnp, meta


def run_dmd(ds, *, members, seed, density, **_):
    from src.assimilation.dmd import DMDDA, feature_obs_dmd, estimate_QR_dmd
    from src.assimilation.tensorvar import solve_window_closed

    dev = C.device()
    cfg, data, runs, sampler, geom = C.load_protocol(ds, density)
    draw = C.Draw(seed)
    model = DMDDA(cfg, f"{ds}_obs{density}", dev, r=0)
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
    Qi, Ri = estimate_QR_dmd(model, cfg, sampler)
    from src.eval.covariance_selection import dmd_sigmas

    sr, sq, source = dmd_sigmas(ds, density)
    Bchol, Bnp = C.chol_from_precision(Bi, dev)
    W, spin, oev, rl = (
        geom["window"],
        geom["spinup"],
        geom["obs_every"],
        geom["run_length"],
    )

    out, refs = [], []
    for si in range(len(runs)):
        gt = [runs[si][t].unsqueeze(0).to(dev) for t in range(rl + 1)]
        with torch.no_grad():
            y1 = feature_obs_dmd(model, gt, sampler, rl, oev)
            z_true = model.encode(gt[min(spin, rl)]).float()[0]
        eps = draw.bg(members, z_true.shape[0]).to(dev) @ Bchol.T
        z_b = z_true.unsqueeze(0) + eps
        cursor = min(spin, rl)
        Wn = min(W, rl - cursor)
        obs_idx = [k for k in range(Wn + 1) if (cursor + k) % oev == 0]
        yb = y1[:, cursor : cursor + Wn + 1].expand(members, -1, -1).contiguous()
        with torch.no_grad():
            Z = solve_window_closed(
                model.C,
                z_b,
                yb,
                obs_idx,
                Bi,
                sq,
                sr,
                Wn + 1,
                cfg.get("constraint", "weak"),
                Qi=Qi,
                Ri=Ri,
            )
        out.append(
            (
                eps.cpu().numpy().astype(np.float64),
                (Z[:, 0] - z_true.unsqueeze(0)).cpu().numpy().astype(np.float64),
            )
        )
        refs.append(float(z_true.pow(2).mean().sqrt()))
    meta = {
        "sigma_r": sr,
        "sigma_q": sq,
        "sigma_source": source,
        "reference_rms": float(np.mean(refs)),
        "geometry": geom,
        "latent_dim": int(Bnp.shape[0]),
    }
    return out, Bnp, meta
