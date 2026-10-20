from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.eval.convexity.hessian import background_trajectory, hvp_fn, window_inputs, term_costs
from src.eval.benchmark import setup, sigmas
from src.assimilation.variational import solve_window

OUT = "results/convexity/min_curvature.json"
TERMS = ("q_model_error", "r_observation", "TOTAL")
CLOSED = ("tensorvar", "dmd")
FAR_RADIUS = 32.0


def _squash(r):
    return r / (abs(float(r)) + 1.0)


def blocks_at(J, Z0, store=None):
    B, n, D = Z0.shape
    N = n * D
    h = hvp_fn(J, Z0)
    store = torch.device(store) if store is not None else Z0.device
    M = torch.empty((B, N, N), device=store, dtype=Z0.dtype)
    e = torch.zeros(N, device=Z0.device, dtype=Z0.dtype)
    for i in range(N):
        e.zero_()
        e[i] = 1.0
        v = e.view(1, n, D).expand(B, n, D).contiguous()
        M[:, :, i] = h(v).reshape(B, N).to(store)
    asym = torch.empty(B, device=store, dtype=M.dtype)
    for b in range(B):
        Mb = M[b]
        T = Mb.t().clone()
        asym[b] = (Mb - T).abs().amax() / Mb.abs().amax().clamp_min(1e-300)
        Mb += T
        Mb *= 0.5
        del T
    return M, asym


def eigvalsh_blocks(M, dev):
    out = []
    for b in range(M.shape[0]):
        blk = M[b].to(dev, torch.float64, copy=True)
        out.append(torch.linalg.eigvalsh(blk).cpu())
        del blk
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return torch.stack(out).to(dev)


def check_block_diagonal(J, Z0, which=0):
    h = hvp_fn(J, Z0)
    v = torch.zeros_like(Z0)
    v[which] = torch.randn_like(Z0[which])
    out = h(v)
    inb = out[which].norm().item()
    others = torch.cat([out[:which], out[which + 1 :]]).norm().item()
    return others / max(inb, 1e-30)


def run(
    ds,
    model,
    dens,
    dev,
    n_dirs,
    radius,
    seed,
    terms_wanted=TERMS,
    centre_kind="background",
    block_store="auto",
    obs_iters=None,
    far_all=False,
    spectrum_dir=None,
):
    if model in CLOSED:
        from src.eval.convexity.hessian_latent_obs import build

        terms, centre, scale, _rec = build(ds, model, dens, dev)
        from src.eval.convexity.hessian_latent_obs import _DT

        centre = centre.to(_DT)
    else:
        sr, sq = sigmas(ds, model, dens)[:2]
        cfg, m, runs, sampler, Bi, rl = setup(ds, model, dens, dev)
        z_b, obs_idx, masks, obs, W = window_inputs(m, runs, sampler, cfg, rl, dev)
        terms = term_costs(m, z_b, Bi, sq, sr, obs_idx, masks, obs)
        background = background_trajectory(m, z_b, W)
        if centre_kind == "background":
            centre = background
        else:
            centre = (
                solve_window(
                    m,
                    z_b,
                    obs_idx,
                    masks,
                    obs,
                    Bi,
                    sq,
                    sr,
                    W + 1,
                    iters=int(cfg["iters"]),
                    lr=float(cfg["lr"]),
                    constraint="strong",
                    converged=True,
                    conv_iters=int(cfg["iters"]),
                )
                .float()
                .detach()
            )
        scale = background.std(dim=1, keepdim=True, correction=0)
    B, n, D = centre.shape
    fast_obs = (
        obs_iters is not None
        and centre.shape[2] > 2 * obs_iters
        and model not in CLOSED
    )
    if block_store == "auto":
        if dev.type != "cuda":
            block_store = "cpu"
        else:
            need = B * (n * D) ** 2 * centre.element_size()
            block_store = (
                "cuda" if need < 0.35 * torch.cuda.mem_get_info(dev)[0] else "cpu"
            )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dirs = [
        torch.randn(centre.shape, generator=gen).to(dev).to(centre.dtype)
        for _ in range(n_dirs)
    ]

    leak = {
        t: check_block_diagonal(terms[t], centre + radius * scale * dirs[0])
        for t in terms_wanted
    }
    bad = {t: v for t, v in leak.items() if v > 1e-10}
    if bad:
        raise SystemExit(f"{ds}:{model} {bad}")

    out = {
        "dataset": ds,
        "model": model,
        "density": dens,
        "radius": radius,
        "n_directions_requested": n_dirs,
        "n_trajectories": B,
        "block_dim": n * D,
        "block_store": block_store,
        "seed": seed,
        "rng_device": "cpu",
        "centre": centre_kind,
        "perturbation_scaling": "background_component_std_over_time",
        "block_diagonal_leakage": leak,
        "terms": {},
    }
    for name in terms_wanted:
        t0 = time.perf_counter()
        drift_raw = 0.0
        probe_dtype = None
        for _ in range(3):
            probe = torch.randn(centre.shape, generator=gen).to(dev).to(centre.dtype)
            ha, hb = (
                hvp_fn(terms[name], centre + radius * scale * d)(probe)
                for d in (dirs[0], dirs[-1])
            )
            probe_dtype = ha.dtype
            drift_raw = max(
                drift_raw,
                float((ha - hb).abs().amax() / ha.abs().amax().clamp_min(1e-300)),
            )
        constant = len(dirs) > 1 and drift_raw <= 1e-7
        floor = float(torch.finfo(probe_dtype).eps) if probe_dtype is not None else None
        drift_is_bound = drift_raw == 0.0 and floor is not None
        drift = drift_raw
        far_drift = None
        far_sep = None
        far_hvp = None
        far_bitsame = None
        far_dtype = None
        far_floor = None
        far_raw = None
        if constant or far_all:
            c1 = centre + radius * scale * dirs[0]
            c2 = centre + radius * scale * FAR_RADIUS * dirs[0]
            far_sep = float((c2 - c1).norm() / centre.norm().clamp_min(1e-300))
            far_drift = 0.0
            far_hvp = 0.0
            far_bitsame = True
            far_dtype = None
            far_floor = None
            for _ in range(3):
                probe = (
                    torch.randn(centre.shape, generator=gen).to(dev).to(centre.dtype)
                )
                ha, hb = (
                    hvp_fn(terms[name], centre + radius * scale * f * dirs[0])(probe)
                    for f in (1.0, FAR_RADIUS)
                )
                far_hvp = max(far_hvp, float(ha.abs().amax()))
                far_bitsame = far_bitsame and bool(torch.equal(ha, hb))
                far_dtype = str(ha.dtype).replace("torch.", "")
                far_floor = float(torch.finfo(ha.dtype).eps)
                far_drift = max(
                    far_drift,
                    float((ha - hb).abs().amax() / ha.abs().amax().clamp_min(1e-300)),
                )
            del c1, c2
            far_raw = far_drift
        if constant:
            if not math.isfinite(far_drift):
                constant = False
                print(f"    [{name}] far drift {far_drift}", flush=True)
            elif far_drift > 1e-7:
                constant = False
                print(f"    [{name}] drift {drift:.1e} far {far_drift:.1e}", flush=True)
        far_is_bound = (
            (far_raw == 0.0 and far_floor is not None) if far_raw is not None else False
        )
        use_dirs = dirs[:1] if constant else dirs
        nd = len(use_dirs)
        del ha, hb, probe
        R = np.zeros((nd, B))
        LOW = np.zeros((nd, B, 3))
        ASY = np.zeros((nd, B))
        CERT = np.zeros((nd, B))
        UNC = np.zeros((nd, B))
        HI = np.zeros((nd, B))
        for k, d in enumerate(use_dirs):
            if fast_obs and name == "r_observation":
                from src.eval.convexity.obs_lanczos import ends as _obs_ends

                Zc = centre + radius * scale * d
                lo, hi, asym, _nh = _obs_ends(
                    m, Zc, masks, obs, sr, obs_idx, dev, iters=obs_iters, seed=seed
                )
                ev = None
            else:
                Mb, asym = blocks_at(
                    terms[name], centre + radius * scale * d, store=block_store
                )
                ev = eigvalsh_blocks(Mb, dev)
                asym = asym.to(dev)
                del Mb
                lo, hi = ev[:, 0], ev[:, -1]
                if spectrum_dir is not None and k == 0:
                    os.makedirs(spectrum_dir, exist_ok=True)
                    fp = f"{spectrum_dir}/{ds}_{model}_obs{dens}_{name}.npz"
                    np.savez_compressed(
                        fp,
                        eigenvalues=ev.cpu().numpy().astype(np.float64),
                        hvp_asymmetry=asym.cpu().numpy().astype(np.float64),
                        dtype_of_block=str(centre.dtype).replace("torch.", ""),
                        dataset=ds,
                        model=model,
                        term=name,
                        density=dens,
                        centre=centre_kind,
                        block_dim=n * D,
                        n_trajectories=B,
                        direction=0,
                        n_directions_requested=n_dirs,
                        radius=radius,
                        seed=seed,
                    )
                    print(f"    [{name}] spectrum -> {fp}", flush=True)
            den = torch.maximum(hi.abs(), lo.abs()).clamp_min(1e-300)
            R[k] = (lo / den).cpu().numpy()
            UNC[k] = (lo / hi.abs().clamp_min(1e-300)).cpu().numpy()
            HI[k] = hi.cpu().numpy()
            _eps = float(
                np.finfo(np.dtype(str(centre.dtype).replace("torch.", ""))).eps
            )
            tol = (n * D * _eps) * hi.abs().clamp_min(1e-300)
            ASY[k] = asym.cpu().numpy()
            CERT[k] = (
                torch.where(lo < -tol, lo / den, torch.zeros_like(lo)).cpu().numpy()
            )
            if ev is None:
                LOW[k, :, 0] = np.nan
                LOW[k, :, 1] = np.nan
                LOW[k, :, 2] = np.nan
                continue
            LOW[k, :, 0] = (ev < -tol[:, None]).sum(1).cpu().numpy()
            LOW[k, :, 1] = (ev.abs() <= tol[:, None]).sum(1).cpu().numpy()
            pos = torch.where(ev > tol[:, None], ev, torch.full_like(ev, float("inf")))
            LOW[k, :, 2] = (pos.min(dim=1).values / den).cpu().numpy()
        per_traj_min = R.min(axis=0)
        spread_dirs = (
            None
            if nd == 1
            else float(
                (R.max(axis=0) - R.min(axis=0)).max() / max(abs(np.median(R)), 1e-300)
            )
        )
        out["terms"][name] = {
            "per_trajectory_min": [float(x) for x in per_traj_min],
            "mean_of_per_trajectory_min": float(per_traj_min.mean()),
            "std_of_per_trajectory_min": float(per_traj_min.std(ddof=1)),
            "worst_trajectory": int(per_traj_min.argmin()),
            "min_over_all": float(per_traj_min.min()),
            "n_directions_used": nd,
            "constant_hessian": bool(constant),
            "constant_hessian_probe_drift": drift,
            "constant_hessian_probe_drift_is_bound": drift_is_bound,
            "constant_hessian_probe_drift_raw": drift_raw,
            "constant_hessian_far_drift": far_drift,
            "constant_hessian_far_drift_is_bound": far_is_bound,
            "constant_hessian_far_drift_raw": far_raw,
            "constant_hessian_far_radius": (
                FAR_RADIUS * radius if far_drift is not None else None
            ),
            "constant_hessian_far_separation": far_sep,
            "constant_hessian_far_hvp_magnitude": far_hvp,
            "constant_hessian_far_bit_identical": far_bitsame,
            "constant_hessian_probe_dtype": (
                far_dtype or str(probe_dtype).replace("torch.", "")
            ),
            "constant_hessian_probe_resolution_floor": floor,
            "max_relative_spread_across_directions": spread_dirs,
            "raw_no_measured_tolerance": {
                "n_trajectories_negative": int((per_traj_min < 0).sum()),
                "n_trajectories_below_fixed_1e-9": int((per_traj_min < -1e-9).sum()),
            },
            "certified_per_trajectory_min": [float(x) for x in CERT.min(axis=0)],
            "mean_of_certified_per_trajectory_min": float(CERT.min(axis=0).mean()),
            "std_of_certified_per_trajectory_min": float(CERT.min(axis=0).std(ddof=1)),
            "n_trajectories_certified_negative": int((CERT.min(axis=0) < 0).sum()),
            "signed": [float(_squash(u)) for u in UNC.min(axis=0)],
            "signed_mean": float(np.mean([_squash(u) for u in UNC.min(axis=0)])),
            "signed_std": float(np.std([_squash(u) for u in UNC.min(axis=0)], ddof=1)),
            "non_convex": bool((CERT.min(axis=0) < 0).any()),
            "uncensored_per_trajectory_min": [float(x) for x in UNC.min(axis=0)],
            "uncensored_mean": float(UNC.min(axis=0).mean()),
            "lambda_max_per_trajectory_median": [
                float(x) for x in np.median(HI, axis=0)
            ],
            "n_trajectories_pinned_at_minus_one": int(
                sum(1 for x in R.min(axis=0) if abs(x + 1.0) < 1e-12)
            ),
            "smallest_positive_mean": float(LOW[:, :, 2].mean()),
            "smallest_positive_std": float(LOW[:, :, 2].std(ddof=1)),
            "hvp_asymmetry_median": float(np.median(ASY)),
            "low_end": {
                "n_negative_eigs_median": float(np.median(LOW[:, :, 0])),
                "n_zero_band_eigs_median": float(np.median(LOW[:, :, 1])),
                "expected_null_dim_D": int(D),
                "smallest_positive_ratio_mean": float(LOW[:, :, 2].mean()),
                "smallest_positive_ratio_std": float(LOW[:, :, 2].std(ddof=1)),
            },
            "wall_s": round(time.perf_counter() - t0, 1),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lorenz96")
    ap.add_argument("--models", default="nl_ae,wl_ae,kae")
    ap.add_argument("--density", type=int, default=30)
    ap.add_argument("--dirs", type=int, default=64)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--centre",
        default="background",
        choices=("background", "strong"),
    )
    ap.add_argument("--with-background", action="store_true")
    ap.add_argument(
        "--spectrum-dir",
        default=None,
    )
    ap.add_argument(
        "--far-all",
        action="store_true",
    )
    ap.add_argument(
        "--terms",
        default=None,
    )
    ap.add_argument(
        "--block-store",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        dest="block_store",
    )
    ap.add_argument(
        "--obs-iters",
        type=int,
        default=None,
        dest="obs_iters",
    )
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    want = (("b_background",) + TERMS) if a.with_background else TERMS
    if a.terms:
        sel = tuple(x.strip() for x in a.terms.split(","))
        bad = [x for x in sel if x not in want]
        if bad:
            raise SystemExit(str(bad))
        want = sel
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = json.load(open(OUT))["cells"] if os.path.exists(OUT) else {}
    for model in a.models.split(","):
        key = f"{a.dataset}:{model}:obs{a.density}:{a.centre}"
        try:
            r = run(
                a.dataset,
                model,
                a.density,
                dev,
                a.dirs,
                a.radius,
                a.seed,
                want,
                a.centre,
                a.block_store,
                a.obs_iters,
                a.far_all,
                a.spectrum_dir,
            )
        except (Exception, SystemExit) as e:
            print(f"FAIL {key:28s} {type(e).__name__}: {str(e)[:110]}", flush=True)
            continue
        if key in res and "terms" in res[key]:
            kept = {t: v for t, v in res[key]["terms"].items() if t not in r["terms"]}
            if kept:
                print(f"    kept {sorted(kept)}", flush=True)
            r["terms"] = {**kept, **r["terms"]}
        res[key] = r
        print(
            f"\n=== {key}   {r['n_trajectories']} trajectories, "
            f"block {r['block_dim']}, up to {a.dirs} directions",
            flush=True,
        )
        print(
            f"    {'term':16s}{'SIGNED (mean)':>16s}{'std':>11s}"
            f"{'raw lam_min':>14s}{'certified<0':>13s}{'dirs':>6s}{'drift':>10s}"
            f"{'far drift':>11s}{'far sep':>10s}{'|Hv|':>10s}{'dtype':>9s}"
        )
        for t, s in r["terms"].items():
            fd = s.get("constant_hessian_far_drift")
            print(
                f"    {t:16s}{s['signed_mean']:+16.4e}{s['signed_std']:11.3e}"
                f"{s['min_over_all']:+14.4e}"
                f"{s['n_trajectories_certified_negative']:8d}/{r['n_trajectories']}"
                f"{s['n_directions_used']:6d}"
                f"{s['constant_hessian_probe_drift']:10.1e}"
                f"{'-' if fd is None else format(fd, '.1e'):>11s}"
                f"{'-' if s.get('constant_hessian_far_separation') is None else format(s['constant_hessian_far_separation'], '.1e'):>10s}"
                f"{'-' if s.get('constant_hessian_far_hvp_magnitude') is None else format(s['constant_hessian_far_hvp_magnitude'], '.1e'):>10s}"
                f"{'-' if s.get('constant_hessian_probe_dtype') is None else s['constant_hessian_probe_dtype']:>9s}"
                f"{'' if s.get('constant_hessian_far_bit_identical') is None else ('  bitsame' if s['constant_hessian_far_bit_identical'] else '  differs')}",
                flush=True,
            )
        if a.write:
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            json.dump(
                {"cells": res},
                open(OUT, "w"),
                indent=1,
            )
    if a.write:
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
