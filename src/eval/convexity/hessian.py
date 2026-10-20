import argparse
import json
import os

import numpy as np
import torch
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from torch.nn.attention import sdpa_kernel, SDPBackend

from src.assimilation.variational import _b_quadratic, _step_seq, solve_window
from src.eval.benchmark import setup, sigmas

OUT = "results/convexity/hessian"

_MATH = lambda: sdpa_kernel(SDPBackend.MATH)


def background_trajectory(m, z_b, steps):
    with torch.no_grad():
        states = [z_b]
        for _ in range(steps):
            states.append(m.step(states[-1]).float())
    return torch.stack(states, dim=1).detach()


def term_costs(m, z_b, Bi, sq, sr, obs_idx, masks, obs):
    def J_b(Z):
        return _b_quadratic(Z[:, 0] - z_b, Bi)

    def J_q(Z):
        return ((Z[:, 1:] - _step_seq(m, Z[:, :-1])) ** 2).sum() / sq**2

    def J_r(Z):
        dec = m.decode(Z[:, obs_idx].reshape(Z.shape[0] * len(obs_idx), -1)).reshape(
            Z.shape[0], len(obs_idx), *obs.shape[2:]
        )
        return ((masks * dec - obs) ** 2).sum() / sr**2

    return {
        "b_background": J_b,
        "q_model_error": J_q,
        "r_observation": J_r,
        "TOTAL": lambda Z: J_b(Z) + J_q(Z) + J_r(Z),
    }


def hvp_fn(J, Z0):
    Z = Z0.clone().detach().requires_grad_(True)
    with _MATH():
        (g,) = torch.autograd.grad(J(Z), Z, create_graph=True)

    def hvp(v):
        with _MATH():
            (out,) = torch.autograd.grad(g, Z, grad_outputs=v, retain_graph=True)
        return out.detach()

    return hvp


def _as_operator(hvp, shape, dev, counter=None):
    n = int(np.prod(shape))

    def mv(x):
        if counter is not None:
            counter[0] += 1
        t = (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            .to(dev)
            .reshape(shape)
        )
        return hvp(t).reshape(-1).detach().cpu().numpy().astype(np.float64)

    return LinearOperator((n, n), matvec=mv, dtype=np.float64)


def _lanczos_end(op, which, v0, tol, maxiter, label):
    try:
        val, vec = eigsh(op, k=1, which=which, v0=v0, tol=tol, maxiter=maxiter)
    except ArpackNoConvergence:
        try:
            val, vec = eigsh(
                op,
                k=1,
                which=which,
                v0=v0,
                tol=tol * 100,
                maxiter=(maxiter or 0) * 5 or None,
            )
        except ArpackNoConvergence as e:
            raise RuntimeError(label, which, tol * 100) from e
    return float(val[0]), vec[:, 0]


def spectrum_ends(hvp, shape, dev, gen, iters):
    return spectrum_ends_witnessed(hvp, shape, dev, gen, iters)["ends"]


def spectrum_ends_witnessed(hvp, shape, dev, gen, iters, tol=1e-2, rtol=0.05):
    n = int(np.prod(shape))
    cnt = [0]
    op = _as_operator(hvp, shape, dev, cnt)
    v0 = torch.randn(shape, generator=gen, device=dev).reshape(-1)
    v0 = (v0 / v0.norm()).detach().cpu().numpy().astype(np.float64)
    budget = max(int(iters) * 20, 200)

    lo, w_lo = _lanczos_end(op, "SA", v0, tol, budget, "lambda_min")
    hi, w_hi = _lanczos_end(op, "LA", v0, tol, budget, "lambda_max")

    def rayleigh(vec):
        t = (
            torch.from_numpy(np.ascontiguousarray(vec, dtype=np.float32))
            .to(dev)
            .reshape(shape)
        )
        cnt[0] += 1
        return float((t * hvp(t)).sum() / (t * t).sum())

    r_lo, r_hi = rayleigh(w_lo), rayleigh(w_hi)

    scale = max(abs(lo), abs(hi), 1e-300)
    ok = abs(r_lo - lo) <= rtol * scale and abs(r_hi - hi) <= rtol * scale
    if lo >= 0 > r_lo:
        raise RuntimeError(lo, r_lo)
    if not ok:
        raise RuntimeError(lo, r_lo, hi, r_hi)

    return {
        "ends": (lo, hi),
        "rayleigh_min": r_lo,
        "rayleigh_max": r_hi,
        "n_hvp": cnt[0],
        "witness_ok": ok,
        "dim": n,
    }


def window_inputs(m, runs, sampler, cfg, rl, dev):
    from src.assimilation.variational import _AMP

    B = len(runs)
    gt = [torch.stack([runs[b][t] for b in range(B)]).to(dev) for t in range(rl + 1)]
    with torch.autocast(**_AMP):
        z = m.encode(gt[0]).float()
        for _ in range(int(cfg["spinup"])):
            z = m.step(z).float()
    z_b = z.detach()
    cur, W, oev = int(cfg["spinup"]), int(cfg["window_size"]), int(cfg["obs_every"])
    obs_idx = [k for k in range(W + 1) if (cur + k) % oev == 0]
    masks = torch.stack(
        [torch.stack([sampler.mask(b, cur + k) for k in obs_idx]) for b in range(B)]
    ).to(dev)
    obs = torch.stack([gt[cur + k] for k in obs_idx], dim=1) * masks
    return z_b, obs_idx, masks, obs, W


def main(ds, models, dens, radii, iters, seed, out, dirs=1):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {
        "dataset": ds,
        "density": dens,
        "radii": radii,
        "power_iters": iters,
        "seed": seed,
        "cells": {},
    }
    if os.path.exists(out):
        try:
            old = json.load(open(out))
            if isinstance(old.get("cells"), dict):
                res["cells"] = dict(old["cells"])
        except Exception as e:
            print(f"  [warn] {out} {type(e).__name__}", flush=True)
    for model in models:
        sr, sq = sigmas(ds, model, dens)[:2]
        cfg, m, runs, sampler, Bi, rl = setup(ds, model, dens, dev)
        z_b, obs_idx, masks, obs, W = window_inputs(m, runs, sampler, cfg, rl, dev)
        terms = term_costs(m, z_b, Bi, sq, sr, obs_idx, masks, obs)
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
        background = background_trajectory(m, z_b, W)
        scale = background.std(dim=1, keepdim=True, correction=0)
        gen = torch.Generator(device=dev).manual_seed(seed)
        key = f"{ds}:{model}:obs{dens}"
        cell = res["cells"].get(key) or {}
        if cell.get("perturbation_scaling") != "background_component_std_over_time":
            cell = {}
        cell.update({
            "sigma_r": sr,
            "sigma_q": sq,
            "zb_std": scale.squeeze(1).cpu().tolist(),
            "perturbation_scaling": "background_component_std_over_time",
            "window": W,
        })
        cell.setdefault("points", {})
        print(f"\n  ===== {key} =====", flush=True)
        for R in radii:
            if dirs > 1 and R != 0:
                vals = []
                for _ in range(dirs):
                    Zd = (
                        centre
                        + torch.randn(centre.shape, generator=gen, device=dev)
                        * R
                        * scale
                    )
                    lo, hi = spectrum_ends(
                        hvp_fn(terms["TOTAL"], Zd), centre.shape, dev, gen, iters
                    )
                    if lo <= hi:
                        vals.append(lo / max(abs(hi), abs(lo), 1e-300))
                cell["points"].setdefault(str(R), {})["TOTAL_dirs"] = vals
                import statistics as _st

                print(
                    f"   R={R:<5g} TOTAL over {len(vals)} directions: "
                    f"median={_st.median(vals):+.3f}  "
                    f"min={min(vals):+.3f} max={max(vals):+.3f}",
                    flush=True,
                )
                res["cells"][key] = cell
                json.dump(res, open(out, "w"), indent=1)
                continue
            Zp = (
                centre
                if R == 0
                else centre
                + torch.randn(centre.shape, generator=gen, device=dev) * R * scale
            )
            rec = {}
            for tname, Jt in terms.items():
                lmin, lmax = spectrum_ends(
                    hvp_fn(Jt, Zp), centre.shape, dev, gen, iters
                )
                if lmin > lmax:
                    print(f"   R={R:<5g} {tname:14s} skipped", flush=True)
                    continue
                ref = max(abs(lmax), abs(lmin))
                ratio = lmin / ref if ref else 0.0
                rec[tname] = {"lambda_min": lmin, "lambda_max": lmax, "ratio": ratio}
                print(
                    f"   R={R:<5g} {tname:14s} lmin={lmin:+.4e} lmax={lmax:+.4e}"
                    f"  ratio={ratio:+.3e}  [{'PSD' if ratio > -1e-6 else 'NEG'}]",
                    flush=True,
                )
            cell["points"].setdefault(str(R), {}).update(rec)
            res["cells"][key] = cell
            res["radii"] = sorted({float(x) for x in cell["points"]})
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            json.dump(res, open(out, "w"), indent=1)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\n  wrote {out}  ({len(res['cells'])} cells)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="nl_ae,wl_ae,kae")
    ap.add_argument("--density", type=int, default=30)
    ap.add_argument(
        "--radii",
        default="0,1.0",
    )
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--dirs",
        type=int,
        default=1,
    )
    a = ap.parse_args()
    main(
        a.dataset,
        a.models.split(","),
        a.density,
        [float(x) for x in a.radii.split(",")],
        a.iters,
        a.seed,
        a.out or f"{OUT}/{a.dataset}.json",
        a.dirs,
    )
