from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.eval.convexity.hessian import spectrum_ends, hvp_fn
from src.eval import benchmark

OUT = "results/convexity/closed_form_curvature.json"
DATASETS = ["lorenz96", "cylinder", "shallow_water", "era5", "era5_high"]


_DT = torch.float32 if os.environ.get("HESSIAN_DTYPE") == "float32" else torch.float64


def build(ds, model, dens, dev):
    from src.assimilation import tensorvar as RT
    from src.assimilation import tensorvar_feature as TVF
    from src.assimilation import dmd as RD

    rec = {}

    def snap(
        C,
        z_b,
        y_or_yhat,
        obs_idx,
        Bi,
        sq,
        sr,
        horizon,
        constraint,
        Qi,
        Ri_or_Sinv,
        H=None,
    ):
        if not rec:
            rec.update(
                C=C,
                z_b=z_b,
                Y=y_or_yhat,
                obs=set(int(k) for k in obs_idx),
                Bi=Bi,
                sq=float(sq),
                sr=float(sr),
                n=int(horizon),
                Qi=Qi,
                RS=Ri_or_Sinv,
                H=H,
                dev=z_b.device,
            )
        raise _Got()

    class _Got(Exception):
        pass

    _dense, _closed = TVF.solve_window_dense, RT.solve_window_closed
    holders = [RT, RD, TVF]

    def w_dense(
        C, z_b, yhat, obs_idx, Bi, sq, sr, horizon, constraint, Qi, Sinv_full, H
    ):
        snap(
            C,
            z_b,
            yhat,
            obs_idx,
            Bi,
            sq,
            sr,
            horizon,
            constraint,
            Qi,
            Sinv_full[int(horizon)],
            H,
        )

    def w_closed(
        C, z_b, y, obs_idx, Bi, sq, sr, horizon, constraint="weak", Qi=None, Ri=None
    ):
        snap(C, z_b, y, obs_idx, Bi, sq, sr, horizon, constraint, Qi, Ri, None)

    TVF.solve_window_dense = w_dense
    for h in holders:
        if hasattr(h, "solve_window_closed"):
            h.solve_window_closed = w_closed
    try:
        (benchmark.run_dmd_cell if model == "dmd" else benchmark.run_tv_cell)(
            ds, dens, dev
        )
    except _Got:
        pass
    finally:
        TVF.solve_window_dense = _dense
        for h in holders:
            if hasattr(h, "solve_window_closed"):
                h.solve_window_closed = _closed
    if not rec:
        raise SystemExit(f"{ds}:{model}")

    D, n, dv = rec["z_b"].shape[1], rec["n"], rec["dev"]
    q, r = 1.0 / rec["sq"] ** 2, 1.0 / rec["sr"] ** 2
    I = torch.eye(D, device=dv, dtype=_DT)
    C64 = rec["C"].to(_DT)
    QiM = q * (rec["Qi"].to(_DT) if rec["Qi"] is not None else I)
    from src.assimilation.tensorvar import _bi_matrix

    BiM = _bi_matrix(rec["Bi"], D).to(_DT)
    zb = rec["z_b"].to(_DT)
    Y = rec["Y"].to(_DT) if rec["Y"] is not None else None
    obs = rec["obs"]

    if rec["H"] is not None:
        H64 = rec["H"].to(_DT)
        Sinv = r * rec["RS"].to(_DT)
        Hb = torch.block_diag(
            *[H64 if t in obs else torch.zeros_like(H64) for t in range(n)]
        )

        def J_r(Z):
            d = (
                Y[:, :n].reshape(Z.shape[0], n * D)
                - Z.reshape(Z.shape[0], n * D) @ Hb.t()
            )
            return ((d @ Sinv) * d).sum()

    else:
        RiM = r * (rec["RS"].to(_DT) if rec["RS"] is not None else I)

        def J_r(Z):
            s = Z.new_zeros(())
            for t in sorted(obs):
                d = Y[:, t] - Z[:, t]
                s = s + ((d @ RiM) * d).sum()
            return s

    def J_b(Z):
        d = Z[:, 0] - zb
        return ((d @ BiM) * d).sum()

    def J_q(Z):
        d = Z[:, 1:] - Z[:, :-1] @ C64.t()
        return ((d.reshape(-1, D) @ QiM) * d.reshape(-1, D)).sum()

    terms = {
        "b_background": J_b,
        "q_model_error": J_q,
        "r_observation": J_r,
        "TOTAL": lambda Z: J_b(Z) + J_q(Z) + J_r(Z),
    }
    Z = [zb]
    for _ in range(n - 1):
        Z.append(Z[-1] @ C64.t())
    centre = torch.stack(Z, 1)
    scale = centre.std(dim=1, keepdim=True, correction=0)
    return terms, centre, scale, rec


def run_cell(ds, model, dens, dev, n_dirs, radius, seed, iters):
    terms, centre, scale, rec = build(ds, model, dens, dev)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {
        "dataset": ds,
        "model": model,
        "density": dens,
        "radius": radius,
        "n_directions": n_dirs,
        "seed": seed,
        "rng_device": "cpu",
        "perturbation_scaling": "background_component_std_over_time",
        "n_steps": rec["n"],
        "block_dim": rec["z_b"].shape[1],
        "n_unobserved": rec["n"] - len(rec["obs"]),
        "terms": {},
    }
    for name, J in terms.items():
        vals, t0 = [], time.perf_counter()
        for k in range(n_dirs):
            d = torch.randn(centre.shape, generator=gen).to(dev).to(_DT)
            Z0 = centre + radius * scale * d
            _h = hvp_fn(J, Z0)
            hvp = lambda v: _h(v.view(Z0.shape)).reshape(-1)
            lo, hi = spectrum_ends(
                hvp,
                Z0.numel(),
                Z0.device,
                torch.Generator(device=Z0.device).manual_seed(seed),
                iters,
            )
            vals.append(lo / max(abs(hi), abs(lo), 1e-300))
        v = np.array(vals, dtype=np.float64)
        out["terms"][name] = {
            "ratio": [float(x) for x in v],
            "median": float(np.median(v)),
            "min": float(v.min()),
            "max": float(v.max()),
            "relative_spread": float(
                (v.max() - v.min()) / max(abs(np.median(v)), 1e-300)
            ),
            "n_negative": int((v < 0).sum()),
            "n_negative_beyond_tol": int((v < -1e-9).sum()),
            "wall_s": round(time.perf_counter() - t0, 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lorenz96")
    ap.add_argument("--models", default="tensorvar,dmd")
    ap.add_argument("--density", type=int, default=30)
    ap.add_argument("--dirs", type=int, default=64)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dss = DATASETS if a.dataset == "all" else [a.dataset]
    res = json.load(open(OUT))["cells"] if os.path.exists(OUT) else {}
    for ds in dss:
        for model in a.models.split(","):
            key = f"{ds}:{model}:obs{a.density}"
            try:
                r = run_cell(
                    ds, model, a.density, dev, a.dirs, a.radius, a.seed, a.iters
                )
            except Exception as e:
                print(f"FAIL {key:30s} {type(e).__name__}: {str(e)[:120]}", flush=True)
                continue
            res[key] = r
            print(
                f"\n=== {key}   {r['n_steps']} steps, {r['n_unobserved']} unobserved, "
                f"D={r['block_dim']}",
                flush=True,
            )
            for t, s in r["terms"].items():
                print(
                    f"    {t:16s} median {s['median']:+.4e}  "
                    f"spread {s['relative_spread']:.2e}  "
                    f"n_neg {s['n_negative']}",
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
        print(f"\nwrote {OUT}  ({len(res)} cells)")


if __name__ == "__main__":
    main()
