from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np

from src.eval.gaussianity import analysis
from src.eval.gaussianity.ensembles import common as C, get as get_runner

OUT_ROOT = "results/gaussianity"
CACHE = ".gauss_cache"


def cell_path(ds: str, model: str) -> str:
    return os.path.join(OUT_ROOT, ds, f"{model}.json")


def cache_path(ds: str, model: str) -> str:
    return os.path.join(CACHE, f"{ds}_{model}.npz")


_EPS32 = 2.0**-24


def _resolution_guard(rec, meta, per) -> dict:
    import numpy as _np

    D = per[0]["D"]
    comp_std = (rec["median"]["trace_analysis"] / D) ** 0.5
    ref = meta.get("reference_rms")
    if ref is None:
        return {"checked": False}
    quant = _EPS32 * float(ref)
    weak = (
        float(_np.median([min(s["analysis"]["proj_B"]["coord_std"]) for s in per]))
        if "coord_std" in per[0]["analysis"]["proj_B"]
        else comp_std
    )
    ratio = weak / max(quant, 1e-300)
    return {
        "checked": True,
        "reference_rms": float(ref),
        "float32_quantisation_std": float(quant),
        "analysis_component_std": float(comp_std),
        "weakest_projB_coord_std": weak,
        "spread_over_quantisation": float(ratio),
        "degenerate": bool(ratio < 100.0),
    }


def run_cell(ds, model, *, members, seed, density, chunk, k):
    t0 = time.time()
    starts, B, meta = get_runner(model)(
        ds, members=members, seed=seed, density=density, chunk=chunk
    )
    wall = time.time() - t0
    per = [analysis.per_start(eb, ea, B, k=k) for eb, ea in starts]
    rec = analysis.summarise(per)
    rec.update(
        {
            "dataset": ds,
            "model": model,
            "density": density,
            "members": members,
            "seed": seed,
            "k_projection": per[0]["k"],
            "wall_seconds": round(wall, 1),
            "provenance": meta,
            "numerical_resolution": _resolution_guard(rec, meta, per),
            "per_start": per,
        }
    )
    os.makedirs(os.path.dirname(cell_path(ds, model)), exist_ok=True)
    with open(cell_path(ds, model), "w") as f:
        json.dump(rec, f, indent=1)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(
        cache_path(ds, model),
        B=B,
        E_b=np.stack([s[0] for s in starts]),
        E_a=np.stack([s[1] for s in starts]),
    )
    a = rec["analysis"]
    print(
        f"  [{model}] p_HZ(projB)={a['hz_projB']['p']:.3g} "
        f"p_Lilliefors(projB)={a['lilliefors_projB']['p']:.3g} "
        f"tr(A)/tr(B)={rec['median']['trace_ratio']:.4g} ({wall:.0f}s)",
        flush=True,
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lorenz96", choices=list(C.DATASETS))
    ap.add_argument("--models", nargs="*", default=list(C.MODELS))
    ap.add_argument("--members", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", type=int, default=30, choices=[5, 15, 30, 50])
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--k", type=int, default=25)
    a = ap.parse_args()
    print(f"=== gaussianity {a.dataset} ({a.members} members, obs{a.density}) ===")
    fail = 0
    for m in a.models:
        try:
            run_cell(
                a.dataset,
                m,
                members=a.members,
                seed=a.seed,
                density=a.density,
                chunk=a.chunk,
                k=a.k,
            )
        except Exception:
            fail += 1
            print(f"  [{m}] FAILED\n{traceback.format_exc()}", flush=True)
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
