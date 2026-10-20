from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from src.eval.convexity.hessian_per_traj import run, CLOSED

OUT_DIR = "results/convexity/eigenvalue_spectra"
MANIFEST = "_manifest.json"


def stem(ds, model, dens, term, radius, seed, centre_kind):
    r = ("%g" % radius).replace(".", "p").replace("-", "m")
    tag = "" if centre_kind == "background" else f"_{centre_kind}"
    return f"{ds}_{model}_obs{dens}_{term}{tag}_R{r}_seed{seed}"


def one(
    ds,
    model,
    dens,
    term,
    radius,
    seed,
    dev,
    out_dir,
    block_store,
    force,
    centre_kind="background",
):
    fp = os.path.join(
        out_dir, stem(ds, model, dens, term, radius, seed, centre_kind) + ".npz"
    )
    if os.path.exists(fp) and not force:
        print(f"    exists, skipping: {fp}", flush=True)
        return None
    tmp = tempfile.mkdtemp(prefix="spt_")
    t0 = time.perf_counter()
    try:
        rec = run(
            ds,
            model,
            dens,
            dev,
            1,
            radius,
            seed,
            (term,),
            centre_kind,
            block_store,
            None,
            False,
            tmp,
        )
        src = os.path.join(tmp, f"{ds}_{model}_obs{dens}_{term}.npz")
        if not os.path.exists(src):
            print(f"    skipped {ds}:{model}:{term}", flush=True)
            return None
        d = dict(np.load(src))
        d["centre_seed"] = seed
        d["centre_index"] = 0
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(fp, **d)
        wall = time.perf_counter() - t0
        s = rec["terms"][term]
        rmin = s["min_over_all"]
        print(f"    wrote {fp}  ({wall:.1f}s, ratio_min {rmin:+.4e})", flush=True)
        return {
            "file": os.path.basename(fp),
            "dataset": ds,
            "model": model,
            "density": dens,
            "term": term,
            "radius": radius,
            "seed": seed,
            "centre_kind": centre_kind,
            "block_dim": rec["block_dim"],
            "n_trajectories": rec["n_trajectories"],
            "wall_s": round(wall, 1),
            "block_diagonal_leakage": rec["block_diagonal_leakage"],
            "ratio_min_over_all_this_centre": rmin,
            "per_trajectory_ratio_min_this_centre": s["per_trajectory_min"],
            "hvp_asymmetry_median": s.get("hvp_asymmetry_median"),
            "signed_mean_this_centre": s.get("signed_mean"),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lorenz96")
    ap.add_argument("--models", default="tensorvar,dmd,kae,nl_ae,wl_ae")
    ap.add_argument("--density", type=int, default=30)
    ap.add_argument(
        "--terms",
        default="TOTAL",
    )
    ap.add_argument(
        "--radii",
        default="0,1.0",
    )
    ap.add_argument(
        "--seeds",
        default="1234",
    )
    ap.add_argument(
        "--centre",
        default="background",
        choices=("background", "strong"),
    )
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--block-store", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument(
        "--force",
        action="store_true",
    )
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    radii = [float(x) for x in a.radii.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []
    bad = (
        [m for m in a.models.split(",") if m in CLOSED] if a.centre == "strong" else []
    )
    if bad:
        raise SystemExit(str(bad))
    for model in a.models.split(","):
        for term in a.terms.split(","):
            for radius in radii:
                use = seeds[:1] if radius == 0.0 else seeds
                for seed in use:
                    print(
                        f"[{a.dataset}:{model}:{term}] centre={a.centre} "
                        f"R={radius} seed={seed}",
                        flush=True,
                    )
                    try:
                        r = one(
                            a.dataset,
                            model,
                            a.density,
                            term,
                            radius,
                            seed,
                            dev,
                            a.out_dir,
                            a.block_store,
                            a.force,
                            a.centre,
                        )
                    except (Exception, SystemExit) as e:
                        print(
                            f"    FAIL {type(e).__name__}: {str(e)[:160]}", flush=True
                        )
                        continue
                    if r:
                        rows.append(r)
    if not rows:
        print("nothing new written")
        return
    mp = os.path.join(a.out_dir, MANIFEST)
    old = json.load(open(mp))["entries"] if os.path.exists(mp) else []
    keep = {e["file"]: e for e in old}
    for r in rows:
        keep[r["file"]] = r
    json.dump(
        {
            "entries": [keep[k] for k in sorted(keep)],
        },
        open(mp, "w"),
        indent=1,
    )
    print(f"\nwrote {mp}  ({len(keep)} entries)")


if __name__ == "__main__":
    main()
