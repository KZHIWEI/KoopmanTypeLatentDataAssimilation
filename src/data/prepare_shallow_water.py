import argparse, glob, os
import numpy as np
import xarray as xr

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.environ.get("SOURCE_DIR", "/mnt/datasets/ShallowWater-2D")
OUT = os.path.join(_REPO, "data", "shallow_water")


def _read_traj(nc_path):
    ds = xr.open_dataset(nc_path)
    u = ds["u"].values[:, 0]
    v = ds["v"].values[:, 0]
    p = ds["pres"].values
    ds.close()
    return np.stack([u, v, p], axis=1).astype(np.float32)


def build_split(src_split, runs_per_seed, limit_seeds=None):
    seeds = sorted(glob.glob(os.path.join(SRC, src_split, "seed=*")))
    if limit_seeds:
        seeds = seeds[:limit_seeds]
    trajs = []
    for si, sd in enumerate(seeds):
        runs = sorted(glob.glob(os.path.join(sd, "run*", "output.nc")))[:runs_per_seed]
        for r in runs:
            trajs.append(_read_traj(r))
        if (si + 1) % 10 == 0:
            print(
                f"  [{src_split}] {si + 1}/{len(seeds)} seeds, {len(trajs)} trajs",
                flush=True,
            )
    arr = np.stack(trajs)
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-runs", type=int, default=10)
    ap.add_argument("--val-runs", type=int, default=3)
    ap.add_argument("--test-runs", type=int, default=3)
    ap.add_argument("--limit-seeds", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ls = args.limit_seeds or None
    print("[train] building...", flush=True)
    tr = build_split("train", args.train_runs, ls)
    np.save(os.path.join(OUT, "train.npy"), tr)
    print(
        f"[train] {tr.shape} -> {OUT}/train.npy ({tr.nbytes / 1000000000.0:.1f} GB)",
        flush=True,
    )
    print("[val] building...", flush=True)
    va = build_split("valid", args.val_runs, ls)
    np.save(os.path.join(OUT, "val.npy"), va)
    print(
        f"[val] {va.shape} -> {OUT}/val.npy ({va.nbytes / 1000000000.0:.1f} GB)",
        flush=True,
    )
    print("[test] building...", flush=True)
    te = build_split("test", args.test_runs, ls)
    np.save(os.path.join(OUT, "test.npy"), te)
    print(
        f"[test] {te.shape} -> {OUT}/test.npy ({te.nbytes / 1000000000.0:.1f} GB)",
        flush=True,
    )
    flat = tr.reshape(-1, tr.shape[2], tr.shape[3], tr.shape[4])
    mean = flat.mean(axis=(0, 2, 3))
    std = flat.std(axis=(0, 2, 3))
    print(f"[stats] per-channel mean(u,v,pres)={mean}  std={std}", flush=True)


if __name__ == "__main__":
    main()
