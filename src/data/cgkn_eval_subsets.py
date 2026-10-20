from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

from src.assimilation.data import tile_starts
from src.config import load_config

LAYOUT = {
    "cylinder": "nchw",
    "shallow_water": "nchw",
    "lorenz96": "nchw",
    "era5": "nhwc",
    "era5_high": "nhwc",
}


def build(ds: str, root: str = "."):
    cfg = load_config("config/da.yaml", ds)
    path = os.path.join(root, cfg["da_data_path"])
    ev = cfg["eval"]
    skip, L = int(cfg["skip"]), int(ev["length"])
    n_frames = L + 1

    if path.endswith(".h5"):
        import h5py

        with h5py.File(path, "r") as f:
            d = f["data"]
            T = d.shape[0]
            starts = tile_starts(ev, skip, 1, T)
            runs = []
            for traj, t0 in starts:
                end = t0 + skip * n_frames
                if end > T:
                    raise SystemExit(f"{ds} {t0} {end} {T}")
                runs.append(np.asarray(d[t0:end:skip]))
    else:
        a = np.load(path, mmap_mode="r")
        T = a.shape[1]
        starts = tile_starts(ev, skip, a.shape[0], T)
        runs = []
        for traj, t0 in starts:
            end = t0 + skip * n_frames
            if end > T:
                raise SystemExit(f"{ds} {traj} {t0} {end} {T}")
            runs.append(np.asarray(a[traj, t0:end:skip]))

    out = np.stack(runs).astype(np.float32)
    if out.shape[1] != n_frames:
        raise SystemExit(f"{ds} {out.shape[1]} {n_frames}")
    return out, {
        "source": cfg["da_data_path"],
        "n_runs": len(starts),
        "skip": skip,
        "run_length": L,
        "frames": n_frames,
        "shape": list(out.shape),
        "layout": LAYOUT.get(ds, "nchw"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=_REPO)
    ap.add_argument(
        "--force-release",
        action="store_true",
    )
    a = ap.parse_args()
    if "ckpt_release" in a.out and not a.force_release:
        ap.error(a.out)

    arr, meta = build(a.dataset, a.root)
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"eval_runs_{a.dataset}.npy")
    np.save(p, arr)
    print(f"  wrote {p}")
    for k, v in meta.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
