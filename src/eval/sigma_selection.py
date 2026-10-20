import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from src.config import load_config
from src.eval.benchmark import TV_DRIVER, dmd_setup, tv_setup
from src.eval.covariance_selection import selection_path

DATASETS = ["lorenz96", "cylinder", "shallow_water", "era5", "era5_high"]
MODELS = ["dmd", "tensorvar"]
DENS = [5, 15, 30, 50]
GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
DMD_GRID = sorted(GRID + [0.05])


def _score(t, sq, sr, constraint):
    kw = dict(t["kw"], constraint=constraint)
    out = t["solve"](
        t["m"], t["runs"], t["sampler"], t["Bi"], sq, sr, spinup=t["spin"], **kw
    )
    arr = out[0] if isinstance(out, tuple) else out
    return float(np.asarray(arr)[:, t["spin"] :].mean())


def _select(t, grid=GRID):
    strong = [(sr, _score(t, 1.0, sr, "strong")) for sr in grid]
    finite = [(value, score) for value, score in strong if np.isfinite(score)]
    if not finite:
        raise ValueError("No finite strong-constraint validation score")
    sr = min(finite, key=lambda r: r[1])[0]
    weak = [(sq, _score(t, sq, sr, "weak")) for sq in grid]
    finite = [(value, score) for value, score in weak if np.isfinite(score)]
    if not finite:
        raise ValueError("No finite weak-constraint validation score")
    sq, best = min(finite, key=lambda r: r[1])
    strong = [(value, score if np.isfinite(score) else None) for value, score in strong]
    weak = [(value, score if np.isfinite(score) else None) for value, score in weak]
    return dict(sigma_r=sr, sigma_q=sq, weak_best=best, strong=strong, weak=weak)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want = sys.argv[1].split(",") if len(sys.argv) > 1 else DATASETS
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else MODELS
    densities = [int(d) for d in sys.argv[3].split(",")] if len(sys.argv) > 3 else DENS
    path = selection_path()
    out = {}
    if path.exists():
        with path.open() as stream:
            out = json.load(stream)
    for ds in want:
        val = load_config("config/da.yaml", f"{ds}_obs30")["val_path"]
        for model in models:
            grid = DMD_GRID if model == "dmd" else GRID
            for d in densities:
                key = f"{ds}:{model}:obs{d}"
                if key in out:
                    cached = out[key]
                    if (
                        cached.get("split") == "validation"
                        and cached.get("validation_path") == val
                        and all(
                            [row[0] for row in cached.get(stage, [])] == grid
                            for stage in ("strong", "weak")
                        )
                    ):
                        continue
                if model == "dmd":
                    t = dmd_setup(ds, d, dev, data_path=val, use_selected_sigmas=False)
                else:
                    t = tv_setup(ds, d, dev, driver=TV_DRIVER[ds], data_path=val)
                out[key] = _select(t, grid)
                out[key].update(split="validation", validation_path=val)
                c = out[key]
                print(
                    f"  {key:<30}sr={c['sigma_r']:<7g}sq={c['sigma_q']:<7g}{c['weak_best']:.6f}",
                    flush=True,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as stream:
                    json.dump(out, stream, indent=1)
                del t
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
