import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from src.config import load_config
from src.eval.benchmark import COV_SCHEMES, TV_DRIVER, tv_setup, tv_solve

OUT = "results/tensorvar_covariance.json"
DATASETS = ["cylinder", "shallow_water", "era5", "era5_high"]
DENS = [5, 15, 30, 50]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in DATASETS:
        val = load_config("config/da.yaml", f"{ds}_obs30")["val_path"]
        rec = out.setdefault(ds, {})
        for scheme, params in COV_SCHEMES.items():
            cell = rec.setdefault(scheme, {})
            for d in DENS:
                if str(d) in cell:
                    continue
                D = {**TV_DRIVER[ds], **params, "cov": scheme}
                t = tv_setup(ds, d, dev, driver=D, data_path=val)
                arr, _ = tv_solve(
                    t["solve"],
                    t["m"],
                    t["runs"],
                    t["sampler"],
                    t["Bi"],
                    t["sq"],
                    t["sr"],
                    spinup=t["spin"],
                    **t["kw"],
                )
                cell[str(d)] = float(np.asarray(arr)[:, t["spin"] :].mean())
                print(f"  {ds:<14}{scheme:<16}obs{d:<3}{cell[str(d)]:.6f}", flush=True)
                os.makedirs(os.path.dirname(OUT), exist_ok=True)
                json.dump(out, open(OUT, "w"), indent=1)
                del t
                torch.cuda.empty_cache()
        mean = {s: float(np.mean([rec[s][str(d)] for d in DENS])) for s in COV_SCHEMES}
        rec["selected"] = min(mean, key=mean.get)
        rec["mean"] = mean
        print(f"{ds}: selected {rec['selected']}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
