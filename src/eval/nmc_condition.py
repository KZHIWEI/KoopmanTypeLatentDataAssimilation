import json
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from src.eval import benchmark

OUT = "results/nmc_condition.json"
DATASETS = ["lorenz96", "cylinder", "shallow_water", "era5", "era5_high"]
MODELS = ["nl_ae", "wl_ae", "kae", "tensorvar", "dmd"]


def background_precision(ds, model, dev):
    if model == "tensorvar":
        return benchmark.tv_setup(ds, 30, dev)["Bi"]
    if model == "dmd":
        return benchmark.dmd_setup(ds, 30, dev)["Bi"]
    return benchmark.setup(ds, model, 30, dev)[4]


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for m in MODELS:
            Bi = background_precision(ds, m, dev).double()
            ev = Bi if Bi.dim() == 1 else torch.linalg.eigvalsh(Bi)
            out[ds][m] = float(ev.max() / ev.min())
            print(f"  {ds:<16}{m:<10}{out[ds][m]:.3e}", flush=True)
            torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
