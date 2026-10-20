import hashlib
import json
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from src.eval.benchmark import REG, tv_checkpoint_paths

OUT = "results/parameter_count.json"
DATASETS = ["lorenz96", "cylinder", "shallow_water", "era5", "era5_high"]
MODELS = ["nl_ae", "wl_ae", "kae", "tensorvar", "dbf", "cgkn"]


def distinct(o, seen, pre=""):
    if torch.is_tensor(o):
        h = hashlib.sha1(o.detach().cpu().numpy().tobytes()).hexdigest()
        seen.setdefault((pre, tuple(o.shape), h), o.numel())
    elif isinstance(o, dict):
        for k, v in o.items():
            distinct(v, seen, f"{pre}{k}.")


def files(ds, m):
    if m == "tensorvar":
        return list(tv_checkpoint_paths(ds, 30))
    return [REG["datasets"][ds][m]["ckpt"].format(d=30)]


def main():
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for m in MODELS:
            seen = {}
            for p in files(ds, m):
                o = torch.load(p, map_location="cpu", weights_only=False)
                distinct(o.get("state_dict", o) if isinstance(o, dict) else o, seen)
            out[ds][m] = sum(seen.values())
            print(f"  {ds:<14}{m:<10}{out[ds][m]:>12,}", flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
