import json
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPECTRA = os.path.join(_REPO, "results", "spectra")
OUT = os.path.join(_REPO, "results", "bias_variance.json")
DATASETS = ["lorenz96", "cylinder", "shallow_water", "era5", "era5_high"]
MODELS = ["NL-AE", "WL-AE", "KAE", "TensorVar", "DMD", "DBF", "CGKN"]
DENS = [5, 15, 30, 50]


def stat(v):
    a = np.array([x for x in v if x is not None], float)
    return float(a.mean()), float(a.std(ddof=1)), int(a.size)


def main():
    out = {}
    for ds in DATASETS:
        p = os.path.join(SPECTRA, f"{ds}.json")
        cells = json.load(open(p))["cells"] if os.path.exists(p) else {}
        out[ds] = {}
        for m in MODELS:
            row = {}
            for d in DENS:
                c = cells.get(f"{m.lower().replace('-', '_')}:obs{d}")
                if not c or "nmb_per_run" not in c:
                    continue
                nmb, nmb_std, n = stat(c["nmb_per_run"])
                vr, vr_std, _ = stat(c["vr_per_run"])
                row[f"obs{d}"] = dict(
                    nmb=nmb, nmb_std=nmb_std, vr=vr, vr_std=vr_std, n=n
                )
            out[ds][m] = row or None
            if row.get("obs30"):
                r = row["obs30"]
                print(f"  {ds:<16}{m:<10}{r['nmb']:+.3f} / {r['vr']:.3f}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
