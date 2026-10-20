import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO)

from src.config import load_config
from src.eval.latent.tsne import DATASETS, MODELS, load

OUT = os.path.join(_REPO, "results", "latent", "distance.json")


def d_ref(tru, ana, steps):
    z_ref = tru[:, steps].reshape(-1, tru.shape[-1])
    z_a = ana[:, steps].reshape(-1, ana.shape[-1])
    ok = np.isfinite(z_a).all(axis=1)
    s_ref = np.sqrt(((z_ref - z_ref.mean(axis=0)) ** 2).sum(axis=1).mean())
    d = np.linalg.norm(z_a[ok] - z_ref[ok], axis=1) / s_ref
    q25, med, q75 = np.percentile(d, [25, 50, 75])
    return dict(
        median=float(med),
        q25=float(q25),
        q75=float(q75),
        n_pairs=int(ok.size),
        n_nonfinite=int((~ok).sum()),
    )


def main():
    out = {}
    for ds in DATASETS:
        cfg = load_config("config/da.yaml", f"{ds}_obs30")
        spin, oev = int(cfg["spinup"]), int(cfg["obs_every"])
        out[ds] = {}
        for m in MODELS:
            c = load(ds, m)
            if c is None:
                out[ds][m] = None
                continue
            T = min(c["tru"].shape[1], c["ana"].shape[1])
            steps = [t for t in range(spin, T) if t % oev == 0]
            out[ds][m] = d_ref(c["tru"], c["ana"], steps)
            r = out[ds][m]
            print(
                f"  {ds:<14}{m:<10}{r['median']:.3f} [{r['q25']:.3f}, {r['q75']:.3f}]",
                flush=True,
            )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
