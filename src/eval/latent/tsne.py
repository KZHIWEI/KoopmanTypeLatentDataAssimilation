import argparse
import json
import os

import numpy as np
from sklearn.manifold import TSNE

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
CACHE = os.path.join(_REPO, "results", "latent", "cache")
OUT = os.path.join(_REPO, "results", "latent", "tsne_{ds}.json")
DATASETS = ["lorenz96", "cylinder", "shallow_water", "era5", "era5_high"]
MODELS = ["NL-AE", "WL-AE", "KAE", "TensorVar", "DMD", "DBF", "CGKN"]
TSNE_KW = dict(
    n_components=2,
    early_exaggeration=12.0,
    learning_rate="auto",
    max_iter=1000,
    metric="euclidean",
    init="pca",
    method="barnes_hut",
    angle=0.5,
    random_state=0,
)


def load(ds, model):
    p = os.path.join(CACHE, f"{ds}_{model}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    return dict(
        tru=d["tru"],
        bef=None if d["bef"].size == 0 else d["bef"],
        ana=d["ana"],
        meta=json.loads(str(d["meta"])),
    )


def embed(legs):
    ref = legs["reference"]
    ctr = ref.mean(axis=0, keepdims=True)
    rad = 10.0 * max(np.linalg.norm(ref - ctr, axis=1).max(), 1e-9)
    kept = {}
    for k, x in legs.items():
        if x is None:
            continue
        x = x - ctr
        ok = np.isfinite(x).all(axis=1)
        if not ok.all():
            x = x[: int(np.argmin(ok))]
        if len(x):
            n = np.linalg.norm(x, axis=1, keepdims=True)
            kept[k] = x * np.minimum(1.0, rad / np.maximum(n, 1e-9))
    joint = np.vstack(list(kept.values()))
    n = joint.shape[0]
    e = TSNE(perplexity=min(30, max(5, n // 4)), **TSNE_KW).fit_transform(joint)
    parts = np.split(e, np.cumsum([len(v) for v in kept.values()])[:-1])
    return {k: np.round(p, 6).tolist() for k, p in zip(kept, parts)}


def median_run(cells):
    ranks = []
    for c in cells.values():
        num = np.linalg.norm(c["ana"] - c["tru"], axis=2)
        den = np.maximum(np.linalg.norm(c["tru"], axis=2), 1e-12)
        ranks.append(np.argsort(np.argsort(np.nanmean(num / den, axis=1))))
    n = min(len(r) for r in ranks)
    avg = np.mean([r[:n] for r in ranks], axis=0)
    return int(np.argsort(avg)[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    a = ap.parse_args()
    cells = {m: c for m in MODELS if (c := load(a.dataset, m)) is not None}
    j = median_run(cells)
    out = {"dataset": a.dataset, "run": j, "tsne": TSNE_KW, "models": {}}
    for m, c in cells.items():
        out["models"][m] = embed(
            {
                "reference": c["tru"][j],
                "forecast": None if c["bef"] is None else c["bef"][j],
                "analysis": c["ana"][j],
            }
        )
        print(f"  {a.dataset} {m} run {j}", flush=True)
    p = OUT.format(ds=a.dataset)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
