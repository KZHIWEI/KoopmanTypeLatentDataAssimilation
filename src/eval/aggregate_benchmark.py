import argparse
import json
import math
import statistics
from pathlib import Path

MODELS = ["NL-AE", "WL-AE", "KAE", "TensorVar", "DBF", "CGKN", "DMD"]


def stats(values):
    if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Missing or non-finite metric; inspect the individual seed results")
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values) > 1 else None, n=len(values), values=values)


def curve_stats(curves):
    if any(not isinstance(c, list) or not c or any(not isinstance(row, list) or not row for row in c) for c in curves):
        raise ValueError("Empty or invalid trajectory curves")
    shape = (len(curves[0]), len(curves[0][0]))
    if any(len(c) != shape[0] or any(len(row) != shape[1] for row in c) for c in curves):
        raise ValueError("Trajectory curve shapes differ between seeds")
    per_seed = [[stats([row[t] for row in c])["mean"] for t in range(shape[1])] for c in curves]
    step_stats = [stats([row[t] for row in per_seed]) for t in range(shape[1])]
    return dict(mean=[s["mean"] for s in step_stats], std=[s["std"] for s in step_stats], n=len(curves), per_seed=per_seed)


def aggregate(root, dataset, seeds, models=MODELS, densities=(5, 15, 30, 50)):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 1 for s in seeds):
        raise ValueError("Seeds must be distinct positive integers")
    root = Path(root)
    result = dict(dataset=dataset, seeds=seeds, std_ddof=1, models={})
    protocol = None
    for model in models:
        selected = seeds[:1] if model == "DMD" else seeds
        records, curves = [], []
        for seed in selected:
            folder = root / f"seed{seed}" / dataset
            for kind, destination in (("result", records), ("curve", curves)):
                meta = json.loads((folder / kind / "_meta.json").read_text())
                if str(meta.get("seed")) != str(seed) or meta.get("dataset") != dataset:
                    raise ValueError(f"Seed/dataset metadata mismatch: {folder / kind}")
                current = meta.get("protocol")
                if not current or (protocol is not None and current != protocol):
                    raise ValueError(f"Evaluation protocol mismatch: {folder / kind}")
                protocol = current
                path = folder / kind / f"{model}.json"
                record = json.loads(path.read_text())
                if record.get("error") or record.get("_last_run_error") or record.get("forecast_from_previous_run"):
                    raise ValueError(f"Failed or stale evaluation: {path}")
                destination.append(record)
        row = dict(seeds=selected, shared_baseline=model == "DMD")
        row["forecast"] = {metric: stats([r["forecast"][metric] for r in records]) for metric in ("rrmse", "rrmse_post_spinup", "rrmse_final_step")}
        row["da"] = {f"obs{d}": {"rrmse": stats([r["da"][f"obs{d}"]["rrmse"] for r in records])} for d in densities}
        row["curves"] = dict(forecast=curve_stats([c["forecast"] for c in curves]), da={f"obs{d}": curve_stats([c["da"][f"obs{d}"] for c in curves]) for d in densities})
        result["models"][model] = row
    result["protocol"] = protocol
    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize independent training seeds without pooling test runs as seeds.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--root", default="results/eval")
    args = parser.parse_args()
    result = aggregate(args.root, args.dataset, args.seeds)
    path = Path(args.root) / args.dataset / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
