# Main benchmark: independent training seeds

Run from the repository root after installing dependencies and downloading the
required data and released checkpoints (the paths in `config/eval.json`).
The entry point requires Bash, for example on Linux or under WSL.

```bash
bash experiments/main/benchmark/run.sh
```

This evaluates neural models with training seeds 1, 2, and 3, including speed
measurements. DMD has no training seed and is evaluated once, with the first
requested seed. Observation masks and test trajectories retain the existing
protocol; changing a training seed selects the corresponding checkpoint.

Outputs are isolated by seed:

```text
results/eval/seed1/<dataset>/{result,curve,wall_time}/
results/eval/seed1/<dataset>/speed.json
results/eval/seed2/<dataset>/...
results/eval/seed3/<dataset>/...
results/eval/<dataset>/summary.json
```

`summary.json` contains scalar RRMSE and forecast/analysis RRMSE curves. Each seed
has equal weight. Curves are averaged over test trajectories within each seed
before calculating the cross-seed mean and sample standard deviation (`ddof=1`).
Individual seed values are retained. DMD is marked `shared_baseline: true`, has
one measurement, and has `std: null`; it is not treated as three independent
training runs. Timing measurements remain in the individual seed outputs.

Missing/failed runs, stale failed-run records, incompatible protocols, curve
shape mismatches, and non-finite metrics prevent summary generation. A failed
evaluation exits with a nonzero status. Older unseeded outputs are not imported
or overwritten; rerun them with an explicit seed.

To evaluate only one dataset, or use another output root:

```bash
DATASETS="lorenz96" EVAL_OUTPUT_ROOT="results/check" bash experiments/main/benchmark/run.sh
```

`SEEDS="1 2 3"` is the default and may be overridden. To regenerate a summary
from completed runs without reevaluating models:

```bash
python -m src.eval.aggregate_benchmark --dataset lorenz96 --seeds 1 2 3
```

A direct single-seed evaluation also uses a seed-specific output directory:

```bash
EVAL_SEED=2 python -m src.eval.benchmark --dataset lorenz96 --models kae
```
