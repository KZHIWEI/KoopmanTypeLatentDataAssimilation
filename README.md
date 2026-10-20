# KoopmanDA

Code for **When Latent Dynamics Meet Data Assimilation: A Benchmark of Koopman-Type Representations**.

This repository benchmarks seven latent data assimilation methods: NL-AE,
WL-AE, KAE, Tensor-Var, DMD, DBF, and CGKN. Experiments cover forecasting,
data assimilation, spatial spectra, latent dynamics, Gaussianity, Hessian
curvature, and computational cost.

## Installation

Use Python 3.11 or 3.12. The full training and evaluation workflows target
Linux with Bash and an NVIDIA GPU with CUDA support.

```bash
git clone https://github.com/KZHIWEI/KoopmanTypeLatentDataAssimilation.git
cd KoopmanTypeLatentDataAssimilation
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r plots/requirements.txt
```

Check that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Run the following commands from the repository root.

## Data and pretrained models

| Data set | Command-line name | State shape |
|---|---|---|
| Lorenz–96 | `lorenz96` | 40 |
| Cylinder | `cylinder` | 2 × 64 × 64 |
| Shallow water | `shallow_water` | 3 × 96 × 192 |
| ERA5–Coarse | `era5` | 5 × 64 × 32 |
| ERA5–High | `era5_high` | 5 × 240 × 120 |

Download the prepared data and released checkpoints:

```bash
bash download_data.sh
bash download_ckpt.sh
```

For a smaller download, select a data set:

```bash
bash download_data.sh --only lorenz96
bash download_ckpt.sh --only lorenz96
```

Downloads use the Hugging Face repositories and file hashes recorded in
[`src/data/manifest.json`](src/data/manifest.json). If authentication is
required, run `hf auth login` with an account that has access to the release.
The downloader places files in `data/` and `ckpt_release/` and verifies their
SHA-256 hashes. Shared checkpoint inputs are included with filtered downloads.

Verify the downloaded files without downloading again:

```bash
bash download_data.sh --only lorenz96 --check
bash download_ckpt.sh --only lorenz96 --check
```

## Evaluate pretrained models

Start with Lorenz–96 and training seed 1:

```bash
DATASETS="lorenz96" SEEDS="1" bash experiments/main/benchmark/run.sh
```

The benchmark entry point evaluates forecasting and assimilation, measures
runtime, and aggregates the results. The default observation ratios are
5%, 15%, 30%, and 50%.

Run all five data sets with training seeds 1, 2, and 3:

```bash
bash experiments/main/benchmark/run.sh
```

Evaluate selected models directly:

```bash
EVAL_SEED=1 python -m src.eval.benchmark \
    --dataset lorenz96 --models nl_ae,wl_ae,kae,tensorvar \
    --densities 30
```

Results are written to:

```text
results/eval/
├── seed1/<dataset>/
│   ├── result/       # Scalar performance metrics
│   ├── curve/        # Per-trajectory error curves
│   ├── wall_time/    # Evaluation timings
│   └── speed.json    # Single-trajectory timing measurements
├── seed2/<dataset>/
├── seed3/<dataset>/
└── <dataset>/summary.json
```

The summary reports equally weighted means and sample standard deviations
across training seeds after averaging over test trajectories within each seed.
DMD is evaluated once as a shared baseline. Checkpoint paths are configured in
`config/eval.json`. Shared data and evaluation intervals are in
`config/dataset.yaml`. NL-AE, WL-AE, KAE, DMD, CGKN, and Lorenz–96 Tensor-Var
read `config/da_sweep.yaml`; DBF and field-data Tensor-Var read `config/da.yaml`.
Keep the corresponding entries in both DA files consistent when changing
window size, spinup, or observation interval.

Use a separate output directory for a different evaluation protocol:

```bash
DATASETS="lorenz96" SEEDS="1" EVAL_OUTPUT_ROOT="results/experiment_a" \
    bash experiments/main/benchmark/run.sh
```

## Covariance scaling sweeps

Select covariance scaling factors on the validation set and hold them fixed
for test evaluation. First scan `sigma_r` with a strong constraint, then fix
the selected value and scan `sigma_q` with a weak constraint. Compare mean
assimilation RRMSE after spinup. Keep checkpoints, observation masks, background
covariance estimation, and the evaluation protocol fixed across candidates.

The available interfaces are:

| Model family | Sweep interface | Evaluation settings |
|---|---|---|
| NL-AE, WL-AE, KAE | `src.assimilation.variational.run_sweep` | `config/sweep.json` |
| Tensor-Var, DMD | `python -m src.eval.sigma_selection` | `config/eval.json` and `results/sigma_selection.json`, respectively |

For the autoencoder family, run the following from the repository root.
Change `dataset`, `models`, and `densities` to select the experiments. Supported
data set names are `lorenz96`, `cylinder`, `shallow_water`, `era5`, and `era5_high`.

<details>
<summary>Sweep example</summary>

```bash
EVAL_SEED=1 python - <<'PY'
import json
from pathlib import Path
import torch
from src.assimilation.data import DAData
from src.assimilation.variational import run_sweep
from src.eval.benchmark import setup

dataset = "lorenz96"
models = ("nl_ae", "wl_ae", "kae")
densities = (5, 15, 30, 50)
grid = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output = Path("results/covariance_sweeps")
output.mkdir(parents=True, exist_ok=True)

for model in models:
    for density in densities:
        cfg, m, runs, sampler, Bi, length = setup(dataset, model, density, device)
        del runs
        cfg["da_data_path"] = cfg["val_path"]
        data = DAData(cfg)
        starts = data.eval_starts(cfg["eval"], int(cfg["skip"]), val=True)
        runs = data.build_runs(starts, int(cfg["skip"]), length)
        kw = dict(window_size=int(cfg["window_size"]), spinup=int(cfg["spinup"]),
                  run_length=length, obs_every=int(cfg["obs_every"]),
                  iters=int(cfg["iters"]), lr=float(cfg["lr"]),
                  converged=True, conv_iters=1600)
        strong = run_sweep(m, runs, sampler, Bi, var="sigma_r", grid=grid,
                           constraint="strong", **kw)
        weak = run_sweep(m, runs, sampler, Bi, var="sigma_q", grid=grid,
                         sr_fixed=strong["best_sr"], constraint="weak", **kw)
        record = dict(sigma_r=weak["best_sr"], sigma_q=weak["best_sq"],
                      weak_best=weak["best"], strong=strong, weak=weak)
        path = output / f"{dataset}_{model}_obs{density}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        del m, Bi, data, runs
PY
```

</details>

Copy `sigma_r`, `sigma_q`, and `weak_best` from each selected result into the
matching `<dataset>:<model>:obs<density>` entry in `config/sweep.json`.
Change `output` in the example when scanning a different checkpoint or protocol.

For the closed-form family, the command accepts comma-separated data sets,
models, and observation percentages:

```bash
EVAL_SEED=1 python -m src.eval.sigma_selection \
    lorenz96,cylinder,shallow_water,era5,era5_high tensorvar,dmd 5,15,30,50
```

This writes the candidate scores and selected factors to
`results/sigma_selection.json`. Transfer the selected `[sigma_r, sigma_q]`
pairs into the corresponding `sigma` entries in `config/eval.json` where
configured; evaluations that use the selection file read it directly.
The candidate grids are defined in `src/eval/sigma_selection.py`.
For a new checkpoint or protocol, choose a new `SIGMA_SELECTION_PATH` and use
the same path during scanning and subsequent evaluation:

```bash
export SIGMA_SELECTION_PATH="results/experiment_a/sigma_selection.json"
EVAL_SEED=1 python -m src.eval.sigma_selection lorenz96 tensorvar,dmd 5,15,30,50
DATASETS="lorenz96" SEEDS="1" EVAL_OUTPUT_ROOT="results/experiment_a/eval" \
    bash experiments/main/benchmark/run.sh
```

Apply the selected factors to the configuration entries listed above before
evaluation. `EVAL_OUTPUT_ROOT` controls evaluation outputs only; it does not
change the selection-file path.

## Training

The training entry point runs the model-specific stages for all five data sets.
Download the release assets first; the workflow uses provided initialization
checkpoints for the Lorenz–96 Tensor-Var and DBF stages.

```bash
bash experiments/main/training/run.sh
```

To train one seed:

```bash
SEEDS="1" bash experiments/main/training/run.sh
```

An individual autoencoder can be trained with:

```bash
SEED=1 python -m src.train.autoencoder \
    --config config/kae.yaml --dataset lorenz96
```

Model settings are in `config/nl_ae.yaml`, `config/wl_ae.yaml`,
`config/kae.yaml` and `config/tensorvar.yaml`. DBF and CGKN training options
are specified in the training entry point.
New checkpoints are saved under `checkpoint/`. To evaluate them, update the
corresponding paths in `config/eval.json`. Tensor-Var reads its own `fwd` and
`ckpt` fields. For retrained Lorenz–96 CGKN models,
set both `ckpt` and `noise` to the matching `cgkn@{d}.pt` and `sigma@{d}.pt`
files from the same training seed.

## Diagnostic experiments

After preparing data, checkpoints, and covariance scaling settings, run the
corresponding script:

| Experiment | Entry point |
|---|---|
| Assimilation interval | `bash experiments/main/assimilation_interval/run.sh` |
| Spatial spectra and bias/variance | `bash experiments/main/spectra/run.sh` |
| Latent trajectories | `bash experiments/main/latent_trajectories/run.sh` |
| ResDMD residuals | `bash experiments/main/resdmd/run.sh` |
| Gaussianity | `bash experiments/main/gaussianity/run.sh` |
| Hessian curvature | `bash experiments/main/convexity/run.sh` |
| Tensor-Var covariance diagnostics | `bash experiments/main/tensorvar_covariance/run.sh` |

Additional experiments are in `experiments/appendix/`, including observation
history, latent dimension, full-observation CGKN, parameter counts, and NMC
conditioning.

## Figures

Plotting reads the JSON and NPZ files produced by the experiments in `results/`.
The plotting commands write images to `results/paper_figures/`.

For the forecast, assimilation-performance, physical-field, and PSD figures:

```bash
EVAL_SEED=1 bash experiments/main/spectra/run.sh
python plots/run_paper_plots.py --only performance obs30 fields additional_fields additional_psd
```

This experiment computes RRMSE and SSIM from the same forecast and analysis
fields and saves per-trajectory curves. It also saves the selected physical-field
panels at 30% observations. The performance figures use seed 1, with means and
sample standard deviations across test trajectories. PSD curves average spectra
across channels, trajectories, and post-spinup times; forecast spectra include
the full forecast interval.

For a single data set:

```bash
EVAL_SEED=1 python -m src.eval.spectra.spectrum --dataset lorenz96 --fields
```

The multi-data-set figure commands require the corresponding data sets to be
complete. Run the other diagnostic experiments for their respective figures;
see [`plots/README.md`](plots/README.md) for the input-to-experiment mapping.

```bash
python plots/run_paper_plots.py --list
python plots/run_paper_plots.py --only tsne resdmd
```

After completing all experiments, generate all figures:

```bash
python plots/run_paper_plots.py
```

Use `--results-dir /path/to/results` to plot results copied from another machine
and `--output-dir /path/to/figures` to choose the image destination.

To plot the Lorenz–96 assimilation-interval results:

```bash
python -m src.ablation.assimilation_interval.run --aggregate-only
python plots/run_paper_plots.py --only interval
```

This figure uses seed 1, with means and sample standard deviations across test
trajectories after averaging each trajectory over post-spinup time steps.

## Repository structure

```text
config/           Data, model, and evaluation settings
src/data/         Data loading and release downloads
src/model/        Model architectures
src/train/        Training routines
src/assimilation/  Variational and filtering solvers
src/eval/         Evaluation and diagnostics
src/ablation/     Supplementary experiments
experiments/      Experiment entry points
plots/            Figure generation
```

## License

Original code in this repository is released under the [MIT License](LICENSE).
Third-party copyright and license notices are provided in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
This license applies to code; data and pretrained models are not covered by it.
