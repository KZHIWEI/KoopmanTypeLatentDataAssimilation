# Paper figures

Plot the JSON and NPZ files produced by the experiment scripts in `results/`.
Run commands from the repository root:

```bash
python -m pip install -r plots/requirements.txt
python plots/run_paper_plots.py --list
python plots/run_paper_plots.py --only performance obs30
```

`--only` selects figure groups. Omit it to generate all groups after their
experiments have completed. Figures are written to `results/paper_figures/`.
Existing images with the same names are overwritten.

For results copied from a remote machine:

```bash
python plots/run_paper_plots.py --results-dir /path/to/results --output-dir /path/to/figures
```

Only the input files below are needed for plotting; model weights and training
data remain on the machine running the experiments.

## Inputs

Experiment entry points are relative to `experiments/` below. Run them with
`bash experiments/<entry point>` after preparing the data and checkpoints.
Paths in the results column are relative to `results/`.

| Figure group | Experiment entry point | Results |
|---|---|---|
| `performance`, `obs30` | `main/spectra/run.sh` | `spectra/<dataset>.json` for all five data sets |
| `fields`, `additional_fields` | `main/spectra/run.sh` | `fields/<dataset>/*.npz`; `fields` also uses `spectra/era5_high.json` |
| `additional_psd` | `main/spectra/run.sh` | `spectra/cylinder.json`, `spectra/shallow_water.json`, `spectra/era5.json` |
| `interval` | `main/assimilation_interval/run.sh` | `assimilation_interval/lorenz96.json` |
| `tsne` | `main/latent_trajectories/run.sh` | `latent/tsne_<dataset>.json` |
| `resdmd` | `main/resdmd/run.sh` | `resdmd/residual_ratio_train_fit.json`, `resdmd/residual_ratio_own_operator.json` |
| `gaussianity` | `main/gaussianity/run.sh` | `gaussianity/<dataset>/<model>.json` |
| `convexity` | `main/convexity/run.sh` | `convexity/min_curvature.json`, `convexity/eigenvalue_spectra/*_TOTAL_R0_seed*.npz` |
| `timing` | `main/benchmark/run.sh` | `eval/seed1/<dataset>/speed.json` |
| `latent_dimension` | `appendix/latent_dimension/run.sh` | `latent_dimension/forecast_rrmse.json` |
| `cgkn_full_obs` | `appendix/cgkn_full_obs/run.sh` | `cgkn_full_obs/tsne.json` |
| `history` | `appendix/observation_history/run.sh` | `observation_history/dmd.json`, `observation_history/tensorvar.json` |

`gaussianity_methods` and `hessian_curvature` are schematics and need no result
files. The overview and methodology illustrations are separate from these plots.

## Statistics and fields

- Performance curves use seed 1 and show the mean and sample standard deviation
  across test trajectories. Assimilation-versus-observation-ratio plots first
  average each trajectory over post-spinup time steps. Run the spectra experiment
  with `EVAL_SEED=1` for these figures.
- Forecast step 0 includes the initial encode/decode reconstruction. Forecast
  curves and PSD use the first requested observation ratio, 5% by default.
- Physical-field panels use one shared test trajectory, selected by the NL-AE
  analysis RRMSE closest to its test-set median at 30% observations. All methods
  show that trajectory near the midpoint of the evaluation interval. Forecasts
  propagate without analysis updates; forecast and analysis panels both use 30%
  model checkpoints. The spectra entry point enables `--fields`; run NL-AE first when exporting
  fields for selected models. Omit `--fields` for metrics and PSD only.
- Field NPZ files contain only the selected panel, run/step indices, and the
  shared truth and normalization needed for display. Cylinder and shallow water
  show channel 0 in standardized units. Both ERA5 figures show channel 4, q600,
  converted to g/kg.
- PSD curves average the channel-wise spectra over channels, trajectories and
  time; they are separate from the single physical-field case.
- The assimilation-interval figure uses seed 1 and the sample standard deviation
  across test trajectories after temporal averaging.
- ResDMD error bars use paired bootstrap ratio statistics from the experiment.
- Timing uses seed 1: forecast seconds per trajectory and assimilation seconds
  per window from the single-trajectory timing experiment.

Plotting writes images only. Missing inputs cause the selected figure group to
fail with the missing path shown in the terminal.
