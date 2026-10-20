import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASETS = ['lorenz96', 'cylinder', 'shallow_water', 'era5', 'era5_high']
SPECTRA = [f'spectra/{ds}.json' for ds in DATASETS]
TASKS = {
    'performance': ('regenerate_performance_figures.py', SPECTRA),
    'obs30': ('plot_da_obs30_per_step.py', SPECTRA),
    'interval': ('plot_lorenz_da_frequency.py', ['assimilation_interval/lorenz96.json']),
    'fields': ('plot_field_comparisons.py', ['fields/lorenz96', 'fields/era5_high', 'spectra/era5_high.json']),
    'additional_fields': ('plot_additional_physical_fields.py', ['fields/cylinder', 'fields/shallow_water', 'fields/era5']),
    'additional_psd': ('plot_additional_psd.py', SPECTRA[1:4]),
    'tsne': ('plot_tsne_per_model.py', [f'latent/tsne_{ds}.json' for ds in DATASETS]),
    'resdmd': ('plot_resdmd_residual_ratios.py', ['resdmd/residual_ratio_train_fit.json', 'resdmd/residual_ratio_own_operator.json']),
    'gaussianity': ('plot_gaussianity.py', ['gaussianity']),
    'convexity': ('plot_convexity.py', ['convexity/min_curvature.json', 'convexity/eigenvalue_spectra']),
    'timing': ('plot_wall_time_net.py', [f'eval/seed1/{ds}/speed.json' for ds in DATASETS]),
    'latent_dimension': ('plot_koopman_latent_dimension_sensitivity.py', ['latent_dimension/forecast_rrmse.json']),
    'gaussianity_methods': ('plot_gaussianity_methods.py', []),
    'hessian_curvature': ('plot_hessian_curvature.py', []),
    'cgkn_full_obs': ('plot_cgkn_full_obs_tsne.py', ['cgkn_full_obs/tsne.json']),
    'history': ('plot_history_all_datasets.py', ['observation_history/dmd.json', 'observation_history/tensorvar.json']),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plot experiment results without running model evaluation.')
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--only', nargs='+', choices=TASKS)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        for name, (script, inputs) in TASKS.items():
            print(f"{name}: {script}; inputs: {', '.join(inputs) or '(schematic)'}")
        return 0
    results = args.results_dir.expanduser().resolve()
    output = (args.output_dir or results / 'paper_figures').expanduser().resolve()
    if output == results or output in results.parents:
        parser.error('Use a separate figure directory')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KOOPMAN_RESULTS=str(results), KOOPMAN_PLOT_OUTPUT=str(output), MPLBACKEND='Agg')
    failed = []
    for name in dict.fromkeys(args.only or TASKS):
        script, inputs = TASKS[name]
        missing = [str(results / p) for p in inputs if not (results / p).exists()]
        if missing:
            print(f"{name}: missing {', '.join(missing)}", file=sys.stderr, flush=True)
            failed.append(name)
            continue
        result = subprocess.run([sys.executable, str(HERE / script)], cwd=ROOT, env=env)
        if result.returncode:
            failed.append(name)
    if failed:
        print(f"Failed plots: {', '.join(failed)}", file=sys.stderr)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
