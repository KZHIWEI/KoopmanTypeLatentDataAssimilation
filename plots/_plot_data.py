import json

import numpy as np

from _plot_paths import RESULTS

DATASETS = {'lorenz96': 'lorenz96', 'cylinder': 'cylinder', 'shallow_water': 'shallow_water', 'era5_24h': 'era5', 'era5_high_24h': 'era5_high', 'era5_high': 'era5_high'}
MODELS = {'MLPAE': 'nl_ae', 'WAE': 'wl_ae', 'KAE': 'kae', 'TensorVar': 'tensorvar', 'DMD': 'dmd', 'DBF': 'dbf', 'CGKN': 'cgkn'}
DISPLAY = {'MLPAE': 'NL-AE', 'WAE': 'WL-AE', 'KAE': 'KAE', 'TensorVar': 'TensorVar', 'DMD': 'DMD', 'DBF': 'DBF', 'CGKN': 'CGKN'}


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def curve_summary(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError('Expected finite trajectory-by-time values')
    return {'mean': x.mean(0).tolist(), 'std': x.std(0, ddof=1 if len(x) > 1 else 0).tolist(), 'n_steps': x.shape[1]}


def performance_data():
    forecast, performance, obs30 = {}, {'da': {}}, {}
    for label, ds in DATASETS.items():
        cells = read_json(RESULTS / 'spectra' / f'{ds}.json')['cells']
        forecast[label], obs30[label], performance['da'][label] = {'models': {}}, {'models': {}}, {}
        for model, key in MODELS.items():
            if model not in ('DBF', 'CGKN'):
                f = cells[f'{key}:forecast']
                forecast[label]['models'][model] = {metric: curve_summary(f[f'{metric}_per_run']) for metric in ('rrmse', 'ssim')}
            performance['da'][label][model] = {}
            for density in (5, 15, 30, 50):
                item = cells[f'{key}:obs{density}']
                point = {}
                for metric in ('rrmse', 'ssim'):
                    values = np.asarray(item[f'{metric}_per_run'], dtype=float)
                    summary = curve_summary(values)
                    per_run = values.mean(1)
                    point[f'{metric}_mean'] = float(per_run.mean())
                    point[f'{metric}_std'] = float(per_run.std(ddof=1 if len(per_run) > 1 else 0))
                    if density == 30:
                        summary['start_step'] = item['ssim_t0']
                        obs30[label]['models'].setdefault(model, {'da_obs30': {}})['da_obs30'][metric] = summary
                performance['da'][label][model][f'obs{density}'] = point
    return forecast, performance, obs30


def load_fields(dataset):
    ds = DATASETS[dataset]
    folder = RESULTS / 'fields' / ds
    with np.load(folder / 'selection.npz') as z:
        arrays = {'truth': z['truth']}
        run, step = int(z['run']), int(z['step'])
        mean, std = z['mean'], z['std']
    for model, key in MODELS.items():
        for suffix, leg in [('bef', 'forecast'), ('ana', 'analysis')]:
            with np.load(folder / f'{key}_{leg}.npz') as z:
                if int(z['run']) != run or int(z['step']) != step:
                    raise ValueError(f'Field selection mismatch: {ds}/{key}/{leg}')
                arrays[f'{model}_{suffix}'] = z['field']
    return arrays, {'run': run, 'step': step, 'mean': mean, 'std': std}


def psd_data(dataset):
    cells = read_json(RESULTS / 'spectra' / f'{DATASETS[dataset]}.json')['cells']
    k = np.asarray(cells['truth']['k'])
    n = float(cells['truth']['n_pixels']) ** 2
    curves = {'truth': np.asarray(cells['truth']['P']) / n}
    for model, key in MODELS.items():
        for leg in ('forecast', 'obs5', 'obs15', 'obs30', 'obs50'):
            cell = cells[f'{key}:{leg}']
            if not np.array_equal(k, cell['k']) or float(cell['n_pixels']) ** 2 != n:
                raise ValueError(f'Spectrum grid mismatch: {dataset}/{key}/{leg}')
            curves[f'{model}:{leg}'] = np.asarray(cell['P']) / n
    return k, curves
