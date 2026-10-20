from _plot_paths import RESULTS, FIGURES, MODEL_COLORS
from _plot_data import DATASETS as DATASET_KEYS, DISPLAY
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
DATASETS = ['lorenz96', 'cylinder', 'shallow_water', 'era5_24h', 'era5_high']
TITLES = ['Lorenz–96', 'Cylinder', 'Shallow water', 'ERA5–Coarse', 'ERA5–High']
MODELS = ['MLPAE', 'WAE', 'KAE', 'TensorVar', 'DMD', 'DBF', 'CGKN']
LABELS = ['NL-AE', 'WL-AE', 'KAE', 'Tensor-Var', 'DMD', 'DBF', 'CGKN']
COLORS = [MODEL_COLORS[m] for m in MODELS]


def main(variants=('fitted', 'exact')):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18,
                         'axes.titlesize': 24, 'axes.titleweight': 'bold',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    for variant, title in [('fitted', 'Fitted operators'), ('exact', 'Model-specific operators')]:
        if variant not in variants:
            continue
        source = RESULTS / 'resdmd' / ('residual_ratio_train_fit.json' if variant == 'fitted' else 'residual_ratio_own_operator.json')
        raw = json.loads(source.read_text(encoding='utf-8'))
        models = MODELS if variant == 'fitted' else ['KAE', 'TensorVar', 'DMD', 'DBF']
        fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
        axes = axes.ravel()
        fig.suptitle(title, fontsize=28, fontweight='bold', y=.99)
        for panel, (ax, dataset, name) in enumerate(zip(axes, DATASETS, TITLES)):
            ax.set_xscale('log')
            ax.set_xlim(.2, 5)
            ax.xaxis.set_major_locator(FixedLocator([.25, .5, 1, 2, 4]))
            ax.xaxis.set_major_formatter(FixedFormatter(['0.25', '0.5', '1', '2', '4']))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.axvspan(.2, 1, color='#EAF1F8', alpha=.48, zorder=0)
            ax.axvspan(1, 5, color='#FAECEA', alpha=.42, zorder=0)
            ax.axvline(1, color='#555555', linewidth=1.8, linestyle='--', zorder=1)
            for y, model in enumerate(models):
                key = f"{DATASET_KEYS[dataset]}_{DISPLAY[model]}"
                cell = raw[key]
                b = cell['ratio_bootstrap']
                if cell['density'] != 30 or cell['operator_kind'] != ('khat_train_dense' if variant == 'fitted' else 'exact'):
                    raise ValueError(f'Unexpected residual experiment: {key}')
                mu, sd = float(b['mean']), float(b['std'])
                if mu <= 0 or not np.isfinite([mu, sd]).all():
                    raise ValueError(f'Invalid residual ratio: {key}')
                ax.set_xlim(min(ax.get_xlim()[0], max(mu - sd, 1e-6) / 1.1), max(ax.get_xlim()[1], (mu + sd) * 1.1))
                color = COLORS[MODELS.index(model)]
                ax.hlines(y, 1, mu, color=color, linewidth=2.6, alpha=.9, zorder=2)
                ax.errorbar(mu, y, xerr=sd, fmt='o', color=color, markersize=14,
                            markeredgecolor='white', markeredgewidth=.9,
                            elinewidth=2.4, capsize=5, capthick=2, zorder=3)
            ax.set_yticks(range(len(models)), [LABELS[MODELS.index(m)] for m in models])
            if panel % 3 != 0:
                ax.tick_params(axis='y', labelleft=False)
            ax.set_ylim(len(models) - .5, -.5)
            ax.set_title(name, pad=15)
            ax.set_xlabel(r'$R_{\mathrm{res}}$', fontsize=22)
            ax.grid(axis='x', color='#777777', alpha=.18, linewidth=.8)
            ax.tick_params(axis='y', length=0, pad=6)
            ax.spines['left'].set_visible(False)
        axes[5].axis('off')
        for y, label, description, color in [
            (.70, r'$R_{\mathrm{res}}<1$', 'Greater consistency after assimilation', '#315F8A'),
            (.39, r'$R_{\mathrm{res}}>1$', 'Reduced consistency after assimilation', '#9A4036')]:
            axes[5].text(.04, y, label, fontsize=22, color=color, transform=axes[5].transAxes)
            axes[5].text(.04, y-.11, description, fontsize=17, color=color, transform=axes[5].transAxes)
        fig.tight_layout(rect=[.02, .02, 1, 1], w_pad=1.7, h_pad=1.9)
        stem = f'resdmd_{"fitted" if variant == "fitted" else "model_specific"}_ratio'
        fig.savefig(FIGURES / f'{stem}.png', dpi=300,
                    bbox_inches='tight', pad_inches=.05)
        plt.close(fig)
        print(FIGURES / f'{stem}.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['fitted', 'exact', 'both'], default='both')
    args = parser.parse_args()
    main(('fitted', 'exact') if args.variant == 'both' else (args.variant,))
