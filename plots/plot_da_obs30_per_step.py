import json
import numpy as np
import matplotlib.pyplot as plt

from _plot_paths import FIGURES, DEFAULT_FONT, MODEL_COLORS
FIG = FIGURES

models = ['MLPAE', 'WAE', 'KAE', 'TensorVar', 'DMD', 'DBF', 'CGKN']
labels = {'MLPAE': 'NL-AE', 'WAE': 'WL-AE', 'KAE': 'KAE', 'TensorVar': 'Tensor-Var', 'DMD': 'DMD', 'DBF': 'DBF', 'CGKN': 'CGKN'}
colors = MODEL_COLORS
linestyles = {'MLPAE': '--', 'WAE': '--', 'KAE': '--', 'TensorVar': '-', 'DMD': '-', 'DBF': ':', 'CGKN': ':'}
datasets = ['lorenz96', 'cylinder', 'shallow_water', 'era5_24h', 'era5_high_24h']
dataset_labels = {'lorenz96': 'Lorenz–96', 'cylinder': 'Cylinder', 'shallow_water': 'Shallow water', 'era5_24h': 'ERA5–Coarse', 'era5_high_24h': 'ERA5–High'}

plt.rcParams.update({
    'font.family': DEFAULT_FONT, 'font.size': 14,
    'axes.titlesize': 18, 'axes.titleweight': 'bold',
    'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 15, 'legend.title_fontsize': 15,
    'axes.linewidth': 1.1, 'figure.dpi': 220, 'savefig.dpi': 220,
})

from _plot_data import performance_data
_, _, data = performance_data()

fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharex=False)
font_scale = 1.6

for j, ds in enumerate(datasets):
    for metric, row, ylabel in [('rrmse', 0, 'RRMSE'), ('ssim', 1, 'SSIM')]:
        ax = axes[row, j]
        for model in models:
            rec = data[ds]['models'][model]['da_obs30'][metric]
            x = rec['start_step'] + np.arange(rec['n_steps'])
            y = np.asarray(rec['mean'], dtype=float)
            s = np.asarray(rec['std'], dtype=float)
            ax.plot(x, y, color=colors[model], linestyle=linestyles[model], linewidth=2.4, label=labels[model])
            ax.fill_between(x, y - s, y + s, color=colors[model], alpha=0.10, linewidth=0)
        if row == 0:
            ax.set_title(dataset_labels[ds], pad=10, fontsize=18 * font_scale, fontweight='bold')
        if row == 1:
            ax.set_xlabel('Assimilation step', labelpad=5, fontsize=15 * font_scale, fontweight='bold')
        if j == 0:
            ax.set_ylabel(ylabel, labelpad=8, fontsize=15 * font_scale, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13 * font_scale, width=1.0, length=4)
        ax.grid(axis='y', alpha=.25)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

handles, leglabels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc='upper center', ncol=7, frameon=False,
           prop={'weight': 'bold', 'size': 15 * font_scale}, bbox_to_anchor=(0.5, 0.995),
           handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=1.3, h_pad=1.5)
fig.savefig(FIG / 'da_obs30_per_step.png', bbox_inches='tight')
plt.close(fig)
print('saved', FIG / 'da_obs30_per_step.png')
