import json
import numpy as np
import matplotlib.pyplot as plt

from _plot_paths import FIGURES, DEFAULT_FONT, MODEL_COLORS, forecast_steps
FIG = FIGURES
FIG.mkdir(exist_ok=True)

models = ['MLPAE','WAE','KAE','TensorVar','DMD','DBF','CGKN']
forecast_models = ['MLPAE','WAE','KAE','TensorVar','DMD']
labels = {'MLPAE':'NL-AE','WAE':'WL-AE','KAE':'KAE','TensorVar':'Tensor-Var','DMD':'DMD','DBF':'DBF','CGKN':'CGKN'}
forecast_colors = MODEL_COLORS
forecast_linestyles = {'MLPAE':'--','WAE':'--','KAE':'--','TensorVar':'-','DMD':'-','DBF':':','CGKN':':'}
datasets = ['lorenz96','cylinder','shallow_water','era5_24h','era5_high_24h']
dataset_labels = {'lorenz96':'Lorenz-96','cylinder':'Cylinder','shallow_water':'Shallow water','era5_24h':'ERA5–Coarse','era5_high_24h':'ERA5–High'}

plt.rcParams.update({
    'font.family':DEFAULT_FONT, 'font.size':14,
    'axes.titlesize':18, 'axes.titleweight':'bold',
    'axes.labelsize':15, 'xtick.labelsize':13, 'ytick.labelsize':13,
    'legend.fontsize':15, 'legend.title_fontsize':15,
    'axes.linewidth':1.1, 'figure.dpi':220, 'savefig.dpi':220,
})


from _plot_data import performance_data
forecast, perf, _ = performance_data()
fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharex=False)
forecast_font = 1.6
for j, ds in enumerate(datasets):
    for metric, row, ylabel in [('rrmse',0,'RRMSE'), ('ssim',1,'SSIM')]:
        ax = axes[row,j]
        for m in forecast_models:
            item = forecast[ds]['models'][m][metric]
            y = np.asarray(item['mean'], float)
            s = np.asarray(item['std'], float)
            x = np.asarray(forecast_steps(len(y)))
            ax.plot(x, y, lw=2.4, color=forecast_colors[m], linestyle=forecast_linestyles[m], label=labels[m], marker=None)
            ax.fill_between(x, y-s, y+s, color=forecast_colors[m], alpha=0.10, linewidth=0)
        if row == 0:
            ax.set_title(dataset_labels[ds], pad=10, fontsize=18*forecast_font, fontweight='bold')
        if row == 1: ax.set_xlabel('Forecast step', labelpad=5, fontsize=15*forecast_font, fontweight='bold')
        if j == 0: ax.set_ylabel(ylabel, labelpad=8, fontsize=15*forecast_font, fontweight='bold')
        ax.tick_params(axis='both', labelsize=13*forecast_font, width=1.0, length=4)
        ax.grid(axis='y', alpha=.25)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
handles, leglabels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc='upper center', ncol=5, frameon=False, prop={'weight':'bold','size':15*forecast_font}, bbox_to_anchor=(0.5,0.995), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=[0,0,1,0.93], w_pad=1.3, h_pad=1.5)
fig.savefig(FIG/'forecast_per_step.png', bbox_inches='tight')
plt.close(fig)


obs = ['obs5','obs15','obs30','obs50']
obs_labels = ['5%','15%','30%','50%']
fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharex=False)
da_font = forecast_font
for j, ds in enumerate(datasets):
    for metric, row, ylabel in [('rrmse',0,'RRMSE'), ('ssim',1,'SSIM')]:
        ax = axes[row,j]
        x = np.arange(len(obs))
        for m in models:
            vals = perf['da'][ds][m]
            y = np.asarray([vals[o][metric+'_mean'] for o in obs], float)
            s = np.asarray([vals[o][metric+'_std'] for o in obs], float)
            ax.errorbar(x, y, yerr=s, lw=2.1, ms=5.5, marker="o", capsize=3, color=forecast_colors[m], linestyle=forecast_linestyles[m], label=labels[m])
        ax.set_title(dataset_labels[ds], pad=10, fontsize=18*da_font, fontweight='bold')
        ax.set_xticks(x, obs_labels)
        ax.set_xlabel('Observation ratio', labelpad=5, fontsize=15*da_font, fontweight='bold')
        if j == 0: ax.set_ylabel(ylabel, labelpad=8, fontsize=15*da_font, fontweight='bold')
        ax.grid(axis='y', alpha=.25)
        ax.tick_params(axis='both', labelsize=13*da_font, width=1.0, length=4)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
handles, leglabels = axes[0,0].get_legend_handles_labels()
fig.legend(handles, leglabels, loc='upper center', ncol=7, frameon=False, prop={'weight':'bold','size':15*da_font}, bbox_to_anchor=(0.5,0.995), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=[0,0,1,0.93], w_pad=1.3, h_pad=1.5)
fig.savefig(FIG/'da_performance.png', bbox_inches='tight')
plt.close(fig)

print('regenerated forecast and DA performance figures')
