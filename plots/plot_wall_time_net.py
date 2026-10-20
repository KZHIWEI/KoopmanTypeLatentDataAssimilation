import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterMathtext, LogLocator
import numpy as np

from _plot_paths import RESULTS, FIGURES, MODEL_COLORS
from _plot_data import DATASETS as DATASET_KEYS, DISPLAY, read_json
OUTPUT_PNG = FIGURES / "computational_cost.png"

DATASETS = ["lorenz96", "cylinder", "shallow_water", "era5_24h", "era5_high_24h"]
DATASET_LABELS = {
    "lorenz96": "Lorenz-96", "cylinder": "Cylinder", "shallow_water": "Shallow water",
    "era5_24h": "ERA5–Coarse", "era5_high_24h": "ERA5–High",
}
MODELS = ["MLPAE", "WAE", "KAE", "TensorVar", "DMD", "DBF", "CGKN"]
MODEL_LABELS = {
    "MLPAE": "NL-AE", "WAE": "WL-AE", "KAE": "KAE",
    "TensorVar": "Tensor-Var", "DMD": "DMD", "DBF": "DBF", "CGKN": "CGKN",
}
COLORS = MODEL_COLORS
OBS_RATIO = 30

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": 24, "axes.titleweight": "bold", "axes.labelsize": 15,
    "xtick.labelsize": 13.5, "ytick.labelsize": 22, "axes.linewidth": 1.0,
    "figure.dpi": 240, "savefig.dpi": 300,
})


def load_wall_time():
    return {ds: read_json(RESULTS / 'eval' / 'seed1' / DATASET_KEYS[ds] / 'speed.json') for ds in DATASETS}


class BoldLogFormatter(LogFormatterMathtext):
    def __call__(self, value, pos=None):
        return super().__call__(value, pos).replace(r"\mathdefault", r"\mathbf")


def style_axis(ax):
    ax.set_yscale("log")
    ax.grid(axis="y", which="major", color="#D7D7D7", linewidth=0.8, alpha=0.72, zorder=0)
    ax.grid(False, which="minor")
    ax.tick_params(axis="both", width=0.9, length=3.5)
    ax.tick_params(axis="y", which="minor", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_figure():
    data = load_wall_time()
    forecast = {}
    assimilation = {}
    for dataset in DATASETS:
        record = data[dataset]
        n_windows = int(np.ceil((record['protocol']['run_length'] - record['protocol']['spinup']) / record['protocol']['window']))
        forecast[dataset] = np.asarray([record['models'][DISPLAY[m]]['forecast'] for m in MODELS], dtype=float)
        assimilation[dataset] = np.asarray([record['models'][DISPLAY[m]]['inference'] / n_windows for m in MODELS], dtype=float)

    forecast_all = np.concatenate(list(forecast.values()))
    da_all = np.concatenate(list(assimilation.values()))
    forecast_ylim = (float(forecast_all.min()) * 0.45, float(forecast_all.max()) * 3.0)
    da_ylim = (float(da_all.min()) * 0.45, float(da_all.max()) * 3.0)
    limits = [forecast_ylim, da_ylim]

    fig, axes = plt.subplots(2, 5, figsize=(22.5, 9.0), sharey="row")
    x = np.arange(len(MODELS))
    colors = [COLORS[model] for model in MODELS]

    for column, dataset in enumerate(DATASETS):
        for row, series in enumerate((forecast, assimilation)):
            ax = axes[row, column]
            ax.bar(x, series[dataset], width=0.72, color=colors,
                   edgecolor="white", linewidth=0.7, zorder=3)
            style_axis(ax)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
            ax.yaxis.set_major_formatter(BoldLogFormatter(base=10))
            ax.set_xticks([])
            ax.set_ylim(*limits[row])
        axes[0, column].set_title(DATASET_LABELS[dataset], pad=12)

    fig.subplots_adjust(left=0.06, right=0.992, bottom=0.13, top=0.83,
                        wspace=0.10, hspace=0.45)
    fig.text(0.06, 0.965, "Forecast wall time (s/trajectory)",
             ha="left", va="center", fontsize=28, fontweight="bold")
    fig.text(0.06, 0.48, "Assimilation wall time (s/window)",
             ha="left", va="center", fontsize=28, fontweight="bold")
    handles = [Patch(facecolor=COLORS[model], label=MODEL_LABELS[model])
               for model in MODELS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.526, 0.01),
               ncol=7, frameon=False, fontsize=28, handlelength=1.2,
               handletextpad=0.4, columnspacing=1.25, borderaxespad=0.0)
    return fig


def main():
    fig = build_figure()
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
