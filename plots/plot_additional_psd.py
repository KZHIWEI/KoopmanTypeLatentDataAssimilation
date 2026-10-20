import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

from _plot_paths import FIGURES
from _plot_data import psd_data
DATASETS = ["cylinder", "shallow_water", "era5_24h"]
MODELS = ["MLPAE", "WAE", "KAE", "TensorVar", "DMD", "DBF", "CGKN"]
NAMES = ["NL-AE", "WL-AE", "KAE", "Tensor-Var", "DMD", "DBF", "CGKN"]
CURVES = [
    ("Truth", "truth", "#CC79A7", "--", 2.8),
    ("Forecast", "forecast", "#111111", ":", 2.4),
    ("obs 5%", "obs5", "#C6DBEF", "-", 2.4),
    ("obs 15%", "obs15", "#9ECAE1", "-", 2.4),
    ("obs 30%", "obs30", "#4292C6", "-", 2.4),
    ("obs 50%", "obs50", "#084594", "-", 2.4),
]


def plot_dataset(dataset):
    k, spectra = psd_data(dataset)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14, "font.weight": "bold",
        "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.linewidth": 1,
    })
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.7), sharex=True, sharey=True,
                             layout="constrained")
    axes = axes.ravel()
    limits = (min(a.min() for a in spectra.values()) / 1.3,
              max(a.max() for a in spectra.values()) * 1.3)
    ticks = [i for i in [1, 2, 5, 10, 20, 50] if k[0] <= i <= k[-1]]
    for ax, model, name in zip(axes[:7], MODELS, NAMES):
        for label, suffix, color, style, width in CURVES:
            key = "truth" if suffix == "truth" else f"{model}:{suffix}"
            ax.loglog(k, spectra[key], label=label, color=color, ls=style, lw=width,
                      zorder=10 if suffix == "truth" else 5)
        ax.set_title(name, fontsize=20, pad=9)
        ax.set_xlim(k[0], k[-1])
        ax.set_ylim(*limits)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(True, which="major", alpha=0.20)
        ax.grid(True, which="minor", alpha=0.07)
        ax.tick_params(axis="both", which="major", labelsize=13)
    for ax in axes[4:7]:
        ax.set_xlabel("Wavenumber $k$", fontsize=16)
    for ax in [axes[0], axes[4]]:
        ax.set_ylabel("Power spectral density", fontsize=16)
    axes[7].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[7].legend(handles, labels, loc="center", ncol=2, frameon=False, fontsize=15,
                   handlelength=2.6, columnspacing=1.2, labelspacing=1.2)

    output = FIGURES / f"{dataset}_psd_observation_ratios.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white")

    for ax, model in zip(axes[:7], MODELS):
        assert len(ax.lines) == 6
        for line, (_, suffix, _, _, _) in zip(ax.lines, CURVES):
            key = "truth" if suffix == "truth" else f"{model}:{suffix}"
            assert np.array_equal(line.get_xdata(), k)
            assert np.array_equal(line.get_ydata(), spectra[key])
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    for dataset in parser.parse_args().datasets:
        plot_dataset(dataset)
