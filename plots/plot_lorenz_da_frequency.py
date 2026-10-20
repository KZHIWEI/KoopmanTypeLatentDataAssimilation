import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _plot_paths import RESULTS, FIGURES, DEFAULT_FONT, MODEL_COLORS
SOURCE = RESULTS / "assimilation_interval/lorenz96.json"
OUTPUT = FIGURES / "lorenz96_assimilation_frequency"
MODELS = ["nl_ae", "wl_ae", "kae", "tensorvar", "dmd", "dbf", "cgkn"]
LABELS = ["NL-AE", "WL-AE", "KAE", "Tensor-Var", "DMD", "DBF", "CGKN"]
COLORS = [MODEL_COLORS[m] for m in ("MLPAE", "WAE", "KAE", "TensorVar", "DMD", "DBF", "CGKN")]
STYLES = ["--", "--", "--", "-", "-", ":", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
RATIOS = [5, 15, 30, 50]
INTERVALS = [1, 2, 3, 5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    data = json.loads(args.source.read_text(encoding="utf-8"))

    checked = 0
    for model in MODELS:
        for ratio in RATIOS:
            for interval in INTERVALS:
                cell = data[model][f"obs{ratio}"][f"oe{interval}"]
                if cell["std_over"] != "test_trajectories" or cell["std_ddof"] != 1 or cell["seed"] != 1:
                    raise ValueError(f"Invalid aggregation: {model}, {ratio}, {interval}")
                values = np.asarray(cell["per_run_rrmse"], dtype=float)
                if len(values) != cell["n"] or not np.isfinite(values).all():
                    raise ValueError(f"Invalid values: {model}, {ratio}, {interval}")
                std = values.std(ddof=1) if len(values) > 1 else 0.0
                if not np.isclose(values.mean(), cell["mean"], atol=1.1e-6, rtol=0):
                    raise ValueError(f"Mean mismatch: {model}, {ratio}, {interval}")
                if not np.isclose(std, cell["std"], atol=1.1e-6, rtol=0):
                    raise ValueError(f"Standard deviation mismatch: {model}, {ratio}, {interval}")
                checked += 1

    plt.rcParams.update({
        "font.family": DEFAULT_FONT, "font.size": 15,
        "axes.titlesize": 17, "axes.titleweight": "bold", "axes.labelsize": 16,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.8), sharex=True, sharey=True)
    for panel, (ax, ratio) in enumerate(zip(axes, RATIOS)):
        for model, label, color, style, marker in zip(MODELS, LABELS, COLORS, STYLES, MARKERS):
            cells = [data[model][f"obs{ratio}"][f"oe{i}"] for i in INTERVALS]
            mean = np.array([c["mean"] for c in cells])
            std = np.array([c["std"] for c in cells])
            ax.errorbar(INTERVALS, mean, yerr=std, label=label, color=color, linestyle=style,
                        marker=marker, markersize=7, linewidth=2.4,
                        markeredgecolor="white", markeredgewidth=0.6,
                        capsize=3, capthick=1.0, elinewidth=1.0)
        ax.set_title(f"({chr(97 + panel)}) Observation ratio: {ratio}%", pad=10)
        ax.set_xlabel("Assimilation interval (steps)", labelpad=7)
        ax.set_xticks(INTERVALS)
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(0, 1.62)
        ax.set_yticks(np.arange(0, 1.61, 0.2))
        ax.grid(axis="y", color="#D5D5D5", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("DA RRMSE", labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=7, frameon=False, fontsize=15, handlelength=2.5, columnspacing=1.6)
    fig.suptitle("Lorenz–96: assimilation interval sensitivity", fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.15, top=0.74, wspace=0.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    path = OUTPUT.with_suffix(".png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close(fig)
    print(f"Validated trajectory means and sample SD for {checked} cells.")


if __name__ == "__main__":
    main()
