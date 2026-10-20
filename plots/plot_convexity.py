import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _plot_paths import RESULTS, FIGURES, DEFAULT_FONT
INPUT = RESULTS / "convexity"
from _plot_data import DATASETS as DATASET_KEYS
PER_TRAJ = INPUT / "min_curvature.json"
SPECTRUM_DIR = INPUT / "eigenvalue_spectra"
OUTPUT_TERMS = FIGURES / "convexity_terms.png"
OUTPUT_SPECTRUM = FIGURES / "convexity_spectrum.png"

DATASETS = ["lorenz96", "cylinder", "shallow_water", "era5_24h", "era5_high_24h"]
DATASET_LABELS = {
    "lorenz96": "Lorenz-96", "cylinder": "Cylinder", "shallow_water": "Shallow water",
    "era5_24h": "ERA5–Coarse", "era5_high_24h": "ERA5–High",
}
SPECTRUM_STEMS = {
    "lorenz96": "lorenz96", "cylinder": "cylinder",
    "shallow_water": "shallow_water", "era5_24h": "era5_24h",
    "era5_high_24h": "era5_high_24h",
}
MODELS = ["mlpae", "wae", "kae", "tensorvar", "dmd"]
MODEL_LABELS = {
    "mlpae": "NL-AE", "wae": "WL-AE", "kae": "KAE",
    "tensorvar": "Tensor-Var", "dmd": "DMD",
}

MODEL_TERM_COLOR = "#2C7FB8"
OBS_TERM_COLOR = "#F28E2B"
NEGATIVE_COLOR = "#C44E52"
ZERO_COLOR = "#A6A6A6"
POSITIVE_COLOR = "#4C78A8"
GRID_COLOR = "#D9D9D9"

plt.rcParams.update({
    "font.family": DEFAULT_FONT, "font.size": 14,
    "axes.titlesize": 18, "axes.titleweight": "bold",
    "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 15, "axes.linewidth": 1.1,
    "figure.dpi": 240, "savefig.dpi": 300,
})


def load_cells():
    with PER_TRAJ.open("r", encoding="utf-8") as handle:
        return json.load(handle)["cells"]


def signed_summary(cells, dataset, model, term):
    key = f"{DATASET_KEYS[dataset]}:{dict(mlpae='nl_ae', wae='wl_ae').get(model, model)}:obs30:background"
    item = cells[key]["terms"][term]
    return float(item["signed_mean"]), float(item["signed_std"])


def style_panel(ax):
    ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1.25, zorder=1)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.72, zorder=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["−1", "−0.5", "0", "+0.5", "+1"])
    ax.tick_params(axis="y", length=0, pad=7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#777777")


def plot_term_sources(cells):
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.2), sharex=True)
    axes = axes.ravel()
    offsets = {"q_model_error": -0.16, "r_observation": 0.16}
    specs = [
        ("q_model_error", "Model-error term", MODEL_TERM_COLOR, "o"),
        ("r_observation", "Observation-error term", OBS_TERM_COLOR, "s"),
    ]
    y = np.arange(len(MODELS))

    for panel, dataset in enumerate(DATASETS):
        ax = axes[panel]
        style_panel(ax)
        for term, _, color, marker in specs:
            centers, errors = [], []
            for model in MODELS:
                center, error = signed_summary(cells, dataset, model, term)
                centers.append(center)
                errors.append(error)
            ax.errorbar(
                centers, y + offsets[term], xerr=errors, fmt=marker,
                markersize=(10.5 if marker == "o" else 8.5), markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.7, ecolor=color, elinewidth=1.55,
                capsize=3.2, capthick=1.15, linestyle="none", zorder=3,
            )
        for boundary in y[:-1] + 0.5:
            ax.axhline(
                boundary,
                color="#C8C8C8",
                linestyle=(0, (3, 3)),
                linewidth=0.9,
                alpha=0.85,
                zorder=0,
            )
        ax.set_title(DATASET_LABELS[dataset], pad=9, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontweight="bold")
        ax.invert_yaxis()
        if panel % 3 != 0:
            ax.tick_params(axis="y", labelleft=False)
        if panel >= 3:
            ax.set_xlabel(r"Normalized minimum curvature  $s$")

    legend_ax = axes[5]
    legend_ax.axis("off")
    legend_handles = [
        Line2D([0], [0], marker=marker, color=color, linewidth=1.6,
               markersize=7, markerfacecolor=color, markeredgecolor="white", label=label)
        for _, label, color, marker in specs
    ]
    legend_ax.legend(handles=legend_handles, loc="center", bbox_to_anchor=(0.5, 0.52),
                     frameon=False, fontsize=15, handlelength=2.0, labelspacing=1.1)

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.085, top=0.95,
                        wspace=0.22, hspace=0.28)
    fig.savefig(OUTPUT_TERMS, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def classify_spectrum(dataset, model, safety_factor=1.0):
    key = dict(mlpae='nl_ae', wae='wl_ae').get(model, model)
    sources = sorted(SPECTRUM_DIR.glob(f'{DATASET_KEYS[dataset]}_{key}_obs30_TOTAL_R0_seed*.npz'))
    if len(sources) != 1:
        raise ValueError(f'Expected one background Hessian spectrum for {dataset}/{model}, got {len(sources)}')
    with np.load(sources[0]) as item:
        values = np.asarray(item['eigenvalues'], dtype=float)
        asymmetry = np.asarray(item['hvp_asymmetry'], dtype=float)
        eps = np.finfo(np.dtype(str(item['dtype_of_block']))).eps
        band = safety_factor * np.maximum(asymmetry, int(item['block_dim']) * eps)
    tolerance = band * np.maximum(np.abs(values).max(1), np.finfo(float).tiny)
    return np.stack([(values < -tolerance[:, None]).mean(1), (np.abs(values) <= tolerance[:, None]).mean(1), (values > tolerance[:, None]).mean(1)], axis=1)


def plot_complete_spectrum(safety_factor=1.0, output_png=OUTPUT_SPECTRUM):
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 8.7), sharex=True)
    axes = axes.ravel()
    y = np.arange(len(MODELS))
    categories = [
        ("Negative", NEGATIVE_COLOR),
        ("Approximately zero", ZERO_COLOR),
        ("Positive", POSITIVE_COLOR),
    ]

    for panel, dataset in enumerate(SPECTRUM_STEMS):
        ax = axes[panel]
        means = []
        boundary_means = []
        boundary_stds = []
        for model in MODELS:
            proportions = classify_spectrum(dataset, model, safety_factor=safety_factor)
            means.append(proportions.mean(axis=0))
            cumulative = np.cumsum(proportions[:, :2], axis=1)
            boundary_means.append(cumulative.mean(axis=0))
            boundary_stds.append(cumulative.std(axis=0, ddof=0))
        means = np.asarray(means)
        boundary_means = np.asarray(boundary_means)
        boundary_stds = np.asarray(boundary_stds)
        left = np.zeros(len(MODELS))
        for index, (_, color) in enumerate(categories):
            ax.barh(y, means[:, index] * 100.0, left=left, height=0.58,
                    color=color, edgecolor="white", linewidth=0.6, zorder=2)
            left += means[:, index] * 100.0
        for boundary in range(2):
            ax.errorbar(
                boundary_means[:, boundary] * 100.0,
                y,
                xerr=boundary_stds[:, boundary] * 100.0,
                fmt="none",
                ecolor="#333333",
                elinewidth=1.15,
                capsize=3.0,
                capthick=1.0,
                zorder=4,
            )
        ax.set_title(DATASET_LABELS[dataset], pad=9, fontweight="bold")
        ax.set_xlim(0.0, 100.0)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.72, zorder=0)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontweight="bold")
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0, pad=7)
        if panel % 3 != 0:
            ax.tick_params(axis="y", labelleft=False)
        if panel >= 3:
            ax.set_xlabel("Proportion of Hessian eigenvalues (%)")
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#777777")


    legend_ax = axes[5]
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label)
                 for label, color in categories],
        loc="center", frameon=False, fontsize=15, labelspacing=1.0,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.95,
                        wspace=0.22, hspace=0.30)
    fig.savefig(output_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cells = load_cells()
    plot_term_sources(cells)
    plot_complete_spectrum()
    print(OUTPUT_TERMS)
    print(OUTPUT_SPECTRUM)
