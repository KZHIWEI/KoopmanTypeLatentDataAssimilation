import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


from _plot_paths import RESULTS, FIGURES, MODEL_COLORS
INPUT = RESULTS / "gaussianity"
from _plot_data import DATASETS as DATASET_KEYS, DISPLAY as MODEL_KEYS
OUTPUT = FIGURES / "gaussianity.png"

DATASETS = [
    "lorenz96",
    "cylinder",
    "shallow_water",
    "era5_24h",
    "era5_high_24h",
]
DATASET_LABELS = {
    "lorenz96": "Lorenz-96",
    "cylinder": "Cylinder",
    "shallow_water": "Shallow water",
    "era5_24h": "ERA5–Coarse",
    "era5_high_24h": "ERA5–High",
}
MODELS = ["MLPAE", "WAE", "KAE", "TensorVar", "DMD", "DBF", "CGKN"]
MODEL_LABELS = {
    "MLPAE": "NL-AE",
    "WAE": "WL-AE",
    "KAE": "KAE",
    "TensorVar": "Tensor-Var",
    "DMD": "DMD",
    "DBF": "DBF",
    "CGKN": "CGKN",
}
GRID_COLOR = "#D9D9D9"
ZERO_COLOR = "#666666"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 20,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 19,
        "axes.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def fisher_ratio(p_values):
    p = np.clip(np.asarray(p_values, dtype=float), 1e-300, 1.0)
    statistic = -2.0 * np.log(p).sum()
    return statistic / (2.0 * len(p))


def load_data():
    data = {}
    for dataset in DATASETS:
        for model in MODELS:
            source = INPUT / DATASET_KEYS[dataset] / f"{MODEL_KEYS[model]}.json"
            with source.open("r", encoding="utf-8") as handle:
                item = json.load(handle)

            starts = item["per_start"]
            delta_kurt = np.asarray(
                [
                    row["analysis"]["proj_B"]["mardia_kurtosis"]["z"]
                    - row["background"]["proj_B"]["mardia_kurtosis"]["z"]
                    for row in starts
                ],
                dtype=float,
            )
            delta_skew = np.asarray(
                [
                    row["analysis"]["proj_B"]["mardia_skewness"]["z"]
                    - row["background"]["proj_B"]["mardia_skewness"]["z"]
                    for row in starts
                ],
                dtype=float,
            )
            p_background = [row["background"]["proj_B"]["hz"]["p"] for row in starts]
            p_analysis = [row["analysis"]["proj_B"]["hz"]["p"] for row in starts]

            data[(dataset, model)] = {
                "kurt_center": float(np.median(delta_kurt)),
                "kurt_std": float(np.std(delta_kurt, ddof=0)),
                "skew_center": float(np.median(delta_skew)),
                "skew_std": float(np.std(delta_skew, ddof=0)),
                "hz_background": fisher_ratio(p_background),
                "hz_analysis": fisher_ratio(p_analysis),
            }
    return data


def method_color(model):
    return MODEL_COLORS[model]


data = load_data()

group_gap = 1.55
row_positions = {}
y = 0.0
for dataset in DATASETS:
    for model in MODELS:
        row_positions[(dataset, model)] = y
        y += 1.0
    y += group_gap

ymax = y - group_gap
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18.5, 15.0),
    sharey=True,
    gridspec_kw={"width_ratios": [1.18, 1.18, 0.78], "wspace": 0.08},
)

metric_specs = [
    ("kurt_center", "kurt_std", r"Mardia kurtosis  $\Delta z$"),
    ("skew_center", "skew_std", r"Mardia skewness  $\Delta z$"),
]

for ax, (center_key, std_key, xlabel) in zip(axes[:2], metric_specs):
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.85, base=10)
    ax.set_xlim(-3.0, 1500.0)
    ax.axvline(0.0, color=ZERO_COLOR, linewidth=1.8, zorder=1)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.9, alpha=0.72, zorder=0)
    ax.set_xlabel(xlabel, labelpad=12)

    for dataset in DATASETS:
        for model in MODELS:
            item = data[(dataset, model)]
            ypos = row_positions[(dataset, model)]
            color = method_color(model)
            ax.errorbar(
                item[center_key],
                ypos,
                xerr=item[std_key],
                fmt="o",
                markersize=8.6,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.1,
                ecolor=color,
                elinewidth=1.8,
                capsize=3.0,
                alpha=0.98,
                zorder=3,
            )

hz_ax = axes[2]
hz_ax.set_xscale("log")
hz_ax.set_xlim(0.2, 1000.0)
hz_ax.axvspan(0.8, 1.25, color="#D6D6D6", alpha=0.72, zorder=0)
hz_ax.axvline(1.0, color=ZERO_COLOR, linewidth=1.8, zorder=1)
hz_ax.grid(axis="x", which="both", color=GRID_COLOR, linewidth=0.75, alpha=0.62, zorder=0)
hz_ax.set_xlabel(r"Henze--Zirkler, Fisher  $\chi^2/\mathrm{df}$", labelpad=12)

for dataset in DATASETS:
    for model in MODELS:
        item = data[(dataset, model)]
        ypos = row_positions[(dataset, model)]
        background = item["hz_background"]
        analysis = item["hz_analysis"]
        color = method_color(model)

        if not math.isclose(background, analysis, rel_tol=1e-4, abs_tol=1e-4):
            hz_ax.annotate(
                "",
                xy=(analysis, ypos),
                xytext=(background, ypos),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.8,
                    "mutation_scale": 11,
                    "shrinkA": 5,
                    "shrinkB": 5,
                },
                zorder=2,
            )
        hz_ax.plot(
            background,
            ypos,
            marker="o",
            markersize=7.7,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.8,
            linestyle="none",
            zorder=3,
        )
        hz_ax.plot(
            analysis,
            ypos,
            marker="o",
            markersize=8.3,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.0,
            linestyle="none",
            zorder=4,
        )

for ax in axes:
    ax.set_ylim(ymax - 0.45, -0.55)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.25)

method_ticks = []
method_ticklabels = []
for dataset in DATASETS:
    for model in MODELS:
        method_ticks.append(row_positions[(dataset, model)])
        method_ticklabels.append(MODEL_LABELS[model])

axes[0].set_yticks(method_ticks)
axes[0].set_yticklabels(method_ticklabels)
for label, (_, model) in zip(axes[0].get_yticklabels(), row_positions.keys()):
    label.set_color(method_color(model))
    label.set_fontweight("bold")
axes[0].tick_params(axis="y", pad=4)
axes[1].tick_params(axis="y", labelleft=False)
hz_ax.tick_params(axis="y", labelleft=False)

for dataset in DATASETS:
    positions = [row_positions[(dataset, model)] for model in MODELS]
    center = float(np.mean(positions))
    axes[0].text(
        -0.39,
        center,
        DATASET_LABELS[dataset],
        rotation=90,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#333333",
        transform=axes[0].get_yaxis_transform(),
        clip_on=False,
    )

for index in range(len(DATASETS) - 1):
    bottom = row_positions[(DATASETS[index], MODELS[-1])]
    top = row_positions[(DATASETS[index + 1], MODELS[0])]
    separator = (bottom + top) / 2.0
    for ax in axes:
        ax.axhline(separator, color="#E4E4E4", linewidth=0.9, zorder=0)

hz_legend = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
           markeredgecolor="#555555", markeredgewidth=1.7, markersize=7.5, label="Background"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#555555",
           markeredgecolor="#555555", markersize=7.5, label="Analysis"),
]
hz_ax.legend(
    handles=hz_legend, loc="lower left", bbox_to_anchor=(0.0, 1.002),
    ncol=2, frameon=False, fontsize=15, handletextpad=0.35, columnspacing=1.0,
)

fig.subplots_adjust(left=0.225, right=0.985, top=0.975, bottom=0.078)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUTPUT)
