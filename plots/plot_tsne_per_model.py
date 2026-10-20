import json

import matplotlib.pyplot as plt
import numpy as np


from _plot_paths import RESULTS, FIGURES, DEFAULT_FONT
from _plot_data import DATASETS, DISPLAY, read_json
OUTPUT = FIGURES / "tsne_per_model.png"

MODEL_ORDER = ["MLPAE", "WAE", "KAE", "TensorVar", "DMD", "DBF", "CGKN"]
MODEL_LABELS = {
    "MLPAE": "NL-AE",
    "WAE": "WL-AE",
    "KAE": "KAE",
    "TensorVar": "Tensor-Var",
    "DMD": "DMD",
    "DBF": "DBF",
    "CGKN": "CGKN",
}
DATASET_ORDER = ["lorenz96", "cylinder", "shallow_water", "era5_24h", "era5_high"]
DATASET_LABELS = {
    "lorenz96": "Lorenz-96",
    "cylinder": "Cylinder",
    "shallow_water": "Shallow water",
    "era5_24h": "ERA5–Coarse",
    "era5_high": "ERA5–High",
}
TRAJECTORY_STYLES = {
    "truth": {
        "label": "Reference",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "o",
        "linewidth": 2.8,
        "alpha": 0.95,
        "markersize": 4.2,
        "zorder": 3,
    },
    "before_DA": {
        "label": "Forecast",
        "color": "#999999",
        "linestyle": ":",
        "marker": "^",
        "linewidth": 1.6,
        "alpha": 0.62,
        "markersize": 3.2,
        "zorder": 1,
    },
    "after_DA": {
        "label": "Analysis",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "s",
        "linewidth": 2.8,
        "alpha": 0.95,
        "markersize": 4.2,
        "zorder": 4,
    },
}

plt.rcParams.update(
    {
        "font.family": DEFAULT_FONT,
        "font.size": 22,
        "axes.titlesize": 29,
        "axes.titleweight": "bold",
        "legend.fontsize": 28,
        "axes.linewidth": 1.0,
        "figure.dpi": 220,
        "savefig.dpi": 220,
    }
)


def plot_trajectory(ax, xy, style):
    if xy is None:
        return
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[0] == 0:
        return

    marker_every = max(1, points.shape[0] // 7)
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        marker=style["marker"],
        markevery=marker_every,
        markersize=style["markersize"],
        markeredgewidth=0,
        label=style["label"],
        zorder=style["zorder"],
    )
    ax.scatter(
        points[-1, 0],
        points[-1, 1],
        s=38,
        color=style["color"],
        marker=style["marker"],
        edgecolors="white",
        linewidths=0.7,
        alpha=style["alpha"],
        zorder=style["zorder"] + 1,
    )


payload = {ds: read_json(RESULTS / 'latent' / f'tsne_{DATASETS[ds]}.json') for ds in DATASET_ORDER}

fig, axes = plt.subplots(5, 7, figsize=(25, 16), constrained_layout=False)

for row, dataset in enumerate(DATASET_ORDER):
    for col, model in enumerate(MODEL_ORDER):
        ax = axes[row, col]
        legs = payload[dataset]['models'][DISPLAY[model]]
        trajectories = {'truth': legs['reference'], 'before_DA': legs.get('forecast'), 'after_DA': legs['analysis']}

        for name in ["before_DA", "truth", "after_DA"]:
            plot_trajectory(ax, trajectories.get(name), TRAJECTORY_STYLES[name])

        if row == 0:
            ax.set_title(MODEL_LABELS[model], pad=9)
        if col == 0:
            ax.set_ylabel(
                DATASET_LABELS[dataset],
                fontsize=24,
                fontweight="bold",
                labelpad=9,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.10)
        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_color("#B8B8B8")
            spine.set_linewidth(0.9)

handles = []
legend_labels = []
for name in ["truth", "after_DA", "before_DA"]:
    style = TRAJECTORY_STYLES[name]
    handle, = axes[0, 0].plot(
        [],
        [],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        marker=style["marker"],
        markersize=5,
    )
    handles.append(handle)
    legend_labels.append(style["label"])

fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    ncol=3,
    prop={"weight": "bold", "size": 28},
    frameon=False,
    bbox_to_anchor=(0.5, 0.997),
    handlelength=2.5,
    columnspacing=1.7,
)
fig.subplots_adjust(left=0.075, right=0.995, bottom=0.025, top=0.925, wspace=0.12, hspace=0.14)
fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(OUTPUT)
