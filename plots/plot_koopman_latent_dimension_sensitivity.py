import json

import matplotlib.pyplot as plt
import numpy as np


from _plot_paths import RESULTS, FIGURES, DEFAULT_FONT
INPUT = RESULTS / "latent_dimension" / "forecast_rrmse.json"
OUTPUT_PNG = FIGURES / "koopman_latent_dimension_sensitivity.png"

LATENT_DIMS = [20, 40, 80, 120]
COLORS = {
    20: "#F6D55C",
    40: "#EDAE49",
    80: "#E67E22",
    120: "#A63D00",
}
MARKERS = {20: "o", 40: "s", 80: "D", 120: "^"}


plt.rcParams.update(
    {
        "font.family": DEFAULT_FONT,
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.linewidth": 1.0,
        "figure.dpi": 240,
        "savefig.dpi": 400,
    }
)


def load_curves():
    with INPUT.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    curves = {}
    expected_steps = None
    for latent_dim in LATENT_DIMS:
        arm = data[str(latent_dim)]
        values = np.asarray(arm["per_step_rrmse"], dtype=float)
        if expected_steps is None:
            expected_steps = len(values)
        elif len(values) != expected_steps:
            raise ValueError("All latent-dimension curves must have the same length")
        curves[latent_dim] = values
    return curves


def main():
    curves = load_curves()
    steps = np.arange(1, len(next(iter(curves.values()))) + 1)

    fig, ax = plt.subplots(figsize=(5.6, 3.35))
    for latent_dim in LATENT_DIMS:
        ax.plot(
            steps,
            curves[latent_dim],
            color=COLORS[latent_dim],
            linewidth=2.25,
            marker=MARKERS[latent_dim],
            markevery=4,
            markersize=5.2,
            markeredgewidth=0.7,
            markeredgecolor="white",
            label=f"{latent_dim}",
        )

    ax.set_xlabel("Forecast step")
    ax.set_ylabel("RRMSE")
    ax.set_xlim(1, steps[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.8, alpha=0.75)
    ax.tick_params(axis="both", width=0.9, length=3.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        title="Latent dimension",
        title_fontsize=10.5,
        frameon=False,
        ncol=2,
        loc="lower right",
        columnspacing=1.1,
        handlelength=2.2,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
