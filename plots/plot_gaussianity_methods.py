
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


from _plot_paths import FIGURES, DEFAULT_FONT
PNG_PATH = FIGURES / "gaussianity_methods.png"

INK = "#252525"
BLUE = "#2A7694"
ORANGE = "#D88900"
RED = "#D64535"
PURPLE = "#8E63A8"
GREY = "#666666"


def style_axis(ax: plt.Axes) -> None:

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GREY)
    ax.spines["bottom"].set_color(GREY)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(colors=GREY, labelsize=9, width=1.0)
    ax.grid(False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": DEFAULT_FONT,
            "mathtext.fontset": "dejavusans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 400,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.0))
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.27,
        top=0.90,
        wspace=0.30,
    )


    ax = axes[0]
    hz_x = np.linspace(-4.0, 4.0, 700)
    gaussian_pdf = stats.norm.pdf(hz_x)
    peak_offset = 1.35
    peak_scale = 0.55
    bimodal_pdf = 0.5 * stats.norm.pdf(
        hz_x, loc=-peak_offset, scale=peak_scale
    ) + 0.5 * stats.norm.pdf(
        hz_x, loc=peak_offset, scale=peak_scale
    )

    ax.plot(
        hz_x,
        gaussian_pdf,
        color=BLUE,
        linewidth=2.5,
        label="Gaussian reference",
    )
    ax.fill_between(hz_x, gaussian_pdf, color=BLUE, alpha=0.10)
    ax.plot(
        hz_x,
        bimodal_pdf,
        color=RED,
        linewidth=2.5,
        label="Non-Gaussian example",
    )
    ax.fill_between(hz_x, bimodal_pdf, color=RED, alpha=0.10)
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Density")
    ax.set_title("(a) HENZE--ZIRKLER TEST", loc="left", fontweight="bold")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=1,
    )
    style_axis(ax)


    ax = axes[1]
    x = np.linspace(-4.2, 4.2, 700)
    normal_pdf = stats.norm.pdf(x)
    shape = 6.0
    delta = shape / np.sqrt(1.0 + shape**2)
    skew_mean = delta * np.sqrt(2.0 / np.pi)
    skew_sd = np.sqrt(1.0 - 2.0 * delta**2 / np.pi)
    skew_pdf = skew_sd * stats.skewnorm.pdf(
        skew_sd * x + skew_mean, shape
    )

    ax.plot(x, normal_pdf, color=BLUE, linewidth=2.3, label="Gaussian reference")
    ax.fill_between(x, normal_pdf, color=BLUE, alpha=0.10)
    ax.plot(
        x,
        skew_pdf,
        color=ORANGE,
        linewidth=2.3,
        label="Asymmetric: skewness > 0",
    )
    ax.fill_between(x, skew_pdf, color=ORANGE, alpha=0.10)
    ax.axvline(0.0, color=GREY, linewidth=0.9, linestyle=":")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Density")
    ax.set_title("(b) MARDIA SKEWNESS", loc="left", fontweight="bold")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=1,
    )
    style_axis(ax)


    ax = axes[2]
    core_weight = 0.90
    core_scale = 0.45
    tail_scale = 2.50
    mixture_variance = (
        core_weight * core_scale**2
        + (1.0 - core_weight) * tail_scale**2
    )
    mixture_sd = np.sqrt(mixture_variance)
    heavy_pdf = core_weight * stats.norm.pdf(x, scale=core_scale / mixture_sd) + (
        1.0 - core_weight
    ) * stats.norm.pdf(x, scale=tail_scale / mixture_sd)
    heavy_excess = 3.0 * (core_weight * core_scale**4 + (1.0 - core_weight) * tail_scale**4) / mixture_variance**2 - 3.0
    light_shape = 4.0
    light_scale = 1.0 / np.sqrt(stats.gennorm.var(light_shape))
    light_pdf = stats.gennorm.pdf(x, light_shape, scale=light_scale)
    light_excess = float(stats.gennorm.stats(light_shape, moments="k"))

    ax.fill_between(x, normal_pdf, color=BLUE, alpha=0.08, zorder=1)
    ax.fill_between(x, heavy_pdf, color=ORANGE, alpha=0.08, zorder=1)
    ax.fill_between(x, light_pdf, color=PURPLE, alpha=0.08, zorder=1)
    ax.plot(
        x,
        normal_pdf,
        color=BLUE,
        linewidth=2.8,
        zorder=4,
        label="Gaussian: excess = 0",
    )
    ax.plot(
        x,
        heavy_pdf,
        color=ORANGE,
        linewidth=2.3,
        zorder=3,
        label="Heavy-tailed: excess > 0",
    )
    ax.plot(
        x,
        light_pdf,
        color=PURPLE,
        linewidth=2.1,
        linestyle="--",
        zorder=2,
        label="Light-tailed: excess < 0",
    )
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("Density")
    ax.set_title("(c) MARDIA KURTOSIS", loc="left", fontweight="bold")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=1,
    )
    style_axis(ax)

    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved {PNG_PATH}")


if __name__ == "__main__":
    main()
