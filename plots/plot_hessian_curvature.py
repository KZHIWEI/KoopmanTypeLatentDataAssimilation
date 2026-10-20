
import matplotlib.pyplot as plt
import numpy as np


from _plot_paths import FIGURES, DEFAULT_FONT
PNG_PATH = FIGURES / "hessian_curvature.png"

INK = "#243447"
GREY = "#77818C"
BLUE = "#2878A0"
RED = "#CB4B3E"
PURPLE = "#7459A6"


def draw_panel(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    color: str,
    panel: str,
    title: str,
    subtitle: str,
    z_limits: tuple[float, float],
) -> None:

    ax.plot_surface(
        x,
        y,
        values,
        color=color,
        alpha=0.30,
        linewidth=0,
        antialiased=True,
        shade=True,
        rcount=80,
        ccount=80,
    )
    ax.plot_wireframe(
        x,
        y,
        values,
        rstride=18,
        cstride=18,
        color=color,
        linewidth=0.55,
        alpha=0.48,
    )
    ax.scatter([0], [0], [0], s=22, color=INK, depthshade=False)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_zlim(*z_limits)
    ax.view_init(elev=29, azim=-52)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel(r"$\mathbf{v}_1$", labelpad=-8, color=GREY)
    ax.set_ylabel(r"$\mathbf{v}_2$", labelpad=-8, color=GREY)
    ax.set_zlabel(r"$J$", labelpad=-5, color=GREY)
    ax.set_box_aspect((1, 1, 0.78))
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax.set_title(
        f"{panel} {title}\n{subtitle}", loc="left", pad=2,
        fontsize=13.5, fontweight="bold", color=INK,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": DEFAULT_FONT,
            "mathtext.fontset": "dejavusans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.labelsize": 12.5,
            "figure.dpi": 150,
            "savefig.dpi": 400,
        }
    )

    grid = np.linspace(-2.0, 2.0, 161)
    x, y = np.meshgrid(grid, grid)

    positive = 0.5 * (0.9 * x**2 + 1.25 * y**2)
    negative = 0.5 * (0.9 * x**2 - 0.9 * y**2)
    near_zero = 0.5 * (0.035 * x**2 + 1.25 * y**2)

    fig = plt.figure(figsize=(15.2, 4.8))
    axes = [fig.add_subplot(1, 3, index, projection="3d") for index in range(1, 4)]
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.005, top=0.86, wspace=0.01)

    draw_panel(
        axes[0], x, y, positive, BLUE, "(a)",
        "Positive curvature", r"$\lambda_1>0,\;\lambda_2>0$", (0.0, 4.4),
    )
    draw_panel(
        axes[1], x, y, negative, RED, "(b)",
        "Negative curvature", r"$\lambda_1>0,\;\lambda_2<0$", (-2.0, 2.0),
    )
    draw_panel(
        axes[2], x, y, near_zero, PURPLE, "(c)",
        "Nearly flat direction", r"$\lambda_1\approx0,\;\lambda_2>0$", (0.0, 2.7),
    )

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {PNG_PATH}")


if __name__ == "__main__":
    main()
