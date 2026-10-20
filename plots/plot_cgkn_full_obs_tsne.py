import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


from _plot_paths import RESULTS, FIGURES
SOURCE = RESULTS / "cgkn_full_obs/tsne.json"
OUTPUT = FIGURES / "cgkn_full_obs_tsne.png"
STYLES = {
    "truth": dict(label="Reference", color="#0072B2", linestyle="--", marker="o"),
    "before_DA": dict(label="Forecast", color="#999999", linestyle=":", marker="^"),
    "after_DA": dict(label="Analysis", color="#D55E00", linestyle="-", marker="s"),
}


def main():
    source_bytes = SOURCE.read_bytes()
    payload = json.loads(source_bytes)
    runs = sorted(payload['windows'].items(), key=lambda pair: int(pair[0]))
    checked = []
    for start, legs in runs:
        arrays = {key: np.asarray(legs[source], dtype=float) for key, source in [('truth', 'reference'), ('before_DA', 'forecast'), ('after_DA', 'analysis')]}
        if not all(x.ndim == 2 and x.shape[1] == 2 and len(x) > 0 and np.isfinite(x).all() for x in arrays.values()):
            raise ValueError(f'Invalid t-SNE coordinates at {start}')
        checked.append((start, arrays))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 14,
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.9,
        "savefig.facecolor": "white",
    })
    ncols = min(3, len(runs))
    nrows = (len(runs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.8, 4.0 * nrows + 1.2), squeeze=False)
    fig.subplots_adjust(left=0.025, right=0.975,
                        top=0.84 if nrows > 1 else 0.74,
                        bottom=0.035,
                        wspace=0.08, hspace=0.16)
    fig.suptitle("CGKN with full observations on ERA5–Coarse", fontsize=26, fontweight="bold", y=0.980)
    handles = [Line2D([], [], **style, linewidth=2.4, markersize=6) for style in STYLES.values()]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.955 if nrows > 1 else 0.90),
               ncol=3, frameon=False, handlelength=2.8, columnspacing=2.0,
               prop={"size": 21, "weight": "bold"})

    for idx, (ax, (run, arrays)) in enumerate(zip(axes.flat, checked)):
        ax.set_title(f"Window {idx + 1}", pad=12)
        for leg in ("before_DA", "truth", "after_DA"):
            xy = arrays[leg]
            style = STYLES[leg]


            line, = ax.plot(xy[:, 0], xy[:, 1], **style,
                            linewidth=1.6 if leg == "before_DA" else 2.3,
                            alpha=0.80 if leg == "before_DA" else 0.95,
                            markersize=4.5, markevery=max(1, len(xy) // 8),
                            markeredgewidth=0, zorder=1 if leg == "before_DA" else 3)
            ax.scatter(*xy[-1], s=57, marker=style["marker"], color=style["color"],
                       edgecolor="white", linewidth=0.7, zorder=5)
            if not (np.array_equal(line.get_xdata(), xy[:, 0])
                    and np.array_equal(line.get_ydata(), xy[:, 1])):
                raise AssertionError("Plotted coordinates changed")
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#FAFAFA")
        for spine in ax.spines.values():
            spine.set_color("#B8B8B8")
    for ax in list(axes.flat)[len(runs):]:
        ax.set_visible(False)

    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
