import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

from _plot_paths import FIGURES
from _plot_data import load_fields
MODELS = ["MLPAE", "WAE", "KAE", "TensorVar", "DMD", "DBF", "CGKN"]
NAMES = ["NL-AE", "WL-AE", "KAE", "Tensor-Var", "DMD", "DBF", "CGKN"]
DATASETS = ["cylinder", "shallow_water", "era5_24h"]


def plot_case(dataset, output_dir):
    raw, meta = load_fields(dataset)
    humidity = dataset == "era5_24h"
    variable = "q600" if humidity else "channel 0"
    if humidity:
        arrays = {key: ((value * meta['std'] + meta['mean']) * 1000.0).T for key, value in raw.items()}
    else:
        arrays = raw
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"Non-finite physical fields: {dataset}")
    truth = arrays["truth"]
    errors = {key: np.abs(value.astype(np.float64) - truth.astype(np.float64))
              for key, value in arrays.items() if key != "truth"}
    field_min = min(float(a.min()) for a in arrays.values())
    field_max = max(float(a.max()) for a in arrays.values())
    error_max = max(float(a.max()) for a in errors.values())
    field_norm = Normalize(field_min, field_max)
    error_norm = Normalize(0, error_max if error_max else 1)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "font.weight": "bold", "axes.titleweight": "bold",
        "axes.labelweight": "bold", "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
    })

    panel_w = 2.0
    panel_h = panel_w * truth.shape[0] / truth.shape[1]
    gap = 0.22
    row_gap = 0.24
    left, bottom, top = 0.10, 0.12, 0.46
    model_start = left + panel_w + 0.58
    x_positions = [left] + [model_start + j * (panel_w + gap) for j in range(7)]
    colorbar_x = x_positions[-1] + panel_w + 0.17
    fig_w = colorbar_x + 1.25
    fig_h = bottom + 4 * panel_h + 3 * row_gap + top
    fig = plt.figure(figsize=(fig_w, fig_h))

    def axes_at(x, row, width=panel_w):
        y = bottom + (3 - row) * (panel_h + row_gap)
        return fig.add_axes([x / fig_w, y / fig_h, width / fig_w, panel_h / fig_h])

    truth_ax = axes_at(x_positions[0], 0)
    truth_ax.imshow(truth, cmap="RdBu_r", norm=field_norm,
                    interpolation="nearest", origin="upper", aspect="equal")
    truth_ax.set_title("Truth", fontsize=19, pad=9)
    truth_ax.set_xticks([])
    truth_ax.set_yticks([])

    row_labels = ["Forecast", "Forecast\nabsolute\nerror", "Analysis", "Analysis\nabsolute\nerror"]
    if panel_h >= 1.5:
        row_labels = ["Forecast", "Forecast\nabsolute error", "Analysis", "Analysis\nabsolute error"]
    row_images = [None] * 4
    for column, (model, name) in enumerate(zip(MODELS, NAMES), start=1):
        panels = [arrays[f"{model}_bef"], errors[f"{model}_bef"],
                  arrays[f"{model}_ana"], errors[f"{model}_ana"]]
        for row, panel in enumerate(panels):
            ax = axes_at(x_positions[column], row)
            row_images[row] = ax.imshow(
                panel, cmap="RdBu_r" if row % 2 == 0 else "jet",
                norm=field_norm if row % 2 == 0 else error_norm,
                interpolation="nearest", origin="upper", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=19, pad=9)
            if column == 1:
                ax.set_ylabel(row_labels[row], fontsize=13, labelpad=11)

    for row, image in enumerate(row_images):
        cax = axes_at(colorbar_x, row, width=0.095)
        colorbar = fig.colorbar(image, cax=cax)
        colorbar.locator = MaxNLocator(nbins=3 if panel_h == 1 else 5)
        colorbar.update_ticks()
        colorbar.ax.tick_params(labelsize=11, length=2.5, pad=3)
        colorbar_label = "Standardized\nvalue" if row % 2 == 0 else "Absolute\nerror"
        if humidity:
            colorbar_label = (f"$q_{{{variable[1:]}}}$\n(g kg$^{{-1}}$)" if row % 2 == 0
                              else "Absolute\nerror\n(g kg$^{-1}$)")
        if panel_h >= 1.5:
            colorbar_label = colorbar_label.replace("\n", " ")
        colorbar.set_label(colorbar_label,
                           fontsize=12, labelpad=7)

    output_dir.mkdir(parents=True, exist_ok=True)
    variable_suffix = f"_{variable}_physical" if humidity else ""
    output = output_dir / f"{dataset}{variable_suffix}_field_comparison.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    for dataset in args.datasets:
        print(plot_case(dataset, args.output_dir))


if __name__ == "__main__":
    main()
