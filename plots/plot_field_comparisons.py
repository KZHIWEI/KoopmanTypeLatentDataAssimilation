import json

import matplotlib.pyplot as plt
import numpy as np

from _plot_paths import FIGURES, DEFAULT_FONT
from _plot_data import load_fields, psd_data
FIGURE_ROOT = FIGURES
MODELS = ["MLPAE", "WAE", "KAE", "TensorVar", "DMD", "DBF", "CGKN"]
DISPLAY_NAMES = {"MLPAE": "NL-AE", "WAE": "WL-AE", "KAE": "KAE", "TensorVar": "Tensor-Var", "DMD": "DMD", "DBF": "DBF", "CGKN": "CGKN"}


def configure_style():
    plt.rcParams.update({
        "font.family": DEFAULT_FONT, "font.size": 15, "font.weight": "bold",
        "axes.titleweight": "bold", "axes.labelweight": "bold", "axes.linewidth": 1.2,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
    })


def plot_lorenz96():
    arrays, _ = load_fields("lorenz96")
    truth = arrays["truth"].reshape(-1)
    all_values = np.concatenate([value.reshape(-1) for value in arrays.values()])
    margin = 0.08 * (all_values.max() - all_values.min())
    ylim = (all_values.min() - margin, all_values.max() + margin)
    analysis_errors = [
        np.abs(arrays[f"{model}_ana"].reshape(-1) - truth)
        for model in MODELS
    ]
    error_max = max(float(error.max()) for error in analysis_errors)
    error_ylim = (0.0, 1.08 * error_max)

    fig = plt.figure(figsize=(30, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 16, hspace=0.15, wspace=0.28)

    truth_ax = fig.add_subplot(gs[0, 0:2])
    truth_ax.plot(truth, color="#5B5B5B", linewidth=2.7)
    truth_ax.set_title("Truth", fontsize=21, pad=8)
    truth_ax.set_xlim(0, truth.size - 1)
    truth_ax.set_ylim(*ylim)
    truth_ax.set_xlabel("State index", fontsize=14)
    truth_ax.grid(alpha=0.20)

    legend_handles = None
    for column, model in enumerate(MODELS):
        start = 2 + 2 * column
        forecast_ax = fig.add_subplot(gs[0, start:start + 2])
        analysis_ax = fig.add_subplot(gs[1, start:start + 2])
        analysis_error_ax = fig.add_subplot(gs[2, start:start + 2])
        forecast = arrays[f"{model}_bef"].reshape(-1)
        analysis = arrays[f"{model}_ana"].reshape(-1)
        analysis_error = np.abs(analysis - truth)

        truth_forecast, = forecast_ax.plot(truth, color="#B8B8B8", linestyle="--", linewidth=2.1, alpha=0.85)
        forecast_line, = forecast_ax.plot(forecast, color="#0072B2", linewidth=2.5)
        analysis_ax.plot(truth, color="#B8B8B8", linestyle="--", linewidth=2.1, alpha=0.85)
        analysis_line, = analysis_ax.plot(analysis, color="#D55E00", linewidth=2.5)
        analysis_error_ax.plot(analysis_error, color="#D55E00", linewidth=2.3)

        forecast_ax.set_title(DISPLAY_NAMES[model], fontsize=21, pad=8)
        for ax in (forecast_ax, analysis_ax):
            ax.set_xlim(0, truth.size - 1)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.18)
            ax.set_xlabel("State index", fontsize=14)
        analysis_error_ax.set_xlim(0, truth.size - 1)
        analysis_error_ax.set_ylim(*error_ylim)
        analysis_error_ax.grid(alpha=0.18)
        analysis_error_ax.set_xlabel("State index", fontsize=14)

        if column == 0:
            forecast_ax.set_ylabel("Forecast", fontsize=18, labelpad=10)
            analysis_ax.set_ylabel("Analysis", fontsize=18, labelpad=10)
            analysis_error_ax.set_ylabel("Analysis\nabsolute error", fontsize=18, labelpad=10)
        else:
            forecast_ax.tick_params(labelleft=False)
            analysis_ax.tick_params(labelleft=False)
            analysis_error_ax.tick_params(labelleft=False)

        if legend_handles is None:
            legend_handles = (truth_forecast, forecast_line, analysis_line)

    for row in range(1, 3):
        blank_ax = fig.add_subplot(gs[row, 0:2])
        blank_ax.axis("off")

    fig.legend(legend_handles, ["Truth", "Forecast", "Analysis"], loc="outside lower center",
               ncol=3, frameon=False, fontsize=17, handlelength=3.0)
    output = FIGURE_ROOT / "lorenz96_field_comparison.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output





def plot_era5_high():
    raw, meta = load_fields('era5_high')
    arrays = {key: ((value * meta['std'] + meta['mean']) * 1000.0).T for key, value in raw.items()}
    truth = arrays.pop('truth')
    field_vmin, field_vmax = (0.0, 9.0)
    error_vmax = 7.5
    field_label = 'Specific humidity (g kg$^{-1}$)'
    error_label = 'Absolute error (g kg$^{-1}$)'
    fig = plt.figure(figsize=(31, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(4, 17, width_ratios=[1] * 16 + [0.1], hspace=0.04, wspace=0.24)
    truth_ax = fig.add_subplot(gs[0, 0:2])
    physical_image = truth_ax.imshow(truth, cmap='RdBu_r', vmin=field_vmin, vmax=field_vmax, aspect='auto')
    truth_ax.set_xticks([])
    truth_ax.set_yticks([])
    truth_ax.set_title('Truth', fontsize=21, pad=8)
    row_labels = ['Forecast', 'Forecast\nabsolute error', 'Analysis', 'Analysis\nabsolute error']
    row_images = [physical_image, None, None, None]
    for column, model in enumerate(MODELS):
        start = 2 + 2 * column
        forecast = arrays[f'{model}_bef']
        analysis = arrays[f'{model}_ana']
        panels = [forecast, np.abs(forecast - truth), analysis, np.abs(analysis - truth)]
        for row, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row, start:start + 2])
            if row in (0, 2):
                image = ax.imshow(panel, cmap='RdBu_r', vmin=field_vmin, vmax=field_vmax, aspect='auto')
            else:
                image = ax.imshow(panel, cmap='jet', vmin=0.0, vmax=error_vmax, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            row_images[row] = image
            if row == 0:
                ax.set_title(DISPLAY_NAMES[model], fontsize=21, pad=8)
            if column == 0:
                ax.set_ylabel(row_labels[row], fontsize=17, labelpad=13)
    for row in range(1, 4):
        blank_ax = fig.add_subplot(gs[row, 0:2])
        blank_ax.axis('off')
    for row, image in enumerate(row_images):
        colorbar_ax = fig.add_subplot(gs[row, 16])
        colorbar = fig.colorbar(image, cax=colorbar_ax)
        if row in (0, 2):
            colorbar.set_label(field_label, fontsize=14, labelpad=8)
            colorbar.set_ticks(np.arange(0, 10, 1))
        else:
            colorbar.set_label(error_label, fontsize=14, labelpad=8)
            colorbar.set_ticks(np.arange(0, 8, 1))
        colorbar.ax.tick_params(labelsize=11)
    suffix = '_physical'
    output = FIGURE_ROOT / f'era5_high_q600_field_comparison{suffix}.png'
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return output




def plot_era5_high_psd():
    k, spectra = psd_data('era5_high')
    truth_power = spectra['truth']
    curves = [
        ("Truth", "truth", "#CC79A7", "--", 3.2),
        ("Forecast", "forecast", "#111111", ":", 2.8),
        ("obs 5%", "obs5", "#C6DBEF", "-", 2.8),
        ("obs 15%", "obs15", "#9ECAE1", "-", 2.8),
        ("obs 30%", "obs30", "#4292C6", "-", 2.8),
        ("obs 50%", "obs50", "#084594", "-", 2.8),
    ]

    fig, axes = plt.subplots(
        2, 4, figsize=(23, 11.5), sharex=True, sharey=True, constrained_layout=True
    )
    axes = axes.ravel()
    for ax, model in zip(axes[:7], MODELS):
        for label, suffix, color, linestyle, linewidth in curves:
            if suffix == "truth":
                power = truth_power
            else:
                power = spectra[f"{model}:{suffix}"]
            ax.loglog(
                k, power, color=color, linestyle=linestyle, linewidth=linewidth,
                label=label, zorder=10 if suffix == "truth" else 5
            )
        ax.set_title(DISPLAY_NAMES[model], fontsize=23, fontweight="bold", pad=10)
        ax.grid(True, which="both", alpha=0.16)
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
        ax.tick_params(axis="both", which="major", labelsize=15)

    for ax in axes[4:7]:
        ax.set_xlabel("Wavenumber $k$", fontsize=18, fontweight="bold")
    axes[0].set_ylabel("Power spectral density", fontsize=18, fontweight="bold")
    axes[4].set_ylabel("Power spectral density", fontsize=18, fontweight="bold")

    legend_ax = axes[7]
    legend_ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(
        handles, labels, loc="center", ncol=2, frameon=False, fontsize=18,
        handlelength=3.0, columnspacing=1.5, labelspacing=1.2
    )
    output = FIGURE_ROOT / "era5_high_psd_observation_ratios.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


if __name__ == "__main__":
    configure_style()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    print(plot_lorenz96())
    print(plot_era5_high())
    print(plot_era5_high_psd())
