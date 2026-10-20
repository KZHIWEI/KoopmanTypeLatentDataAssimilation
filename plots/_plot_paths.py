import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("KOOPMAN_RESULTS", ROOT / "results")).expanduser().resolve()
FIGURES = Path(os.environ.get("KOOPMAN_PLOT_OUTPUT", RESULTS / "paper_figures")).expanduser().resolve()
FIGURES.mkdir(parents=True, exist_ok=True)
DEFAULT_FONT = "DejaVu Sans"
# One model palette for every figure (Okabe-Ito, colour-blind safe).
MODEL_COLORS = {"MLPAE": "#0072B2", "WAE": "#009E73", "KAE": "#E69F00", "TensorVar": "#CC79A7",
                "DMD": "#D55E00", "DBF": "#56B4E9", "CGKN": "#333333"}

def forecast_steps(n):

    return range(n)
