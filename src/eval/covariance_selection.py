import json
import math
import os
from pathlib import Path


def selection_path():
    return Path(os.environ.get("SIGMA_SELECTION_PATH", "results/sigma_selection.json"))


def dmd_sigmas(dataset, density):
    path = selection_path()
    key = f"{dataset}:dmd:obs{density}"
    command = f"python -m src.eval.sigma_selection {dataset} dmd {density}"
    if not path.is_file():
        raise FileNotFoundError(f"Missing DMD calibration: {path}. Run: {command}")
    with path.open() as stream:
        records = json.load(stream)
    if key not in records:
        raise KeyError(f"Missing DMD calibration: {key}. Run: {command}")
    record = records[key]
    if record.get("split") != "validation":
        raise ValueError(f"DMD calibration must use validation data. Run: {command}")
    values = []
    for name in ("sigma_r", "sigma_q"):
        value = record.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid {name} for {key}: {value!r}")
        values.append(float(value))
    return *values, f"{path}:{key}"
