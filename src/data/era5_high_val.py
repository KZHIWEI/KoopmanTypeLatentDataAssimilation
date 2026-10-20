from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import xarray as xr

ZARR = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
SPEC = [
    ("geopotential", 7),
    ("u_component_of_wind", 10),
    ("v_component_of_wind", 10),
    ("temperature", 10),
    ("specific_humidity", 8),
]
PROC = "data/era5_high"
OUT = os.path.join(PROC, "val.h5")
VAL = ("2016-01-01T00:00:00", "2016-12-31T18:00:00")
PROBE_TIME = "2015-12-31T18:00:00"


def _stack(ds, sel):
    return np.stack([ds[v][:, k].sel(time=sel).values for v, k in SPEC], axis=-1)[
        :, :, :120
    ]


def verify(ds) -> float:
    got = _stack(ds, slice(PROBE_TIME, PROBE_TIME))[0]
    with h5py.File(os.path.join(PROC, "train.h5")) as f:
        disk = np.asarray(f["data"][-1])
    if got.shape != disk.shape:
        raise SystemExit(f"{got.shape} {disk.shape}")
    return float(np.abs(got - disk).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    ds = xr.open_zarr(ZARR)

    d = verify(ds)
    print(f"  probe {PROBE_TIME} {d:.3e}", flush=True)
    if d != 0.0:
        raise SystemExit(d)
    if a.verify:
        return
    if os.path.exists(a.out):
        print(f"  {a.out} exists")
        return

    print(f"  fetching {VAL[0]} .. {VAL[1]}", flush=True)
    val = _stack(ds, slice(*VAL))
    print(f"  got {val.shape}", flush=True)
    tmp = a.out + ".part"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("data", data=val)
    os.replace(tmp, a.out)
    print(f"  wrote {a.out}  {val.shape}  {os.path.getsize(a.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
