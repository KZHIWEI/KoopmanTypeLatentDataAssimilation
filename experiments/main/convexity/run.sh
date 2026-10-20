#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
for ds in lorenz96 cylinder shallow_water era5 era5_high; do
    $PY -m src.eval.convexity.hessian_per_traj --dataset $ds --models nl_ae,wl_ae,kae,tensorvar,dmd --density 30 --dirs 32 --write
done
for ds in lorenz96 cylinder shallow_water era5; do
    $PY -m src.eval.convexity.eigenvalue_spectra --dataset $ds --models nl_ae,wl_ae,kae --density 30 --radii 0
    HESSIAN_DTYPE=float32 $PY -m src.eval.convexity.eigenvalue_spectra --dataset $ds --models tensorvar,dmd --density 30 --radii 0
done
HESSIAN_DTYPE=float32 $PY -m src.eval.convexity.eigenvalue_spectra --dataset era5_high --models nl_ae,wl_ae,kae,tensorvar,dmd --density 30 --radii 0
