#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
$PY -m src.eval.latent.cache --dataset lorenz96 --n-runs 128
$PY -m src.eval.latent.cache --dataset cylinder --n-runs 25
$PY -m src.eval.latent.cache --dataset shallow_water
$PY -m src.eval.latent.cache --dataset era5
$PY -m src.eval.latent.cache --dataset era5_high
for ds in lorenz96 cylinder shallow_water era5 era5_high; do
    $PY -m src.eval.latent.cache_cgkn --dataset $ds --density 30
done
$PY -m src.eval.resdmd.subspace_residual --obs30-only --train-fit --fitted-all --out results/resdmd/residual_ratio_train_fit.json
$PY -m src.eval.resdmd.subspace_residual --obs30-only --out results/resdmd/residual_ratio_own_operator.json
