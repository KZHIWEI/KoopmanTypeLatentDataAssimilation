#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
for ds in lorenz96 cylinder shallow_water era5 era5_high; do
    $PY -m src.eval.latent.cache --dataset $ds
    $PY -m src.eval.latent.cache_cgkn --dataset $ds --density 30
done
for ds in lorenz96 cylinder shallow_water era5 era5_high; do
    $PY -m src.eval.latent.tsne --dataset $ds
done
$PY -m src.eval.latent.distance
