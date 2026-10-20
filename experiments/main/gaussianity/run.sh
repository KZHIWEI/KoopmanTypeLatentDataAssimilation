#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
for ds in lorenz96 cylinder shallow_water era5 era5_high; do
    $PY -m src.eval.gaussianity.run --dataset $ds
done
