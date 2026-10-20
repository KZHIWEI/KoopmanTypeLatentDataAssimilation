#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
for n in 120 80 40 20; do
    $PY -m src.train.ae --dataset lorenz96_latent$n
    $PY -m src.train.autoencoder --config config/kae.yaml --dataset lorenz96_latent$n
done
$PY -m src.eval.latent_dimension.run
