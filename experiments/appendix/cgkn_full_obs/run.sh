#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
for st in 1 2; do
    FULL_OBS=1 CKDIR=checkpoint/ablation/cgkn_full_obs/era5 H=64 W=32 DZU=32 DATA_DIR=data/era5 SKIP=4 \
        $PY -m src.model.cgkn.cgkn_era5 --stage $st
done
cp checkpoint/ablation/cgkn_full_obs/era5/stage2.pt checkpoint/ablation/cgkn_full_obs/era5/cgkn.pt
CKDIR=${CKDIR:-ckpt_release/ablation/cgkn_full_obs/era5} $PY -m src.eval.cgkn_full_obs.tsne --windows 6
