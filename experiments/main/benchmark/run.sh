#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
read -r -a seeds <<< "${SEEDS:-1 2 3}"
read -r -a datasets <<< "${DATASETS:-lorenz96 cylinder shallow_water era5 era5_high}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/eval}"
for ds in "${datasets[@]}"; do
    EVAL_SEED="${seeds[0]}" "$PY" -m src.eval.sigma_selection "$ds" dmd
    for s in "${seeds[@]}"; do
        models="nl_ae,wl_ae,kae,tensorvar,dbf,cgkn"
        if [[ "$s" == "${seeds[0]}" ]]; then
            models="$models,dmd"
        fi
        EVAL_SEED="$s" "$PY" -m src.eval.benchmark --dataset "$ds" --models "$models"
        EVAL_SEED="$s" "$PY" -m src.eval.benchmark --dataset "$ds" --models "$models" --speed
    done
    "$PY" -m src.eval.aggregate_benchmark --dataset "$ds" --seeds "${seeds[@]}" --root "$EVAL_OUTPUT_ROOT"
done
