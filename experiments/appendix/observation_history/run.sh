#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
DENS="5 15 30 50"
$PY -m src.eval.sigma_selection lorenz96,cylinder,shallow_water dmd
$PY -m src.eval.sigma_selection era5 dmd 1,5,15,30,50
for d in $DENS; do
    for m in 1 5 10 20; do
        SEED=2 $PY -m src.train.tensorvar_inverse --dataset lorenz96_obs$d --history-len $m --patience 60 --name inverse_encoder@${d}_m$m.pt
    done
    for m in 2 6 10; do
        SEED=2 $PY -m src.train.tensorvar_inverse --dataset cylinder_obs$d --ksteps 8 --history-len $m --name inverse_encoder@${d}_m$m.pt
        SEED=2 $PY -m src.train.tensorvar_inverse --dataset shallow_water_obs$d --history-len $m --name inverse_encoder@${d}_m$m.pt
    done
done
for d in 1 $DENS; do
    for m in 1 5 10; do
        SEED=2 $PY -m src.train.tensorvar_inverse --dataset era5_obs$d --history-len $m --name inverse_encoder@${d}_m$m.pt
    done
done
for d in $DENS; do
    for m in 1 5 10 20; do
        $PY -m src.ablation.observation_history.hist_ablation --family dmd --dataset lorenz96 --density $d --history $m
        $PY -m src.ablation.observation_history.hist_ablation --family tensorvar --dataset lorenz96 --density $d --history $m
    done
    for ds in cylinder shallow_water; do
        for m in 1 5 10; do
            $PY -m src.ablation.observation_history.hist_ablation --family dmd --dataset $ds --density $d --history $m
        done
        for m in 2 6 10; do
            $PY -m src.ablation.observation_history.hist_ablation --family tensorvar --dataset $ds --density $d --history $m
        done
    done
done
for d in 1 $DENS; do
    for m in 1 5 10; do
        $PY -m src.ablation.observation_history.hist_ablation --family dmd --dataset era5 --density $d --history $m
        $PY -m src.ablation.observation_history.hist_ablation --family tensorvar --dataset era5 --density $d --history $m
    done
done
