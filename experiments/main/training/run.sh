#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-python}
SEEDS=${SEEDS:-1 2 3}
DENS="5 15 30 50"
FIELD="cylinder shallow_water era5 era5_high"

for s in $SEEDS; do
    export SEED=$s

    for ds in $FIELD; do
        $PY -m src.train.autoencoder --config config/kae.yaml --dataset $ds
        $PY -m src.train.autoencoder --config config/nl_ae.yaml --dataset $ds
        $PY -m src.train.autoencoder --config config/wl_ae.yaml --dataset $ds
    done
    $PY -m src.train.autoencoder --config config/kae.yaml --dataset lorenz96
    $PY -m src.train.autoencoder --config config/nl_ae.yaml --dataset lorenz96
    $PY -m src.train.autoencoder --config config/wl_ae.yaml --dataset lorenz96

    for ds in $FIELD lorenz96; do
        $PY -m src.train.tensorvar_forward --dataset $ds --stage forward
    done
    for d in $DENS; do
        $PY -m src.train.tensorvar_inverse --dataset cylinder_obs$d --ksteps 8
        $PY -m src.train.tensorvar_inverse --dataset shallow_water_obs$d
        $PY -m src.train.tensorvar_inverse --dataset era5_obs$d
        $PY -m src.train.tensorvar_inverse --dataset era5_high_obs$d --epochs 30 --patience 6 --batch 4 --train-windows 4000
    done
    mkdir -p checkpoint/lorenz96/tensorvar/seed$s
    cp ckpt_release/lorenz96/tensorvar/inverse_encoder_pretrained.pt checkpoint/lorenz96/tensorvar/seed$s/
    for d in $DENS; do
        $PY -m src.train.tensorvar_inverse_lorenz96 --dataset lorenz96_obs$d --seed $([ "$s" = 1 ] && echo 0 || echo "$s")
    done

    for ds in $FIELD; do
        $PY -m src.train.ae --dataset $ds
    done
    for d in $DENS; do
        $PY -m src.train.dbf --dataset cylinder --obs-ratio 0.$(printf %02d $d) --out dbf@$d.pt
        $PY -m src.train.dbf --dataset shallow_water --obs-ratio 0.$(printf %02d $d) --out dbf@$d.pt
        $PY -m src.train.dbf --dataset era5 --obs-ratio 0.$(printf %02d $d) --out dbf@$d.pt
        $PY -m src.train.dbf --dataset era5_high --obs-ratio 0.$(printf %02d $d) --batch 8 --out dbf@$d.pt
    done
    mkdir -p checkpoint/lorenz96_latent120/ae/seed$s checkpoint/lorenz96/dbf/seed$s
    cp ckpt_release/ablation/latent_dimension/latent120/ae.pt checkpoint/lorenz96_latent120/ae/seed$s/ae.pt
    $PY -m src.train.dbf --dataset lorenz96_latent120 --seq-config config/kae.yaml --out dbf.pt
    cp checkpoint/lorenz96_latent120/dbf/seed$s/dbf.pt checkpoint/lorenz96/dbf/seed$s/dbf.pt

    for d in $DENS; do
        r=0.$(printf %02d $d)
        for st in 1 2; do
            CKDIR=checkpoint/cylinder/cgkn/seed$s/obs$d OBS_R=$r H=64 W=64 C=2 DZU=32 DATA_DIR=data/cylinder SKIP=5 \
                $PY -m src.model.cgkn.cgkn_grid --stage $st
            CKDIR=checkpoint/shallow_water/cgkn/seed$s/obs$d OBS_R=$r H=96 W=192 C=3 DZU=8 DATA_DIR=data/shallow_water SKIP=2 \
                $PY -m src.model.cgkn.cgkn_grid --stage $st
            CKDIR=checkpoint/era5/cgkn/seed$s/obs$d OBS_R=$r H=64 W=32 DZU=32 DATA_DIR=data/era5 SKIP=4 \
                $PY -m src.model.cgkn.cgkn_era5 --stage $st
            CKDIR=checkpoint/era5_high/cgkn/seed$s/obs$d OBS_R=$r H=240 W=120 DZU=12 DATA_DIR=data/era5_high SKIP=4 \
                $PY -m src.model.cgkn.cgkn_era5 --stage $st
        done
        for ds in cylinder shallow_water era5 era5_high; do
            cp checkpoint/$ds/cgkn/seed$s/obs$d/stage2.pt checkpoint/$ds/cgkn/seed$s/cgkn@${d}.pt
        done
    done
    for pair in 5:2 15:6 30:12 50:20; do
        d=${pair%%:*}
        for st in 1 2; do
            CKDIR=checkpoint/lorenz96/cgkn/seed$s/obs$d OBS_N=${pair##*:} CGKN_SEED=$((s - 1)) \
                $PY -m src.model.cgkn.cgkn_lorenz96 --stage $st
        done
        cp checkpoint/lorenz96/cgkn/seed$s/obs$d/stage2.pt checkpoint/lorenz96/cgkn/seed$s/cgkn@${d}.pt
        cp checkpoint/lorenz96/cgkn/seed$s/obs$d/sigma.pt checkpoint/lorenz96/cgkn/seed$s/sigma@${d}.pt
    done
done
