from __future__ import annotations

import argparse
import os

import lightning as L

from src.config import load_config, build_ae_config
from src.model.kae import KoopmanAutoEncoder
from src.model.latent_dynamics import MLPDynamicsAutoEncoder
from src.data import SequenceDataModule
from src.train.common import compile_model
from src.train.autoencoder_module import KAELitModule, build_kae_trainer


def _guard_config(path, dataset):
    import glob
    import yaml as _yaml

    try:
        raw = _yaml.safe_load(open(path)) or {}
    except OSError:
        return
    if dataset in raw:
        return
    here = os.path.dirname(path) or "."
    elsewhere = [
        os.path.basename(f)
        for f in sorted(glob.glob(os.path.join(here, "*.yaml")))
        if f != path
        and dataset
        in (
            (_yaml.safe_load(open(f)) or {})
            if os.path.basename(f) != "dataset.yaml"
            else {}
        )
    ]
    if elsewhere:
        raise SystemExit(f"{dataset} {path} {elsewhere}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lorenz96")
    ap.add_argument("--config", default="config/kae.yaml")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--ae-lr", type=float, default=0.0)
    ap.add_argument(
        "--koopman-lr",
        type=float,
        default=0.0,
    )
    ap.add_argument(
        "--w-latent",
        type=float,
        default=-1.0,
    )
    ap.set_defaults(compile=True)
    args = ap.parse_args()

    _guard_config(args.config, args.dataset)
    cfg = load_config(args.config, args.dataset)
    if args.batch > 0:
        cfg["batch_size"] = args.batch
    if args.epochs > 0:
        cfg["max_epochs"] = args.epochs
    if args.ae_lr > 0:
        cfg["ae_lr"] = args.ae_lr
    if args.koopman_lr > 0:
        cfg["koopman_lr"] = args.koopman_lr
    if args.w_latent >= 0:
        cfg["w_latent"] = args.w_latent
    L.seed_everything(int(cfg["seed"]), verbose=False)

    ae_cfg = build_ae_config(cfg)
    if cfg.get("dynamics", "linear") == "mlp":
        model = MLPDynamicsAutoEncoder(
            ae_cfg,
            width=int(cfg.get("mlp_width", 512)),
            depth=int(cfg.get("mlp_depth", 3)),
            hidden_mult=int(cfg.get("mlp_hidden_mult", 2)),
            residual=bool(cfg.get("mlp_residual", True)),
        )
    else:
        model = KoopmanAutoEncoder(ae_cfg)
    if cfg.get("pretrained_ae"):
        model.load_pretrained_ae(cfg["pretrained_ae"])
        print(f"[kae] {cfg['pretrained_ae']}", flush=True)
    spec = (
        f" spec_radius0={model.spectral_radius():.4f}"
        if hasattr(model, "spectral_radius")
        else ""
    )
    print(
        f"[{args.dataset}] latent={ae_cfg.latent_dim} K={cfg['koopman_steps']} "
        f"params={model.num_parameters()/1e6:.2f}M{spec} "
        f"batch={cfg['batch_size']} "
        f"ae_lr={cfg['ae_lr']} koopman_lr={cfg['koopman_lr']} "
        f"w_latent={cfg.get('w_latent', 0.0)}",
        flush=True,
    )

    lit = KAELitModule(compile_model(model, args.compile), cfg)
    dm = SequenceDataModule(cfg)
    trainer, ckpt_path = build_kae_trainer(cfg)
    trainer.fit(lit, datamodule=dm)
    tail = (
        f"  final spec_radius={model.spectral_radius():.4f}"
        if hasattr(model, "spectral_radius")
        else ""
    )
    print(f"[done] {ckpt_path}{tail}", flush=True)


if __name__ == "__main__":
    main()
