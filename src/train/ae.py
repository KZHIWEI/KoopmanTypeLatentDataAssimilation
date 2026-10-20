from __future__ import annotations

import argparse

import torch
import lightning as L

from src.config import load_config, build_ae_config
from src.model.ae import ViTAutoEncoder
from src.data import FieldDataModule
from src.train.common import ReconLitModule, build_trainer, compile_model, rrmse, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lorenz96")
    ap.add_argument("--config", default="config/ae.yaml")
    ap.add_argument("--tok_dim", type=int, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.set_defaults(compile=True)
    args = ap.parse_args()

    cfg = load_config(args.config, args.dataset)
    if args.tok_dim is not None:
        cfg["tok_dim"] = args.tok_dim
    L.seed_everything(int(cfg["seed"]), verbose=False)

    ae_cfg = build_ae_config(cfg)
    model = ViTAutoEncoder(ae_cfg)
    print(
        f"[{args.dataset}] latent={ae_cfg.latent_dim} params={model.num_parameters()/1e6:.2f}M "
        f"compile={args.compile}",
        flush=True,
    )

    lit = ReconLitModule(compile_model(model, args.compile), cfg)
    dm = FieldDataModule(cfg)
    trainer, ckpt_path = build_trainer(cfg)
    trainer.fit(lit, datamodule=dm)
    print(f"[done] {ckpt_path}", flush=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    best = ViTAutoEncoder(ae_cfg)
    best.load_state_dict(ckpt["state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = best.to(dev).eval()
    ds = dm.full_val_dataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], num_workers=8)
    rr, ss, n = 0.0, 0.0, 0
    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"
    ):
        for x in dl:
            x = x.to(dev)
            rec = best.decode(best.encode(x)).float()
            b = x.shape[0]
            rr += float(rrmse(rec, x)) * b
            ss += float(ssim(rec, x)) * b
            n += b
    print(
        f"[FULL-VAL] {args.dataset} ({n} frames)  RRMSE={rr/n:.4f}  SSIM={ss/n:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
