from __future__ import annotations

import argparse

import torch
import lightning as L

from src.config import load_config, build_ae_config, ckpt_path
from src.data import SequenceDataModule
from src.model.tensorvar import TensorVarForward, TensorVarInverse
from src.train.common import rrmse, BestStateDictSaver, make_trainer




class InverseLit(L.LightningModule):
    def __init__(self, cfg, forward_model):
        super().__init__()
        self.cfg = cfg
        self.model = TensorVarInverse(
            build_ae_config(cfg),
            forward_model,
            history_len=int(cfg.get("history_len", 10)),
            obs_ratio=float(cfg.get("obs_ratio", 0.15)),
            nu=float(cfg.get("nu", 1.0)),
            hist_obs_every=int(cfg.get("hist_obs_every", 1)),
            hist_include_current=bool(cfg.get("hist_include_current", False)),
            hist_mask_kind=str(cfg.get("hist_mask_kind", "random")),
            mask_aware=bool(cfg.get("mask_aware", False)),
        )
        self.max_epochs = int(cfg["inv_max_epochs"])
        self._val = []

    def training_step(self, batch, _):
        losses = self.model.compute_loss(batch)
        self.log(
            "train/feat", losses["feat"], prog_bar=True, on_step=True, on_epoch=False
        )
        self.log(
            "train/state", losses["state"], prog_bar=True, on_step=True, on_epoch=False
        )
        return losses["total"]

    def validation_step(self, batch, _):
        with torch.no_grad():
            B, T = batch.shape[:2]
            hist = self.model.make_obs_history(batch, seed=0)
            z = self.model.K_O(hist.reshape(B * T, *hist.shape[2:]))
            dec = self.model.phi_inv_S(z)
            x = batch.reshape(B * T, *batch.shape[2:])
            obs_rec = rrmse(dec, x)
        self._val.append(obs_rec)

    def on_validation_epoch_end(self):
        m = torch.stack(self._val).mean()
        self._val.clear()
        self.log("val/obs_rrmse", m, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.cfg.get("obs_lr", self.cfg["koopman_lr"])),
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=float(self.cfg.get("min_lr", 1e-6))
        )
        return {"optimizer": opt, "lr_scheduler": sched}


@torch.no_grad()
def fit_c_full(
    model: TensorVarForward,
    cfg,
    device,
    lam: float = 1e-2,
    max_pairs: int = 20000,
    batch: int = 256,
):
    import numpy as np
    from src.data import _load_trajs, channel_norm

    tr = _load_trajs(cfg["data_path"])
    mean, std = channel_norm(np.asarray(tr[:, :: max(1, tr.shape[1] // 200)]), cfg)
    skip = int(cfg.get("skip", 1))
    ntraj, T = tr.shape[0], tr.shape[1]
    per = max(2, max_pairs // ntraj)
    ts = np.linspace(0, T - skip - 1, per).astype(int)
    D = model.hidden_dim
    ZtZ = torch.zeros(D, D, dtype=torch.float64, device=device)
    ZtZp = torch.zeros(D, D, dtype=torch.float64, device=device)
    n = 0
    for ti in range(ntraj):
        a = (
            np.asarray(tr[ti, ts]).astype(np.float32) - mean.reshape(1, -1, 1, 1)
        ) / std.reshape(1, -1, 1, 1)
        b = (
            np.asarray(tr[ti, ts + skip]).astype(np.float32) - mean.reshape(1, -1, 1, 1)
        ) / std.reshape(1, -1, 1, 1)
        for i in range(0, len(ts), batch):
            za = model.encode(torch.tensor(a[i : i + batch], device=device)).double()
            zb = model.encode(torch.tensor(b[i : i + batch], device=device)).double()
            ZtZ += za.t() @ za
            ZtZp += za.t() @ zb
            n += za.shape[0]
    ridge = lam * (ZtZ.diagonal().mean())
    C = torch.linalg.solve(
        ZtZ + ridge * torch.eye(D, dtype=torch.float64, device=device), ZtZp
    )
    model.C_forward.copy_(C.float())
    rho = model.spectral_radius()
    print(f"[tv] C n={n} D={D} rho={rho:.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="config/tensorvar.yaml")
    ap.add_argument("--stage", choices=["forward", "inverse", "both"], default="both")
    ap.add_argument("--batch", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config, args.dataset)
    cfg["mask_aware"] = True
    if args.batch > 0:
        cfg["batch_size"] = args.batch
    L.seed_everything(int(cfg.get("seed", 1)), verbose=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fwd_ckpt = ckpt_path(cfg, "tensorvar", "forward.pt")
    model = TensorVarForward(build_ae_config(cfg))
    model.load_pretrained_ae(cfg["pretrained_ae"])
    if args.stage in ("forward", "both"):
        model = model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        fit_c_full(model, cfg, dev)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "cfg": cfg,
                "spectral_radius": model.spectral_radius(),
            },
            fwd_ckpt,
        )
        print(f"[tv] forward -> {fwd_ckpt}", flush=True)
    if args.stage in ("inverse", "both"):
        ck = torch.load(fwd_ckpt, map_location="cpu")
        model.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in ck["state_dict"].items()}
        )
        out = ckpt_path(cfg, "tensorvar", "inverse_encoder_pretrained.pt")
        trainer = make_trainer(
            cfg,
            "val/obs_rrmse",
            "tv_inverse",
            BestStateDictSaver(out, monitor="val/obs_rrmse"),
            max_epochs=int(cfg["inv_max_epochs"]),
            force_fp32=False,
        )
        trainer.fit(InverseLit(cfg, model), datamodule=SequenceDataModule(cfg))
        print(f"[tv] inverse -> {out}", flush=True)


if __name__ == "__main__":
    main()
