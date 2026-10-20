from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelSummary

from src.train.common import rrmse, BestStateDictSaver, make_trainer


class KAELitModule(L.LightningModule):
    def __init__(self, model, cfg: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.steps = int(cfg["koopman_steps"])
        self.w_recon = float(cfg["w_recon"])
        self.w_pred = float(cfg["w_pred"])
        self.w_latent = float(cfg["w_latent"])
        self.w_orth = float(cfg.get("w_orth", 0.0))
        self.w_spec = float(cfg.get("w_spec", 0.0))
        self.spec_max = float(cfg.get("spec_max", 1.0))
        self.w_straight = float(cfg.get("w_straight", 0.0))
        self.ae_lr = float(cfg["ae_lr"])
        self.koopman_lr = float(cfg["koopman_lr"])
        self.wd = float(cfg["weight_decay"])
        self.max_epochs = int(cfg["max_epochs"])
        self._val = []
        self.save_hyperparameters(ignore=["model"])

    def _losses(self, w):
        b, kp1 = w.shape[0], w.shape[1]
        tau = kp1 - 1
        flat = w.reshape(b * kp1, *w.shape[2:])
        z_all = self.model.encode(flat).reshape(b, kp1, -1)

        rec_all = self.model.decode(z_all.reshape(b * kp1, -1)).reshape(
            b, kp1, *w.shape[2:]
        )
        recon = F.mse_loss(rec_all, w)

        straight = self._straighten(z_all)

        if tau == 0:
            zero = recon.new_zeros(())
            total = self.w_recon * recon + self.w_straight * straight
            return total, recon, zero, zero, straight, rec_all, rec_all[:, :0]

        zr = self.model.rollout(z_all[:, 0], tau).transpose(0, 1)
        pred_dec = self.model.decode(zr[:, 1:].reshape(b * tau, -1)).reshape(
            b, tau, *w.shape[2:]
        )
        pred = F.mse_loss(pred_dec, w[:, 1:])
        latent = F.mse_loss(zr[:, 1:], z_all[:, 1:])

        total = (
            self.w_recon * recon
            + self.w_pred * pred
            + self.w_latent * latent
            + self.w_straight * straight
        )
        return total, recon, pred, latent, straight, rec_all, pred_dec

    def _straighten(self, z_all: torch.Tensor):
        if z_all.shape[1] < 3:
            return z_all.new_zeros(())
        vel = z_all[:, 1:] - z_all[:, :-1]
        cos = F.cosine_similarity(vel[:, :-1], vel[:, 1:], dim=-1, eps=1e-8)
        return (1.0 - cos).mean()

    def _orth_loss(self):
        Kw = self.model.K.weight
        d = Kw.shape[0]
        return (
            (Kw.t() @ Kw - torch.eye(d, device=Kw.device, dtype=Kw.dtype)) ** 2
        ).mean()

    def _spec_target(self) -> float:
        prog = min(1.0, self.current_epoch / max(1, self.max_epochs))
        return self.spec_max - (self.spec_max - 1.0) * prog

    def _specrad_penalty(self, n_iter: int = 8):
        Kw = self.model.K.weight
        d = Kw.shape[0]
        if getattr(self, "_pi_v", None) is None or self._pi_v.shape[0] != d:
            self._pi_v = torch.randn(d, device=Kw.device, dtype=Kw.dtype)
        v = self._pi_v.to(Kw.dtype)
        with torch.no_grad():
            for _ in range(n_iter):
                v = Kw @ v
                v = v / (v.norm() + 1e-8)
            self._pi_v = v.detach()
        Kv = Kw @ v
        rho = Kv.norm() / (v.norm() + 1e-8)
        return torch.relu(rho - self._spec_target()) ** 2

    def training_step(self, batch, _):
        tau = int(torch.randint(0, self.steps + 1, (1,)).item())
        w = batch[:, : tau + 1]
        total, recon, pred, latent, straight, *_ = self._losses(w)
        orth = self._orth_loss() if self.w_orth > 0 else total.new_zeros(())
        spec = self._specrad_penalty() if self.w_spec > 0 else total.new_zeros(())
        total = total + self.w_orth * orth + self.w_spec * spec
        self.log("train/loss", total, prog_bar=True, on_step=True, on_epoch=False)
        self.log_dict(
            {
                "train/recon": recon,
                "train/pred": pred,
                "train/latent": latent,
                "train/straight": straight,
                "train/orth": orth,
                "train/spec": spec,
                "train/tau": float(tau),
            },
            on_step=False,
            on_epoch=True,
        )
        return total

    def validation_step(self, batch, _):
        w = batch
        *_, rec_all, pred_dec = self._losses(w)
        rr_rec = rrmse(rec_all[:, 0], w[:, 0])
        rr_pred = rrmse(
            pred_dec.reshape(-1, *w.shape[2:]), w[:, 1:].reshape(-1, *w.shape[2:])
        )
        self._val.append(torch.stack([rr_rec, rr_pred]).detach())

    def on_validation_epoch_end(self):
        m = torch.stack(self._val).mean(0)
        self.log("val/recon_rrmse", m[0], prog_bar=True)
        self.log("val/pred_rrmse", m[1], prog_bar=True)
        self.log("val/score", m[0] + m[1], prog_bar=True)
        if hasattr(self.model, "spectral_radius"):
            self.log("val/spec_radius", self.model.spectral_radius(), prog_bar=True)
        self._val.clear()

    def configure_optimizers(self):
        groups = [
            {"params": list(self.model.ae_parameters()), "lr": self.ae_lr},
            {"params": list(self.model.koopman_parameters()), "lr": self.koopman_lr},
        ]
        opt = torch.optim.AdamW(groups, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=float(self.cfg["min_lr"])
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


def build_kae_trainer(cfg: Dict[str, Any]):
    from src.config import ckpt_path as _ck

    ckpt_path = _ck(cfg, cfg.get("model_dir", "kae"), cfg.get("ckpt_name", "kae.pt"))

    def _payload(pl):
        m = pl.model
        return (
            {"spectral_radius": m.spectral_radius()}
            if hasattr(m, "spectral_radius")
            else {}
        )

    saver = BestStateDictSaver(ckpt_path, monitor="val/score", payload_fn=_payload)
    trainer = make_trainer(
        cfg, "val/score", "kae", saver, extra_callbacks=[ModelSummary(max_depth=2)]
    )
    return trainer, ckpt_path
