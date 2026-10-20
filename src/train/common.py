from __future__ import annotations

import math
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, Callback, ModelSummary
from lightning.pytorch.loggers import CSVLogger


def rrmse(rec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    num = torch.linalg.norm((rec - x).flatten(1), dim=1)
    den = torch.linalg.norm(x.flatten(1), dim=1).clamp_min(1e-12)
    return (num / den).mean()


def _gaussian_kernel(size: int, sigma: float, device) -> torch.Tensor:
    c = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-(c**2) / (2 * sigma**2))
    k = g[:, None] * g[None, :]
    return (k / k.sum())[None, None]


def ssim(rec: torch.Tensor, x: torch.Tensor, data_range: float = 8.0) -> torch.Tensor:
    b, c, h, w = x.shape
    K = _gaussian_kernel(11, 1.5, x.device).to(x.dtype).repeat(c, 1, 1, 1)
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx = F.conv2d(rec, K, padding=5, groups=c)
    my = F.conv2d(x, K, padding=5, groups=c)
    sx = F.conv2d(rec * rec, K, padding=5, groups=c) - mx * mx
    sy = F.conv2d(x * x, K, padding=5, groups=c) - my * my
    sxy = F.conv2d(rec * x, K, padding=5, groups=c) - mx * my
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / (
        (mx * mx + my * my + C1) * (sx + sy + C2)
    )
    return m.mean()


def spectral_l1(rec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    rf = torch.fft.rfft2(rec.float(), norm="ortho").abs()
    xf = torch.fft.rfft2(x.float(), norm="ortho").abs()
    return (rf - xf).abs().mean()


class ReconLitModule(L.LightningModule):

    def __init__(self, model: nn.Module, cfg: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = float(cfg["lr"])
        self.min_lr = float(cfg["min_lr"])
        self.warmup = int(cfg["warmup_epochs"])
        self.max_epochs = int(cfg["max_epochs"])
        self.wd = float(cfg["weight_decay"])
        self.spec_w = float(cfg.get("spectral_loss_weight", 0.0))
        self._val_rr: list = []
        self._val_ss: list = []
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        return self.model.decode(self.model.encode(x))

    def training_step(self, batch, _):
        x = batch
        rec = self(x)
        loss = F.mse_loss(rec, x)
        if self.spec_w > 0:
            loss = loss + self.spec_w * spectral_l1(rec, x)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, _):
        x = batch
        rec = self(x)
        self._val_rr.append(rrmse(rec, x).detach())
        self._val_ss.append(ssim(rec, x).detach())

    def on_validation_epoch_end(self):
        rr = torch.stack(self._val_rr).mean()
        ss = torch.stack(self._val_ss).mean()
        self.log("val/rrmse", rr, prog_bar=True)
        self.log("val/ssim", ss, prog_bar=True)
        self._val_rr.clear()
        self._val_ss.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)

        def lr_lambda(epoch):
            if epoch < self.warmup:
                return (epoch + 1) / max(1, self.warmup)
            prog = (epoch - self.warmup) / max(1, self.max_epochs - self.warmup)
            cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
            floor = self.min_lr / self.lr
            return floor + (1 - floor) * cos

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


class BestStateDictSaver(Callback):

    def __init__(
        self, path: str, monitor: str = "val/score", payload_fn=None, state_fn=None
    ):
        super().__init__()
        self.path = path
        self.monitor = monitor
        self.payload_fn = payload_fn
        self.state_fn = state_fn or (lambda pl: pl.model.state_dict())
        self.best = float("inf")
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val = trainer.callback_metrics.get(self.monitor)
        if val is None:
            return
        val = float(val)
        if val < self.best:
            self.best = val
            sd = {
                k.replace("_orig_mod.", ""): v
                for k, v in self.state_fn(pl_module).items()
            }
            payload = {"state_dict": sd, "cfg": pl_module.cfg, self.monitor: val}
            if self.payload_fn is not None:
                payload.update(self.payload_fn(pl_module))
            torch.save(payload, self.path)


def make_trainer(
    cfg: Dict[str, Any],
    monitor: str,
    log_name: str,
    saver: Callback,
    max_epochs: int = None,
    extra_callbacks=None,
    force_fp32: bool = False,
):
    precision = (
        "32-true"
        if force_fp32
        else {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}[
            cfg.get("precision", "bf16")
        ]
    )
    cbs = [
        EarlyStopping(
            monitor=monitor, mode="min", patience=int(cfg.get("patience", 8))
        ),
        saver,
    ]
    cbs += list(extra_callbacks or [])
    return L.Trainer(
        max_epochs=int(max_epochs or cfg["max_epochs"]),
        precision=precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=cbs,
        logger=CSVLogger(
            os.path.join(
                cfg["out_root"],
                cfg["dataset"],
                cfg.get("model_dir", log_name),
                "logs_" + log_name,
            ),
            name="",
        ),
        log_every_n_steps=20,
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
    )


def build_trainer(cfg: Dict[str, Any]):
    from src.config import ckpt_path as _ck

    model = cfg.get("model_dir", "ae")
    ckpt_path = _ck(cfg, model, "ae.pt")
    log_dir = os.path.join(os.path.dirname(ckpt_path), "logs")
    precision = {"bf16": "bf16-mixed", "fp16": "16-mixed", "fp32": "32-true"}[
        cfg.get("precision", "bf16")
    ]

    callbacks = [
        EarlyStopping(monitor="val/rrmse", mode="min", patience=int(cfg["patience"])),
        BestStateDictSaver(ckpt_path, monitor="val/rrmse"),
        ModelSummary(max_depth=2),
    ]
    trainer = L.Trainer(
        max_epochs=int(cfg["max_epochs"]),
        precision=precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        logger=CSVLogger(log_dir, name=""),
        log_every_n_steps=20,
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
    )
    return trainer, ckpt_path


def compile_model(model, enabled: bool = True):
    if not (enabled and hasattr(torch, "compile")):
        return model
    if hasattr(model, "compile_submodules"):
        return model.compile_submodules()
    return torch.compile(model)
