from __future__ import annotations
import argparse
import contextlib
import io
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, Callback, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from src.config import load_config
from src.data import SequenceDataModule
from src.model.dbf.our_ae import build_dbf_for_dataset


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


from src.train.common import rrmse as _field_rrmse


def _rrmse(pred, gt):
    x = pred.shape[-1]
    return _field_rrmse(pred.reshape(-1, x), gt.reshape(-1, x))


class DBFLit(L.LightningModule):

    def __init__(
        self, dataset, n_step, m_step, lr, epochs, obs_ratio=0.15, mask_fixed=False
    ):
        super().__init__()
        dbf = build_dbf_for_dataset(dataset, load_ae=True, compile=True)
        self.dbf = dbf
        self.f_network = dbf.f_network
        self.G_network = dbf.G_network
        self.h_network = dbf.h_network
        self.lambdas = dbf.lambdas
        self.lambdas.requires_grad_(True)
        self.n_step, self.m_step = (n_step, m_step)
        self.lr, self.epochs = (lr, epochs)
        self.obs_ratio = float(obs_ratio)
        self.mask_fixed = bool(mask_fixed)
        self._fixed_mask = None
        self._val = []

    def _flat(self, x):
        return x.reshape(x.shape[0], x.shape[1], -1).float()

    def _masked_full(self, batch, seed=None):
        B, T, C, H, W = batch.shape
        if self.mask_fixed:
            if self._fixed_mask is None or self._fixed_mask.shape[-2:] != (H, W):
                from src.assimilation.observation import build_mask

                self._fixed_mask = build_mask([H, W], self.obs_ratio, "random", 0)
            mask = (
                self._fixed_mask.to(batch.device)
                .view(1, 1, 1, H, W)
                .expand(B, T, 1, H, W)
            )
            return (self._flat(batch * mask), self._flat(batch))
        g = (
            torch.Generator(device=batch.device).manual_seed(int(seed))
            if seed is not None
            else None
        )
        mask = (
            torch.rand(B, T, 1, H, W, device=batch.device, generator=g) < self.obs_ratio
        ).float()
        return (self._flat(batch * mask), self._flat(batch))

    def training_step(self, batch, _):
        obs, full = self._masked_full(batch)
        try:
            with _quiet():
                loss = self.dbf.compute_loss(obs, full, self.n_step, self.m_step, 1, 1)[
                    0
                ]
        except torch._C._LinAlgError:
            return None
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, _):
        obs, full = self._masked_full(batch, seed=0)
        try:
            with torch.no_grad(), _quiet():
                loss = self.dbf.compute_loss(obs, full, self.n_step, self.m_step, 1, 1)[
                    0
                ]
                out = self.dbf.compute_predictions(
                    obs[:, : self.n_step + self.m_step], self.n_step, self.m_step, 1, 1
                )
                h_filt = out[7]
                rr = _rrmse(h_filt, full[:, : h_filt.shape[1]])
        except torch._C._LinAlgError:
            return
        self._val.append(torch.stack([loss.detach(), rr.detach()]))

    def on_validation_epoch_end(self):
        if not self._val:
            self._val.clear()
            return
        m = torch.stack(self._val).mean(0)
        self.log("val/loss", m[0], prog_bar=True)
        self.log("val/recon_rrmse", m[1], prog_bar=True)
        self._val.clear()

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=1e-06
        )
        return {"optimizer": opt, "lr_scheduler": sched}


class DBFSaver(Callback):

    def __init__(self, path, monitor="val/recon_rrmse", best=float("inf")):
        super().__init__()
        self.path, self.monitor, self.best = (path, monitor, float(best))
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def on_validation_end(self, trainer, pl):
        if trainer.sanity_checking:
            return
        v = trainer.callback_metrics.get(self.monitor)
        if v is None or float(v) >= self.best:
            return
        self.best = float(v)
        strip = lambda d: {k.replace("_orig_mod.", ""): x for (k, x) in d.items()}
        mf = bool(getattr(pl, "mask_fixed", False))
        torch.save(
            {
                "mask_fixed": mf,
                "mask_kind": "fixed" if mf else "random",
                "obs_ratio": float(getattr(pl, "obs_ratio", float("nan"))),
                "f_network": strip(pl.f_network.state_dict()),
                "G_network": strip(pl.G_network.state_dict()),
                "h_network": strip(pl.h_network.state_dict()),
                "lambdas": pl.lambdas.detach().cpu(),
                self.monitor: self.best,
                "z_dim": pl.dbf.z_dim,
            },
            self.path,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cylinder")
    ap.add_argument("--seq-config", default="config/kae.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument(
        "--obs-ratio",
        type=float,
        default=0.15,
    )
    ap.add_argument(
        "--mask-fixed",
        action="store_true",
    )
    ap.add_argument(
        "--out",
        default="dbf.pt",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
    )
    args = ap.parse_args()
    cfg = load_config(args.seq_config, args.dataset)
    L.seed_everything(int(cfg.get("seed", 1)), verbose=False)
    if args.batch > 0:
        cfg["batch_size"] = args.batch
    dm = SequenceDataModule(cfg)
    K = int(cfg["koopman_steps"])
    lit = DBFLit(
        args.dataset,
        n_step=K + 1,
        m_step=0,
        lr=args.lr,
        epochs=args.epochs,
        obs_ratio=args.obs_ratio,
        mask_fixed=args.mask_fixed,
    )
    from src.config import ckpt_path as _ck

    ckpt = _ck(dict(cfg, dataset=args.dataset), "dbf", args.out)
    best0 = float("inf")
    if args.resume and os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)

        def _fit_keys(mod, sd):
            want = set(mod.state_dict().keys())
            if set(sd.keys()) <= want:
                return sd
            bare = {k.replace("_orig_mod.", ""): v for (k, v) in sd.items()}
            for pre in ("", "_orig_mod."):
                cand = {}
                for k, v in bare.items():
                    head, _, rest = k.partition(".")
                    cand[f"{head}.{pre}{rest}" if rest else k] = v
                if set(cand.keys()) <= want:
                    return cand
            return bare

        for name in ("f_network", "G_network", "h_network"):
            mod = getattr(lit, name)
            mod.load_state_dict(_fit_keys(mod, ck[name]))
        if "lambdas" in ck and hasattr(lit, "lambdas"):
            with torch.no_grad():
                lit.lambdas.copy_(ck["lambdas"].to(lit.lambdas.device))
        best0 = float(ck.get("val/recon_rrmse", float("inf")))
        prev_mf = ck.get("mask_fixed")
        if prev_mf is not None and bool(prev_mf) != bool(args.mask_fixed):
            raise SystemExit(f"{ckpt} mask_fixed={prev_mf} {args.mask_fixed}")
        print(
            f"[resume] {ckpt}  val/recon_rrmse={best0:.5f}  mask_fixed={prev_mf}",
            flush=True,
        )
    elif args.resume:
        print(f"[resume] {ckpt} missing", flush=True)
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="32-true",
        gradient_clip_val=1.0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        callbacks=[
            EarlyStopping("val/recon_rrmse", mode="min", patience=args.patience),
            DBFSaver(ckpt, best=best0),
            TQDMProgressBar(refresh_rate=5),
        ],
        logger=CSVLogger(os.path.join(os.path.dirname(ckpt), "logs_dbf"), name=""),
        log_every_n_steps=10,
    )
    trainer.fit(lit, datamodule=dm)
    print(
        f"[done] {args.dataset} dbf -> {ckpt} best val/recon_rrmse={trainer.callback_metrics.get('val/recon_rrmse')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
