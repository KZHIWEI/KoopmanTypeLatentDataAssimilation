import argparse

import torch
import lightning as L

from src.config import load_config, build_ae_config, ckpt_path
from src.data import SequenceDataModule
from src.model.tensorvar import TensorVarForward
from src.train.common import rrmse, BestStateDictSaver, make_trainer
from src.assimilation.tensorvar_inpaint import InverseInpaintFullHist


class Lit(L.LightningModule):
    def __init__(self, cfg, fwd, n_sup):
        super().__init__()
        self.cfg = cfg
        self.model = InverseInpaintFullHist(
            build_ae_config(cfg),
            fwd,
            n_sup,
            width=96,
            n_attn=4,
            w_pix=1.0,
            history_len=int(cfg["history_len"]),
            obs_ratio=float(cfg["obs_ratio"]),
            nu=float(cfg.get("nu", 1.0)),
            hist_obs_every=int(cfg["hist_obs_every"]),
            hist_include_current=True,
            mask_aware=True,
        )
        self.max_epochs = int(cfg["inv_max_epochs"])
        self._val = []

    def training_step(self, batch, _):
        l = self.model.compute_loss(batch)
        for k in ("feat", "state", "pix"):
            self.log(f"train/{k}", l[k], prog_bar=True, on_step=True, on_epoch=False)
        return l["total"]

    def validation_step(self, batch, _):
        with torch.no_grad():
            hist = self.model.make_obs_history(batch, seed=0)[:, -1]
            z = self.model.phi_S(self.model.K_O.field(hist))
            self._val.append(rrmse(self.model.phi_inv_S(z), batch[:, -1]))

    def on_validation_epoch_end(self):
        self.log("val/obs_rrmse", torch.stack(self._val).mean(), prog_bar=True)
        self._val.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=3e-4,
            weight_decay=1e-4,
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.max_epochs, eta_min=1e-6
        )
        return {"optimizer": opt, "lr_scheduler": sch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default="config/tensorvar.yaml")
    ap.add_argument("--ksteps", type=int, default=0)
    ap.add_argument("--history-len", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--train-windows", type=int, default=0)
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    cfg = load_config(a.config, a.dataset)
    for arg, key in (
        (a.ksteps, "koopman_steps"),
        (a.history_len, "history_len"),
        (a.epochs, "inv_max_epochs"),
        (a.patience, "patience"),
        (a.batch, "batch_size"),
        (a.train_windows, "num_train_windows"),
    ):
        if arg:
            cfg[key] = arg
    cfg["mask_aware"] = True
    cfg["hist_include_current"] = True
    cfg["hist_obs_every"] = int(load_config("config/da.yaml", a.dataset)["obs_every"])
    n_sup = int(cfg["koopman_steps"]) + 1
    cfg["koopman_steps"] = int(cfg["koopman_steps"]) + int(cfg["history_len"]) - 1
    L.seed_everything(int(cfg.get("seed", 1)), verbose=False)

    ck = torch.load(
        ckpt_path(cfg, "tensorvar", "forward.pt"),
        map_location="cpu",
        weights_only=False,
    )
    fwd = TensorVarForward(build_ae_config(ck["cfg"]))
    fwd.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ck["state_dict"].items()}
    )
    for p in fwd.parameters():
        p.requires_grad_(False)

    name = a.name or f"inverse_encoder@{round(float(cfg['obs_ratio']) * 100)}.pt"
    out = ckpt_path(cfg, "tensorvar", name)
    tr = make_trainer(
        cfg,
        "val/obs_rrmse",
        "tv_" + name.replace(".pt", ""),
        BestStateDictSaver(out, monitor="val/obs_rrmse"),
        max_epochs=int(cfg["inv_max_epochs"]),
        force_fp32=False,
    )
    tr.fit(Lit(cfg, fwd, n_sup), datamodule=SequenceDataModule(cfg))
    print(f"[done] -> {out}", flush=True)


if __name__ == "__main__":
    main()
