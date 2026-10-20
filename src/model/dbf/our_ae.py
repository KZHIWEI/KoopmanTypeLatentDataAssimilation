from __future__ import annotations

import torch
import torch.nn as nn

from src.model.ae import AEConfig, ViTEncoder, ViTDecoder
from src.model.dbf.dbf import DBF
from src.model.dbf.utils import compute_K


class _FieldEncoder(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.enc = ViTEncoder(cfg)
        self.shape = (cfg.in_channels, cfg.image_size[0], cfg.image_size[1])

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        x = o.reshape(o.shape[0], *self.shape)
        return self.enc(x)


class _FieldDecoder(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.dec = ViTDecoder(cfg)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.dec(z)
        return x.reshape(x.shape[0], -1)


def build_dbf_with_our_ae(
    cfg: AEConfig,
    *,
    mode: str = "external",
    G_val: float = 1.0,
    log_sysnoise: float = -2.0,
    log_obsnoise: float = -2.0,
    save_folder: str = ".",
    device: str = "cuda",
    **dbf_kwargs,
) -> DBF:
    z_dim = cfg.latent_dim
    assert z_dim % 2 == 0
    x_dim = cfg.in_channels * cfg.image_size[0] * cfg.image_size[1]

    h_network = _FieldDecoder(cfg).to(device)
    lambdas = nn.Parameter(0.01 * torch.randn(z_dim, device=device))
    F = compute_K(lambdas.detach(), steps=2, num_real=0, num_complex=z_dim // 2)[1].to(
        device
    )

    dbf = DBF(
        mode=mode,
        F=F,
        h_network=h_network,
        take_blockstep=False,
        take_loss_physical=True,
        log_sysnoise=log_sysnoise,
        log_obsnoise=log_obsnoise,
        G_val=G_val,
        z_dim=z_dim,
        x_dim=x_dim,
        time_series_input=False,
        time_series_length=1,
        aux_alpha=0.0,
        save_folder=save_folder,
        **dbf_kwargs,
    )
    dbf.f_network = _FieldEncoder(cfg).to(device)
    dbf.G_network = _FieldEncoder(cfg).to(device)
    dbf.h_network = h_network
    dbf.lambdas = lambdas
    dbf.train_F = True
    return dbf


def build_dbf_for_dataset(
    dataset: str,
    *,
    ae_config: str = "config/ae.yaml",
    load_ae: bool = True,
    compile: bool = True,
    device: str = "cuda",
    **dbf_kwargs,
) -> DBF:
    from src.config import load_config, build_ae_config

    cfg_d = load_config(ae_config, dataset)
    ae_cfg = build_ae_config(cfg_d)
    dbf = build_dbf_with_our_ae(ae_cfg, device=device, **dbf_kwargs)

    if load_ae:
        from src.config import ckpt_path

        ckpt = ckpt_path(cfg_d, "ae", "ae.pt")
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd["state_dict"] if "state_dict" in sd else sd
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        enc = {
            k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")
        }
        dec = {
            k[len("decoder.") :]: v for k, v in sd.items() if k.startswith("decoder.")
        }
        dbf.f_network.enc.load_state_dict(enc)
        dbf.h_network.dec.load_state_dict(dec)
        print(f"[dbf] {ckpt} z_dim={ae_cfg.latent_dim}")

    if compile:
        try:
            dbf.f_network.enc = torch.compile(dbf.f_network.enc, dynamic=True)
            dbf.G_network.enc = torch.compile(dbf.G_network.enc, dynamic=True)
            dbf.h_network.dec = torch.compile(dbf.h_network.dec, dynamic=True)
            print("[dbf] compiled")
        except Exception as e:
            print(f"[dbf] {e}")
    return dbf
