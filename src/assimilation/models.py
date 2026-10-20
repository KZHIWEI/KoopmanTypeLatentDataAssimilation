from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.config import build_ae_config
from src.assimilation.variational import LatentDAModel
from src.model.kae import KoopmanAutoEncoder


class KAEModel(LatentDAModel):

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[torch.device] = None,
        compile: bool = True,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ck = torch.load(ckpt_path, map_location="cpu")
        if "cfg" not in ck:
            raise KeyError(ckpt_path)
        self.train_cfg: Dict[str, Any] = ck["cfg"]
        model = KoopmanAutoEncoder(build_ae_config(self.train_cfg))
        sd = ck["state_dict"] if "state_dict" in ck else ck
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(self.device).eval()
        self.spectral_radius = float(
            ck.get("spectral_radius", self.model.spectral_radius())
        )
        if compile:
            self.model.compile_submodules(dynamic=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x.to(self.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.step(z)

    def linear_operator(self) -> Optional[torch.Tensor]:
        return self.model.K.weight.detach().t()


class TensorVarModel(LatentDAModel):

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[torch.device] = None,
        compile: bool = True,
    ):
        from src.model.tensorvar import TensorVarForward

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ck = torch.load(ckpt_path, map_location="cpu")
        if "cfg" not in ck:
            raise KeyError(ckpt_path)
        self.train_cfg: Dict[str, Any] = ck["cfg"]
        model = TensorVarForward(build_ae_config(self.train_cfg))
        sd = ck.get("state_dict", ck)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(self.device).eval()
        self.spectral_radius = float(self.model.spectral_radius())

    def encode(self, x):
        return self.model.encode(x.to(self.device))

    def decode(self, z):
        return self.model.decode(z)

    def step(self, z):
        return self.model.step(z)

    def linear_operator(self) -> Optional[torch.Tensor]:
        return self.model.C_forward.detach()


class MLPDynamicsModel(LatentDAModel):

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[torch.device] = None,
        compile: bool = True,
    ):
        from src.model.latent_dynamics import MLPDynamicsAutoEncoder

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ck = torch.load(ckpt_path, map_location="cpu")
        if "cfg" not in ck:
            raise KeyError(ckpt_path)
        self.train_cfg: Dict[str, Any] = ck["cfg"]
        c = self.train_cfg
        model = MLPDynamicsAutoEncoder(
            build_ae_config(c),
            width=int(c.get("mlp_width", 512)),
            depth=int(c.get("mlp_depth", 3)),
            hidden_mult=int(c.get("mlp_hidden_mult", 2)),
            residual=bool(c.get("mlp_residual", True)),
        )
        sd = ck.get("state_dict", ck)
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(self.device).eval()
        self.spectral_radius = float("nan")

    def encode(self, x):
        return self.model.encode(x.to(self.device))

    def decode(self, z):
        return self.model.decode(z)

    def step(self, z):
        return self.model.step(z)

    def linear_operator(self) -> Optional[torch.Tensor]:
        return None


ADAPTERS = {
    "kae": KAEModel,
    "nl_ae": MLPDynamicsModel,
    "wl_ae": MLPDynamicsModel,
    "tensorvar": TensorVarModel,
}


def build_model(
    name: str,
    ckpt_path: str,
    device: Optional[torch.device] = None,
    compile: bool = True,
) -> LatentDAModel:
    if name not in ADAPTERS:
        raise KeyError(name)
    return ADAPTERS[name](ckpt_path, device=device, compile=compile)
