from __future__ import annotations

import torch
import torch.nn as nn

from src.model.ae import AEConfig, ViTEncoder, ViTDecoder


class KoopmanAutoEncoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d = cfg.latent_dim
        self.encoder = ViTEncoder(cfg)
        self.decoder = ViTDecoder(cfg)
        self.K = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.K.weight.copy_(torch.eye(d) + 1e-4 * torch.randn(d, d))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        return self.K(z)

    def rollout(self, z0: torch.Tensor, steps: int) -> torch.Tensor:
        zs = [z0]
        z = z0
        for _ in range(steps):
            z = self.K(z)
            zs.append(z)
        return torch.stack(zs, dim=0)

    def forward(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        zs = self.rollout(self.encode(x0), steps)
        s1, b = zs.shape[0], zs.shape[1]
        dec = self.decode(zs.reshape(s1 * b, -1))
        return dec.reshape(s1, b, *dec.shape[1:])

    def ae_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def koopman_parameters(self):
        return list(self.K.parameters())

    def load_pretrained_ae(self, path: str, map_location="cpu") -> None:
        ck = torch.load(path, map_location=map_location)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert all(m.startswith("K.") for m in missing), missing

    def compile_submodules(self, **kw) -> "KoopmanAutoEncoder":
        self.encoder = torch.compile(self.encoder, **kw)
        self.decoder = torch.compile(self.decoder, **kw)
        return self

    def spectral_radius(self) -> float:
        with torch.no_grad():
            ev = torch.linalg.eigvals(self.K.weight.detach().float().cpu())
            return float(ev.abs().max())

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
