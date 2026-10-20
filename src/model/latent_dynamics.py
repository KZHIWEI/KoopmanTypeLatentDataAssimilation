from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.ae import AEConfig, ViTEncoder, ViTDecoder


class SwiGLULayer(nn.Module):

    def __init__(self, d_in: int, d_out: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.fc = nn.Linear(d_in, 2 * hidden)
        self.proj = nn.Linear(hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc(self.norm(x)).chunk(2, dim=-1)
        return self.proj(F.silu(a) * b)


class LatentMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        width: int = 512,
        depth: int = 3,
        hidden_mult: int = 2,
        zero_init: bool = True,
    ):
        super().__init__()
        w = min(width, dim)
        hidden = hidden_mult * w
        self.in_proj = nn.Linear(dim, w)
        self.layers = nn.ModuleList([SwiGLULayer(w, w, hidden) for _ in range(depth)])
        self.out_proj = nn.Linear(w, dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z)
        for layer in self.layers:
            h = layer(h)
        return self.out_proj(h)


class MLPDynamicsAutoEncoder(nn.Module):

    def __init__(
        self,
        cfg: AEConfig,
        width: int = 512,
        depth: int = 3,
        hidden_mult: int = 2,
        residual: bool = True,
    ):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.residual = residual
        self.encoder = ViTEncoder(cfg)
        self.decoder = ViTDecoder(cfg)
        self.dynamics = LatentMLP(
            cfg.latent_dim,
            width=width,
            depth=depth,
            hidden_mult=hidden_mult,
            zero_init=residual,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.dynamics(z) if self.residual else self.dynamics(z)

    def rollout(self, z0: torch.Tensor, steps: int) -> torch.Tensor:
        zs = [z0]
        z = z0
        for _ in range(steps):
            z = self.step(z)
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
        return list(self.dynamics.parameters())

    def load_pretrained_ae(self, path: str, map_location="cpu") -> None:
        ck = torch.load(path, map_location=map_location)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert all(m.startswith("dynamics.") for m in missing), missing

    def compile_submodules(self, **kw) -> "MLPDynamicsAutoEncoder":
        self.encoder = torch.compile(self.encoder, **kw)
        self.decoder = torch.compile(self.decoder, **kw)
        self.dynamics = torch.compile(self.dynamics, **kw)
        return self

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LatentMLPFiLM(nn.Module):

    def __init__(
        self,
        dim: int,
        ctrl_dim: int,
        width: int = 512,
        depth: int = 3,
        hidden_mult: int = 2,
        film_hidden: int = 64,
        zero_init: bool = True,
    ):
        super().__init__()
        w = min(width, dim)
        hidden = hidden_mult * w
        self.depth, self.w = depth, w
        self.in_proj = nn.Linear(dim, w)
        self.layers = nn.ModuleList([SwiGLULayer(w, w, hidden) for _ in range(depth)])
        self.film = nn.Sequential(
            nn.Linear(ctrl_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, depth * 2 * w),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.out_proj = nn.Linear(w, dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        gb = self.film(e).view(-1, self.depth, 2, self.w)
        h = self.in_proj(z)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = h * (1 + gb[:, i, 0]) + gb[:, i, 1]
        return self.out_proj(h)


class MLPAECAutoEncoder(nn.Module):

    def __init__(
        self,
        cfg: AEConfig,
        control_dim: int = 16,
        embed_hidden: int = 64,
        width: int = 512,
        depth: int = 3,
        hidden_mult: int = 2,
        residual: bool = True,
    ):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.m = int(control_dim)
        self.residual = residual
        self.encoder = ViTEncoder(cfg)
        self.decoder = ViTDecoder(cfg)
        self.embed = nn.Sequential(
            nn.Linear(1, embed_hidden), nn.SiLU(), nn.Linear(embed_hidden, self.m)
        )
        self.dynamics = LatentMLPFiLM(
            cfg.latent_dim,
            self.m,
            width=width,
            depth=depth,
            hidden_mult=hidden_mult,
            zero_init=residual,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def control(self, eta: torch.Tensor) -> torch.Tensor:
        eta = eta.reshape(-1, 1).to(self.embed[0].weight.dtype)
        return self.embed(eta)

    def step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return z + self.dynamics(z, u) if self.residual else self.dynamics(z, u)

    def rollout(self, z0: torch.Tensor, eta: torch.Tensor, steps: int) -> torch.Tensor:
        u = self.control(eta)
        zs, z = [z0], z0
        for _ in range(steps):
            z = self.step(z, u)
            zs.append(z)
        return torch.stack(zs, dim=0)

    def forward(self, x0: torch.Tensor, eta: torch.Tensor, steps: int) -> torch.Tensor:
        zs = self.rollout(self.encode(x0), eta, steps)
        s1, b = zs.shape[0], zs.shape[1]
        dec = self.decode(zs.reshape(s1 * b, -1))
        return dec.reshape(s1, b, *dec.shape[1:])

    def ae_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def koopman_parameters(self):
        return list(self.dynamics.parameters()) + list(self.embed.parameters())

    def load_pretrained_ae(self, path: str, map_location="cpu") -> None:
        ck = torch.load(path, map_location=map_location)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        missing, _ = self.load_state_dict(sd, strict=False)
        assert all(m.split(".")[0] in ("dynamics", "embed") for m in missing), missing

    def compile_submodules(self, **kw) -> "MLPAECAutoEncoder":
        self.encoder = torch.compile(self.encoder, **kw)
        self.decoder = torch.compile(self.decoder, **kw)
        return self

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
