from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from timm.layers import PatchEmbed, RotaryEmbeddingCat
from timm.models.eva import EvaBlock


@dataclass
class AEConfig:

    in_channels: int = 1
    image_size: Tuple[int, int] = (256, 256)
    patch: int = 16
    dim: int = 224
    depth: int = 9
    heads: int = 8
    tok_dim: int = 8
    mlp_ratio: float = 8.0 / 3.0
    init_values: float = 1e-5
    refine: bool = True
    arch: str = "vit"
    mlp_hidden: Tuple[int, ...] = (512, 256)
    latent_dim_override: Optional[int] = None

    @property
    def grid(self) -> Tuple[int, int]:
        return self.image_size[0] // self.patch, self.image_size[1] // self.patch

    @property
    def num_tokens(self) -> int:
        gh, gw = self.grid
        return gh * gw

    @property
    def latent_dim(self) -> int:
        return (
            self.num_tokens * self.tok_dim
            if self.latent_dim_override is None
            else int(self.latent_dim_override)
        )

    def validate(self) -> None:
        h, w = self.image_size
        assert h % self.patch == 0 and w % self.patch == 0
        assert self.dim % self.heads == 0
        assert (self.dim // self.heads) % 2 == 0
        if self.latent_dim_override is not None:
            assert int(self.latent_dim_override) > 0
            assert (
                self.arch == "mlp"
                or int(self.latent_dim_override) == self.num_tokens * self.tok_dim
            ), (self.num_tokens * self.tok_dim)


def _blocks(cfg: AEConfig) -> nn.ModuleList:
    return nn.ModuleList(
        [
            EvaBlock(
                dim=cfg.dim,
                num_heads=cfg.heads,
                qkv_fused=True,
                mlp_ratio=cfg.mlp_ratio,
                swiglu_mlp=True,
                scale_attn_inner=True,
                num_prefix_tokens=0,
                init_values=cfg.init_values,
            )
            for _ in range(cfg.depth)
        ]
    )


class ViTEncoder(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.arch == "mlp":
            d_in = cfg.in_channels * cfg.image_size[0] * cfg.image_size[1]
            dims = [d_in, *cfg.mlp_hidden, cfg.latent_dim]
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.GELU()]
            self.mlp = nn.Sequential(*layers[:-1])
            return
        head_dim = cfg.dim // cfg.heads
        gh, gw = cfg.grid
        self.rope = RotaryEmbeddingCat(
            dim=head_dim, feat_shape=[gh, gw], in_pixels=False
        )
        self.patch_embed = PatchEmbed(
            img_size=cfg.image_size,
            patch_size=cfg.patch,
            in_chans=cfg.in_channels,
            embed_dim=cfg.dim,
            flatten=True,
        )
        self.blocks = _blocks(cfg)
        self.norm = nn.LayerNorm(cfg.dim)
        self.to_latent = nn.Linear(cfg.dim, cfg.tok_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.arch == "mlp":
            return self.mlp(x.flatten(1))
        rope = self.rope.get_embed().to(x.dtype)
        t = self.patch_embed(x)
        for blk in self.blocks:
            t = blk(t, rope=rope)
        return self.to_latent(self.norm(t)).flatten(1)


class ViTDecoder(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.in_channels
        if cfg.arch == "mlp":
            d_out = c * cfg.image_size[0] * cfg.image_size[1]
            dims = [cfg.latent_dim, *reversed(cfg.mlp_hidden), d_out]
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.GELU()]
            self.mlp = nn.Sequential(*layers[:-1])
            return
        head_dim = cfg.dim // cfg.heads
        gh, gw = cfg.grid
        self.rope = RotaryEmbeddingCat(
            dim=head_dim, feat_shape=[gh, gw], in_pixels=False
        )
        self.from_latent = nn.Linear(cfg.tok_dim, cfg.dim)
        self.blocks = _blocks(cfg)
        self.norm = nn.LayerNorm(cfg.dim)
        self.pixel_head = nn.Linear(cfg.dim, cfg.patch * cfg.patch * c)
        self.refine = None
        if cfg.refine:
            self.refine = nn.Sequential(
                nn.Conv2d(c, 48, 5, 1, 2),
                nn.GELU(),
                nn.Conv2d(48, 48, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(48, c, 3, 1, 1),
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.arch == "mlp":
            H, W = cfg.image_size
            return self.mlp(z).view(-1, cfg.in_channels, H, W)
        gh, gw = cfg.grid
        rope = self.rope.get_embed().to(z.dtype)
        t = self.from_latent(z.view(-1, cfg.num_tokens, cfg.tok_dim))
        for blk in self.blocks:
            t = blk(t, rope=rope)
        p = self.pixel_head(self.norm(t))
        p = p.view(-1, gh, gw, cfg.in_channels, cfg.patch, cfg.patch)
        img = p.permute(0, 3, 1, 4, 2, 5).reshape(
            -1, cfg.in_channels, gh * cfg.patch, gw * cfg.patch
        )
        if self.refine is not None:
            img = img + self.refine(img)
        return img


class ViTAutoEncoder(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.encoder = ViTEncoder(cfg)
        self.decoder = ViTDecoder(cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def compile_submodules(self, **kw) -> "ViTAutoEncoder":
        self.encoder = torch.compile(self.encoder, **kw)
        self.decoder = torch.compile(self.decoder, **kw)
        return self

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
