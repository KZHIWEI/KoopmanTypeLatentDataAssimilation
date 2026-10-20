from __future__ import annotations

import numpy as np
import torch

_SEED_MOD = 2**31 - 1


def build_mask(
    image_size,
    obs_ratio: float,
    kind: str = "random",
    seed: int = 0,
) -> torch.Tensor:
    h, w = int(image_size[0]), int(image_size[1])
    if obs_ratio >= 1.0:
        return torch.ones(1, h, w)
    rng = np.random.RandomState(seed)
    m = np.zeros((h, w), dtype=np.float32)

    if kind == "random":
        n = int(round(obs_ratio * h * w))
        idx = rng.choice(h * w, size=n, replace=False)
        m.flat[idx] = 1.0
    elif kind == "stride":
        k = max(1, int(round(1.0 / obs_ratio)))
        m.flat[::k] = 1.0
    elif kind == "block":
        side = int(round(np.sqrt(obs_ratio * h * w)))
        side = max(1, min(side, min(h, w)))
        r0 = (h - side) // 2
        c0 = (w - side) // 2
        m[r0 : r0 + side, c0 : c0 + side] = 1.0
    else:
        raise ValueError(kind)

    return torch.from_numpy(m)[None]


class MaskSampler:

    def __init__(
        self,
        image_size,
        obs_ratio: float,
        kind: str = "random",
        base_seed: int = 0,
        fixed: bool = False,
    ):
        self.image_size = image_size
        self.obs_ratio = float(obs_ratio)
        self.kind = kind
        self.base_seed = int(base_seed)
        self.fixed = bool(fixed)
        self._fixed_mask = (
            build_mask(image_size, self.obs_ratio, kind, self.base_seed)
            if self.fixed
            else None
        )

    def mask(self, run_idx: int, step: int) -> torch.Tensor:
        if self.fixed:
            return self._fixed_mask
        seed = (self.base_seed * 1_000_003 + run_idx * 10_007 + step) % _SEED_MOD
        return build_mask(self.image_size, self.obs_ratio, self.kind, seed)
