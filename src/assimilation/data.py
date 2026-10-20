from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from src.data import _load_trajs, channel_norm, load_split


def tile_starts(
    ev: Dict[str, Any], skip: int, ntraj: int, T: int
) -> List[Tuple[int, int]]:
    lo, hi = ev.get("frame_range", [0, T])
    last = min(int(hi), T - 1)
    span = skip * int(ev["length"])
    if int(lo) + span > last:
        raise ValueError(lo, span, last)
    if ntraj > 1:
        return [(traj, int(lo)) for traj in range(ntraj)]
    return [(0, t0) for t0 in range(int(lo), last - span + 1, span + skip)]


class DAData:

    def __init__(self, cfg: Dict[str, Any]):
        da_path = cfg.get("da_data_path") or cfg.get("val_path") or cfg["data_path"]
        self.trajs = _load_trajs(da_path)
        self.ntraj, self.T = self.trajs.shape[0], self.trajs.shape[1]
        self.c = self.trajs.shape[2]
        self.nmc_trajs = load_split(cfg["data_path"])

        norm_src = (
            _load_trajs(cfg["data_path"]) if "norm_mean" not in cfg else self.trajs
        )
        mean, std = channel_norm(norm_src, cfg)
        self.mean = mean.reshape(self.c, 1, 1)
        self.std = std.reshape(self.c, 1, 1)

    def frame(self, traj: int, frame_idx: int) -> torch.Tensor:
        x = np.asarray(self.trajs[traj, frame_idx], dtype=np.float32)
        return torch.from_numpy((x - self.mean) / self.std)

    def frame_batched(self, traj: int, frame_idx: int) -> torch.Tensor:
        x = np.asarray(self.nmc_trajs[traj, [frame_idx]], dtype=np.float32)[0]
        return torch.from_numpy((x - self.mean) / self.std).unsqueeze(0)

    def denorm(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def run(self, traj: int, t0: int, skip: int, run_length: int) -> List[torch.Tensor]:
        return [self.frame(traj, t0 + skip * k) for k in range(run_length + 1)]

    def eval_starts(
        self, ev: Dict[str, Any], skip: int, val: bool = False
    ) -> List[Tuple[int, int]]:
        if val:
            return [tuple(s) for s in ev["val_starts"]]
        return tile_starts(ev, skip, self.ntraj, self.T)

    def build_runs(
        self, starts: Sequence[Tuple[int, int]], skip: int, run_length: int
    ) -> List[List[torch.Tensor]]:
        runs = []
        for traj, t0 in starts:
            end = t0 + skip * run_length
            if end >= self.T:
                raise ValueError(traj, t0, end, self.T)
            runs.append(self.run(traj, t0, skip, run_length))
        return runs

    def nmc_samples(
        self, lead: int, skip: int, n: int, seed: int = 0
    ) -> List[Tuple[int, int]]:
        rng = np.random.RandomState(seed)
        lo = 2 * lead * skip
        out = []
        for _ in range(n):
            traj = rng.randint(0, self.nmc_trajs.shape[0])
            frame = rng.randint(lo, self.nmc_trajs.shape[1])
            out.append((int(traj), int(frame)))
        return out
