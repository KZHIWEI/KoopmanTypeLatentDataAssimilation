from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


def _load_trajs(path: str) -> np.ndarray:
    path = _abspath(path)
    if path.endswith(".npy"):
        a = np.load(path, mmap_mode="r")
        if a.ndim == 4:
            return a[:, :, None]
        if a.ndim == 5:
            return a
    elif path.endswith(".h5"):
        import h5py

        with h5py.File(path, "r") as f:
            d = f["data"][:]
        d = np.transpose(d, (0, 3, 1, 2))
        return d[None]
    raise ValueError(path)


_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _abspath(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def channel_norm(trajs: np.ndarray, cfg: dict):
    arr = trajs[None] if trajs.ndim == 4 else trajs
    c = arr.shape[2]
    if "norm_mean" in cfg and "norm_std" in cfg:

        def _v(x):
            a = np.asarray(x, np.float32).reshape(-1)
            if a.size == 1:
                return np.full(c, float(a[0]), np.float32)
            if a.size != c:
                raise ValueError(a.size, c)
            return a

        return _v(cfg["norm_mean"]), _v(cfg["norm_std"])
    flat = (
        np.asarray(arr[:, :: max(1, arr.shape[1] // 200)])
        .astype(np.float32)
        .reshape(-1, c, arr.shape[3], arr.shape[4])
    )
    return (
        flat.mean(axis=(0, 2, 3)).astype(np.float32),
        (flat.std(axis=(0, 2, 3)) + 1e-6).astype(np.float32),
    )


def load_split(path):
    p = _abspath(path)
    if p.endswith(".h5"):
        from src.assimilation.dmd import _LazyH5

        return _LazyH5(p)
    return _load_trajs(p)


def train_norm(cfg):
    tr = load_split(cfg["data_path"])
    return channel_norm(np.asarray(tr[:, :: max(1, tr.shape[1] // 200)]), cfg)


class _FrameDataset(Dataset):
    def __init__(self, frames: np.ndarray, idx: np.ndarray, mean, std):
        self.frames = frames
        self.idx = idx
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x = np.asarray(self.frames[self.idx[i]], dtype=np.float32)
        return torch.from_numpy((x - self.mean) / self.std)


class FieldDataModule(L.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.bs = int(cfg["batch_size"])
        self.nval = int(cfg["num_val_frames"])
        self.c = int(cfg["in_channels"])
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        self._train = None
        self._val = None

    def setup(self, stage: Optional[str] = None):
        if self._train is not None:
            return
        tr_t = _load_trajs(self.cfg["data_path"])
        va_t = _load_trajs(self.cfg["val_path"])
        self.mean, self.std = channel_norm(tr_t, self.cfg)
        tr = tr_t.reshape(-1, *tr_t.shape[2:])
        va = va_t.reshape(-1, *va_t.shape[2:])
        gtr = np.arange(tr.shape[0])
        gva = np.random.RandomState(0).choice(
            va.shape[0], size=min(self.nval, va.shape[0]), replace=False
        )
        self._train = _FrameDataset(tr, gtr, self.mean, self.std)
        self._val = _FrameDataset(va, gva, self.mean, self.std)
        self._val_all = va

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self.bs,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self.bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def full_val_dataset(self):
        return _FrameDataset(
            self._val_all, np.arange(self._val_all.shape[0]), self.mean, self.std
        )


class _SeqDataset(Dataset):

    def __init__(self, trajs: np.ndarray, starts, K: int, skip: int, mean, std):
        self.trajs = trajs
        self.starts = starts
        self.K = K
        self.skip = skip
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        tr, t0 = self.starts[i]
        idx = t0 + self.skip * np.arange(self.K + 1)
        w = np.asarray(self.trajs[tr, idx], dtype=np.float32)
        return torch.from_numpy((w - self.mean) / self.std)


class SequenceDataModule(L.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bs = int(cfg["batch_size"])
        self.K = int(cfg["koopman_steps"])
        self.skip = int(cfg.get("skip", 1))
        self.nval = int(cfg["num_val_windows"])
        self.mean = None
        self.std = None
        self._train = self._val = None

    def _starts(self, trajs, rng, n=None):
        ntraj, T = trajs.shape[0], trajs.shape[1]
        t_lo = int(self.cfg.get("train_frame_start", 0))
        t_hi = (
            min(T, int(self.cfg["train_frames"])) if self.cfg.get("train_frames") else T
        )
        span = self.skip * self.K
        all_starts = [
            (tr, t0) for tr in range(ntraj) for t0 in range(t_lo, t_hi - span)
        ]
        if n is not None and n < len(all_starts):
            idx = rng.choice(len(all_starts), size=n, replace=False)
            return [all_starts[i] for i in idx]
        return all_starts

    def setup(self, stage=None):
        if self._train is not None:
            return
        tr = _load_trajs(self.cfg["data_path"])
        va = _load_trajs(self.cfg["val_path"])
        self.mean, self.std = channel_norm(tr, self.cfg)
        rng = np.random.RandomState(0)
        ntr = self.cfg.get("num_train_windows")
        self._train = _SeqDataset(
            tr,
            self._starts(tr, rng, int(ntr) if ntr else None),
            self.K,
            self.skip,
            self.mean,
            self.std,
        )
        self._val = _SeqDataset(
            va, self._starts(va, rng, self.nval), self.K, self.skip, self.mean, self.std
        )

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self.bs,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self.bs,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
