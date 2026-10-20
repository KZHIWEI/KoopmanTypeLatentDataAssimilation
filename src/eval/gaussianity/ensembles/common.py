from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

DATASETS = ("lorenz96", "cylinder", "shallow_water", "era5", "era5_high")

MODELS = ("NL-AE", "WL-AE", "KAE", "TensorVar", "DMD", "DBF", "CGKN")


def sigma_for(ds: str, model: str, density: int) -> Tuple[float, float, str]:
    from src.eval.benchmark import sigmas

    sr, sq, shard, _ = sigmas(ds, model.lower().replace("-", "_"), density)
    return (
        sr,
        sq,
        f"config/sweep.json[{ds}:{model.lower().replace('-', '_')}:obs{density}] <- {shard}",
    )


def chol_from_precision(Bi: torch.Tensor, dev) -> Tuple[torch.Tensor, np.ndarray]:
    if Bi.dim() == 1:
        var = Bi.double().clamp(min=1e-30).reciprocal()
        L = torch.diag(var.sqrt()).float()
        return L, torch.diag(var).cpu().numpy()
    Bm = torch.linalg.inv(Bi.double())
    Bm = 0.5 * (Bm + Bm.T)
    Bm = Bm + (1e-10 * torch.trace(Bm) / Bm.shape[0]) * torch.eye(
        Bm.shape[0], dtype=torch.float64, device=Bm.device
    )
    return torch.linalg.cholesky(Bm).float(), Bm.cpu().numpy()


class Draw:
    def __init__(self, seed: int):
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def bg(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.g)


def load_protocol(ds: str, density: int, cfgfile: str = "config/da.yaml"):
    from src.config import load_config
    from src.assimilation.data import DAData
    from src.assimilation.observation import MaskSampler

    cfg = load_config(cfgfile, f"{ds}_obs{density}")
    data = DAData(cfg)
    ev = cfg["eval"]
    run_length = int(ev["length"])
    runs = data.build_runs(
        data.eval_starts(ev, int(cfg["skip"])), int(cfg["skip"]), run_length
    )
    sampler = MaskSampler(
        cfg["image_size"],
        float(cfg["obs_ratio"]),
        cfg.get("mask_kind", "random"),
        int(cfg.get("mask_seed", 0)),
    )
    geom = dict(
        window=int(cfg["window_size"]),
        spinup=int(cfg["spinup"]),
        obs_every=int(cfg["obs_every"]),
        run_length=run_length,
        n_starts=len(runs),
    )
    return cfg, data, runs, sampler, geom


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
