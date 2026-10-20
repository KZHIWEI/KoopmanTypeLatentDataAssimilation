from __future__ import annotations

import os
from typing import Any, Dict

import yaml

from src.model.ae import AEConfig

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CFGDIR = next(
    (
        d
        for d in ("config", "config")
        if os.path.exists(os.path.join(_REPO, d, "dataset.yaml"))
    ),
    "config",
)
_DATASET_YAML = os.path.join(_REPO, _CFGDIR, "dataset.yaml")


def load_dataset(dataset: str, path: str = _DATASET_YAML) -> Dict[str, Any]:
    if not os.path.isabs(path):
        path = os.path.join(_REPO, path)
    with open(path) as f:
        reg = yaml.safe_load(f) or {}
    if dataset not in reg:
        raise KeyError(dataset, path)
    block = dict(reg[dataset])
    base = block.pop("data", None)
    if base is not None:
        fields = load_dataset(base, path)
        fields.update(block)
        block = fields
    block["dataset"] = dataset
    return block


def _resolve_cfg(path: str) -> str:
    full = os.path.join(_REPO, path)
    if os.path.exists(full):
        return full
    head, _, tail = path.partition("/")
    other = {"config": "config", "config": "config"}.get(head)
    if other and tail:
        alt = os.path.join(_REPO, other, tail)
        if os.path.exists(alt):
            return alt
    return full


def load_config(path: str, dataset: str) -> Dict[str, Any]:
    if not os.path.isabs(path):
        path = _resolve_cfg(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = dict(raw.get("defaults", {}))
    with open(_DATASET_YAML) as f:
        registered = dataset in (yaml.safe_load(f) or {})
    if registered:
        cfg.update(load_dataset(dataset))
    if dataset in raw:
        cfg.update(raw[dataset])
    elif not registered:
        raise KeyError(dataset, path)
    cfg["dataset"] = dataset
    cfg["out_root"] = ckpt_root(cfg)
    _s = os.environ.get("SEED")
    if _s:
        cfg["seed"] = int(_s)
    _oe = os.environ.get("OBS_EVERY_OVERRIDE")
    if _oe and "obs_every" in cfg:
        cfg["obs_every"] = int(_oe)
    return cfg


def ckpt_root(cfg: Dict[str, Any]) -> str:
    return os.environ.get("CKPT_ROOT") or cfg["out_root"]


def build_ae_config(cfg: Dict[str, Any]) -> AEConfig:
    return AEConfig(
        in_channels=int(cfg["in_channels"]),
        image_size=tuple(cfg["image_size"]),
        patch=int(cfg["patch"]),
        dim=int(cfg["dim"]),
        depth=int(cfg["depth"]),
        heads=int(cfg["heads"]),
        tok_dim=int(cfg["tok_dim"]),
        mlp_ratio=float(cfg["mlp_ratio"]),
        refine=bool(cfg["refine"]),
        arch=str(cfg.get("arch", "vit")),
        mlp_hidden=tuple(cfg.get("mlp_hidden", (512, 256))),
        latent_dim_override=(
            None if cfg.get("latent_dim") is None else int(cfg["latent_dim"])
        ),
    )


def ckpt_path(cfg, model: str, name: str) -> str:
    import os

    parts = [
        ckpt_root(cfg),
        cfg.get("release_dataset", cfg["dataset"]),
        cfg.get("release_model", model),
    ]
    seed = os.environ.get("SEED")
    if seed:
        parts.append(f"seed{seed}")
    d = os.path.join(*parts)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)
