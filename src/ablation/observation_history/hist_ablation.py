from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from src.assimilation.variational import nmc_background
from src.assimilation.observation import MaskSampler, build_mask
from src.assimilation.dmd import DMDDA, assimilate_runs_dmd, estimate_QR_dmd, proj_frame
from src.config import load_config
from src.data import _load_trajs
from src.eval.benchmark import TV_DRIVER, _common, tv_setup, tv_solve
from src.eval.covariance_selection import dmd_sigmas

OUT = "results/observation_history/{family}.json"


@torch.no_grad()
def fit_diag(
    model: DMDDA,
    cfg,
    m: int,
    mean,
    std,
    n_samples: int = 6000,
    ridge: float = 1e-2,
    seed: int = 777,
) -> torch.Tensor:
    tr = _load_trajs(cfg["data_path"])
    mean, std = mean.reshape(1, -1, 1, 1), std.reshape(1, -1, 1, 1)
    ntraj, T = tr.shape[0], tr.shape[1]
    skip = int(cfg["skip"])
    H, W = cfg["image_size"]
    ratio, kind = float(cfg["obs_ratio"]), cfg.get("mask_kind", "random")
    r, dev = model.D, model.device
    oe = int(cfg["obs_every"])
    rng = np.random.RandomState(seed)
    n_obs = -(-m // oe)
    G = torch.zeros(r, n_obs, n_obs, device=dev, dtype=torch.float64)
    B = torch.zeros(r, n_obs, device=dev, dtype=torch.float64)
    for _ in range(n_samples):
        ti = int(rng.randint(0, ntraj))
        t0 = int(rng.randint((n_obs - 1) * oe * skip, T))
        idx = [max(0, t0 - k * oe * skip) for k in range(n_obs)][::-1]
        fr = (np.asarray(tr[ti, idx]).astype(np.float32) - mean) / std
        f = torch.stack(
            [
                proj_frame(
                    model,
                    torch.from_numpy(fr[j]).to(dev),
                    build_mask([H, W], ratio, kind, int(rng.randint(0, 1 << 30))).to(
                        dev
                    ),
                )
                for j in range(n_obs)
            ],
            dim=1,
        ).double()
        z = model.encode(torch.from_numpy(fr[-1:]).to(dev))[0].double()
        G += f.unsqueeze(2) * f.unsqueeze(1)
        B += f * z.unsqueeze(1)
    scale = G.diagonal(dim1=1, dim2=2).mean(dim=1).clamp(min=1e-30)
    G += (
        ridge * scale.view(-1, 1, 1) * torch.eye(n_obs, device=dev, dtype=torch.float64)
    )
    return torch.linalg.solve(G, B.unsqueeze(2)).squeeze(2).float()


def dmd(a, dev):
    cfg, data, runs, rl = _common(a.dataset, a.density)
    model = DMDDA(cfg, f"{a.dataset}_obs{a.density}", dev, r=0)
    w = fit_diag(model, cfg, a.history, data.mean, data.std, n_samples=a.nfit)
    sampler = MaskSampler(
        cfg["image_size"],
        float(cfg["obs_ratio"]),
        cfg.get("mask_kind", "random"),
        int(cfg.get("mask_seed", 0)),
    )
    samples = data.nmc_samples(
        int(cfg["nmc_lead"]),
        int(cfg["skip"]),
        int(cfg["nmc_samples"]),
        seed=int(cfg.get("mask_seed", 0)),
    )
    Bi = nmc_background(
        model,
        data.frame_batched,
        samples,
        int(cfg["nmc_lead"]),
        int(cfg["skip"]),
        diagonal=bool(cfg.get("b_diagonal", False)),
        ridge=float(cfg["b_ridge"]),
    ).to(dev)
    Qi, Ri = estimate_QR_dmd(model, cfg, sampler)
    spin = int(cfg["spinup"])
    sr, sq, source = dmd_sigmas(a.dataset, a.density)

    def da(hw, hm):
        arr = assimilate_runs_dmd(
            model,
            runs,
            sampler,
            Bi,
            sq,
            sr,
            window_size=int(cfg["window_size"]),
            spinup=spin,
            run_length=rl,
            obs_every=int(cfg["obs_every"]),
            constraint=cfg.get("constraint", "weak"),
            Qi=Qi,
            Ri=Ri,
            hist_m=hm,
            hist_w=hw,
        )
        return float(arr[:, spin:].mean())

    return dict(
        rrmse=da(w, a.history), baseline=da(None, 1), r=model.D,
        sigma_r=sr, sigma_q=sq, sigma_source=source,
    )


def tensorvar(a, dev):
    root = a.root or f"ckpt_release/ablation/observation_history/{a.dataset}"
    inv = f"inverse_encoder@{a.density}_m{a.history}.pt"
    if not os.path.exists(os.path.join(root, inv)):
        inv = os.path.abspath(
            f"ckpt_release/{a.dataset}/tensorvar/seed2/inverse_encoder@{a.density}.pt"
        )
    D = dict(
        TV_DRIVER[a.dataset],
        fwd=os.path.abspath(f"ckpt_release/{a.dataset}/tensorvar/seed2/forward.pt"),
        inv=inv,
    )
    if a.dataset == "lorenz96":
        D.update(
            kind="closed_form", cov="calibration", stats_batch=64, width=96, n_attn=4
        )
    t = tv_setup(a.dataset, a.density, dev, driver=D, root=root)
    arr, _ = tv_solve(
        t["solve"],
        t["m"],
        t["runs"],
        t["sampler"],
        t["Bi"],
        t["sq"],
        t["sr"],
        spinup=t["spin"],
        **t["kw"],
    )
    return dict(
        rrmse=float(arr[:, t["spin"] :].mean()), operator=os.path.join(root, inv)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["dmd", "tensorvar"], default="dmd")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--density", type=int, default=30)
    ap.add_argument("--history", type=int, default=1)
    ap.add_argument("--nfit", type=int, default=6000)
    ap.add_argument("--root", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = dict(
        dataset=a.dataset,
        density=a.density,
        m=a.history,
        n_obs_frames=-(
            -a.history
            // int(
                load_config("config/da.yaml", f"{a.dataset}_obs{a.density}")[
                    "obs_every"
                ]
            )
        ),
        **(dmd if a.family == "dmd" else tensorvar)(a, dev),
    )
    print(json.dumps(rec), flush=True)
    out = OUT.format(family=a.family)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    all_ = json.load(open(out)) if os.path.exists(out) else {}
    all_[f"{a.dataset}@{a.density}::m{a.history}"] = rec
    json.dump(all_, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
