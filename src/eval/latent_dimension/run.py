import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from src.config import build_ae_config, load_config
from src.data import SequenceDataModule
from src.model.kae import KoopmanAutoEncoder
from src.train.common import compile_model

ARMS = (120, 80, 40, 20)
CKPT = "ckpt_release/ablation/latent_dimension/latent{n}/kae.pt"
OUT = "results/latent_dimension/forecast_rrmse.json"


@torch.no_grad()
def pred_curve(n, steps, dev, compile_on=True):
    cfg = dict(load_config("config/kae.yaml", f"lorenz96_latent{n}"))
    ck = torch.load(CKPT.format(n=n), map_location="cpu", weights_only=False)
    model = KoopmanAutoEncoder(build_ae_config(ck["cfg"]))
    model.load_state_dict(ck["state_dict"], strict=True)
    model = compile_model(model, compile_on).to(dev).eval()
    cfg["koopman_steps"] = steps
    dm = SequenceDataModule(cfg)
    dm.setup()
    amp = torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=dev == "cuda" and cfg.get("precision", "bf16") == "bf16",
    )
    rows = []
    for w in torch.utils.data.DataLoader(
        dm._val, batch_size=int(cfg.get("batch_size", 32))
    ):
        w = w.to(dev)
        b, tau = w.shape[0], w.shape[1] - 1
        with amp:
            zr = model.rollout(model.encode(w[:, 0]), tau).transpose(0, 1)
            dec = model.decode(zr[:, 1:].reshape(b * tau, -1)).reshape(
                b, tau, *w.shape[2:]
            )
        d, g = dec.float().flatten(2), w[:, 1:].float().flatten(2)
        rows.append(
            (
                torch.linalg.norm(d - g, dim=2)
                / torch.linalg.norm(g, dim=2).clamp_min(1e-12)
            )
            .cpu()
            .numpy()
        )
    R = np.concatenate(rows, 0)
    return {
        "n_windows": int(R.shape[0]),
        "per_step_rrmse": R.mean(0).round(6).tolist(),
        "per_step_rrmse_std": R.std(0, ddof=1).round(6).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--arms", default=",".join(map(str, ARMS)))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for n in map(int, a.arms.split(",")):
        r = pred_curve(n, a.steps, dev, not a.no_compile)
        out[str(n)] = r
        print(
            f"  latent {n:3d}  step1 {r['per_step_rrmse'][0]:.4f}  "
            f"step{a.steps} {r['per_step_rrmse'][-1]:.4f}  ({r['n_windows']} windows)",
            flush=True,
        )
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
