import argparse
import importlib.util
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
os.environ["FULL_OBS"] = "1"
os.environ.setdefault("CKDIR", "ckpt_release/ablation/cgkn_full_obs/era5")

from src.eval.latent.tsne import embed

SOLVER = "src/model/cgkn/cgkn_era5.py"
OUT = "results/cgkn_full_obs/tsne.json"


def legs(M, model, x):
    T = x.shape[0]
    u1, u2 = M.split_field(x)
    with torch.no_grad():
        z_true = model.ae.enc(u2).reshape(T, M.DZ)
        u1c, zc = u1[:1].clone(), model.ae.enc(u2[:1])
        fc = [zc.reshape(1, M.DZ)]
        for _ in range(T - 1):
            f1, g1, f2, g2 = model.cgn(u1c)
            znb = zc[:, model.cgn.nbr.reshape(-1), :].reshape(-1, M.NB, 9 * M.DZU)
            du1 = f1 + torch.einsum(
                "bnok,bnk->bno", g1.reshape(-1, M.NB, M.DU1_P, 9 * M.DZU), znb
            )
            dz = f2 + torch.einsum(
                "bnok,bnk->bno", g2.reshape(-1, M.NB, M.DZU, 9 * M.DZU), znb
            )
            u1c, zc = u1c + du1, zc + dz
            fc.append(zc.reshape(1, M.DZ))
        mu = M.kalman_window(
            model,
            u1.reshape(T, M.DU1),
            s1sq=0.05,
            s2sq=1.0,
            mu0=model.ae.enc(u2[:1]).reshape(M.DZ),
        )
    return {
        "reference": z_true.float().cpu().numpy(),
        "forecast": torch.cat(fc, 0).float().cpu().numpy(),
        "analysis": mu.float().cpu().numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--length", type=int, default=40)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = importlib.util.spec_from_file_location("cgkn_solver", SOLVER)
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    model = M.CGKN().to(dev)
    ck = f"{M.CKDIR}/cgkn.pt"
    model.load_state_dict(torch.load(ck, map_location=dev))
    model.eval()
    te = M.load_frames("test")
    nrm = M.Norm(M.load_frames("train"))
    last = te.shape[0] - a.length - 2
    starts = np.linspace(0, last, a.windows).astype(int)
    out = {"ckpt": ck, "length": a.length + 1, "windows": {}}
    for s in starts:
        x = nrm(te[s : s + a.length + 1]).to(dev)
        out["windows"][str(int(s))] = embed(legs(M, model, x))
        print(f"  window at step {int(s)}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
