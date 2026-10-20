from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
OUT = os.path.join(_REPO, "results", "latent", "cache")
DS = ["lorenz96", "cylinder", "shallow_water", "era5", "era5_high"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DS))
    ap.add_argument(
        "--density",
        type=int,
        required=True,
        choices=[5, 15, 30, 50],
    )
    ap.add_argument(
        "--members",
        type=int,
        default=200,
    )
    a = ap.parse_args()
    sys.path.insert(0, _REPO)
    os.chdir(_REPO)
    from src.eval.benchmark import cgkn_env

    script, base = cgkn_env(a.dataset, a.density)
    if not os.path.exists(base["CKPT"]):
        raise SystemExit(base["CKPT"])

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "strict.npz")
        env = dict(
            os.environ,
            OUT=raw,
            LATENT=str(a.members),
            DEVICE="cuda" if os.environ.get("DEVICE") != "cpu" else "cpu",
            **base,
        )
        r = subprocess.run(
            [sys.executable, os.path.join(_REPO, script)], env=env, cwd=_REPO
        )
        if r.returncode != 0 or not os.path.exists(raw):
            raise SystemExit(r.returncode)
        d = np.load(raw, allow_pickle=True)
        if "lat_tru" not in d:
            raise SystemExit(raw)
        meta = json.loads(str(d["meta"]))
        rr = meta.get("rrmse")
        if rr is None or not np.isfinite(float(rr)):
            raise SystemExit(rr)
        tru, ana = d["lat_tru"], d["lat_ana"]

    suffix = "" if a.density == 30 else f"_obs{a.density}"
    p = os.path.join(OUT, f"{a.dataset}_CGKN{suffix}.npz")
    np.savez(
        p,
        tru=tru,
        bef=np.zeros(0),
        ana=ana,
        meta=json.dumps(
            dict(
                dataset=a.dataset,
                model="CGKN",
                density=a.density,
                nr=int(tru.shape[0]),
                n_protocol_starts=int(tru.shape[0]),
                extended_members=False,
                T=int(tru.shape[1]),
                spinup=meta.get("spinup"),
                obs_ratio=a.density / 100.0,
                rrmse=meta.get("rrmse"),
                protocol=meta.get("protocol"),
            )
        ),
    )
    print(f"  wrote {p}   tru{tru.shape} ana{ana.shape}  rrmse={meta.get('rrmse')}")


if __name__ == "__main__":
    main()
