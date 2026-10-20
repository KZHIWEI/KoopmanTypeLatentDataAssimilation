import argparse
import json
import os
import statistics
import subprocess
import sys

DS = "lorenz96"
DENSITIES = [5, 15, 30, 50]
MODELS = ["dmd", "kae", "nl_ae", "wl_ae", "tensorvar", "cgkn", "dbf"]
OE = [1, 2, 3, 5]
SEEDS = [1, 2, 3]
ROOT = os.path.join("results", "assimilation_interval", DS)
OUT = os.path.join("results", "assimilation_interval", f"{DS}.json")


def cell_path(model, dens, oe, seed):
    return os.path.join(ROOT, model, f"obs{dens}_oe{oe}_seed{seed}.json")


def seed_ckpts(model, seed, dens):
    from src.eval import benchmark

    e = benchmark.REG["datasets"][DS].get(model) or {}
    out = []
    for key in ("ckpt", "fwd"):
        raw = e.get(key)
        for q in [] if raw is None else raw if isinstance(raw, list) else [raw]:
            out.append(q.replace("/seed1/", f"/seed{seed}/").format(d=dens))
    return out


def one(model, oe, dens, seed):
    import torch
    from src.assimilation.variational import assimilate_runs
    from src.eval import benchmark

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model in ("nl_ae", "wl_ae", "kae"):
        cfg, m, runs, sampler, Bi, rl = benchmark.setup(DS, model, dens, dev)
        sr, sq, _, _ = benchmark.sigmas(DS, model, dens)
        da = assimilate_runs(
            m,
            runs,
            sampler,
            Bi,
            sq,
            sr,
            window_size=int(cfg["window_size"]),
            spinup=int(cfg["spinup"]),
            run_length=rl,
            obs_every=int(cfg["obs_every"]),
            iters=int(cfg["iters"]),
            lr=float(cfg["lr"]),
            constraint="weak",
            converged=True,
        )
    else:
        fn = {
            "dmd": benchmark.run_dmd_cell,
            "tensorvar": benchmark.run_tv_cell,
            "dbf": benchmark.run_dbf_cell,
            "cgkn": benchmark.run_cgkn_cell,
        }[model]
        da, _, cfg = fn(DS, dens, dev)
    assert int(cfg["obs_every"]) == oe, (cfg["obs_every"], oe)
    spin = int(cfg["spinup"])
    rec = {
        "model": model,
        "density": dens,
        "obs_every": oe,
        "seed": seed,
        "rrmse": float(da[:, spin:].mean()),
        "per_run_rrmse": [float(x) for x in da[:, spin:].mean(axis=1)],
    }
    out = cell_path(model, dens, oe, seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rec, open(out, "w"), indent=1)
    print(
        f"  {model:<10} obs{dens:<2} oe={oe} seed{seed}  {rec['rrmse']:.5f}", flush=True
    )


def aggregate():
    out = {}
    for model in MODELS:
        for dens in DENSITIES:
            for oe in OE:
                p = cell_path(model, dens, oe, 1)
                if not os.path.exists(p):
                    continue
                record = json.load(open(p))
                v = record["per_run_rrmse"]
                if record["seed"] != 1 or not v:
                    raise ValueError(f"Invalid seed 1 trajectories: {model}, {dens}, {oe}")
                out.setdefault(model, {}).setdefault(f"obs{dens}", {})[f"oe{oe}"] = {
                    "mean": statistics.mean(v),
                    "std": statistics.stdev(v) if len(v) > 1 else 0.0,
                    "n": len(v),
                    "per_run_rrmse": v,
                    "seed": 1,
                    "std_over": "test_trajectories",
                    "std_ddof": 1,
                }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--obs-every", type=int, nargs="+", default=OE)
    ap.add_argument("--density", type=int, nargs="+", default=DENSITIES)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--child", nargs=4)
    a = ap.parse_args()
    if a.aggregate_only:
        aggregate()
        return
    if a.child:
        one(a.child[0], *map(int, a.child[1:]))
        return
    for seed in a.seeds:
        for dens in a.density:
            for model in a.models:
                if seed != 1 and model == "dmd":
                    continue
                if seed != 1 and not all(
                    os.path.exists(q) for q in seed_ckpts(model, seed, dens)
                ):
                    print(f"  {model} obs{dens} seed{seed} missing", flush=True)
                    continue
                for oe in a.obs_every:
                    if os.path.exists(cell_path(model, dens, oe, seed)) and not a.force:
                        continue
                    env = dict(
                        os.environ, OBS_EVERY_OVERRIDE=str(oe), TQDM_DISABLE="1",
                        SEED=str(seed), EVAL_SEED=str(seed),
                    )
                    args = [model, oe, dens, seed]
                    cmd = [sys.executable, "-u", "-m", __spec__.name, "--child"]
                    subprocess.run(cmd + [str(x) for x in args], env=env, check=True)
    aggregate()


if __name__ == "__main__":
    main()
