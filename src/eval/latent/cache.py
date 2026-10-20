import argparse, json, os, sys
import numpy as np
import torch

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO)
os.chdir(_REPO)

from src.config import load_config
from src.assimilation.variational import nmc_background, assimilate_runs, _rrmse_batch
from src.assimilation.data import DAData
from src.assimilation.observation import MaskSampler
from src.assimilation.models import build_model
from src.eval import benchmark

OUT = "results/latent/cache"
DS = ["lorenz96", "cylinder", "shallow_water", "era5", "era5_high"]
dev = torch.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=list(DS))
ap.add_argument("--models", default="kae,wl_ae,nl_ae,tv,dmd,dbf")
ap.add_argument("--density", type=int, default=30, choices=[5, 15, 30, 50])
ap.add_argument(
    "--n-runs",
    type=int,
    default=0,
)
ap.add_argument("--runs", default="")
ap.add_argument("--data-path", default="")
ap.add_argument("--tag", default="")
args = ap.parse_args()

ds = args.dataset
cfg = load_config("config/da.yaml", f"{ds}_obs{args.density}")
if args.data_path:
    cfg = dict(cfg, da_data_path=args.data_path)
os.makedirs(OUT, exist_ok=True)

data = DAData(cfg)
ev = cfg["eval"]
run_length = int(ev["length"])
T = run_length + 1
skip = int(cfg["skip"])
all_starts = data.eval_starts(ev, skip)
sel = (
    [int(i) for i in args.runs.split(",")]
    if args.runs
    else list(range(len(all_starts)))
)
starts = [all_starts[i] for i in sel]
NR = max(args.n_runs, len(starts))
if NR > len(starts):
    from src.data import _load_trajs

    a = _load_trajs(cfg.get("da_data_path") or cfg["data_path"])
    L = run_length * skip
    base = min(int(t0) for _, t0 in starts)
    seen = set(starts)
    for t0 in range(base, max(base + 1, a.shape[1] - L), L + 1):
        for tr in range(a.shape[0]):
            if len(starts) < NR and (tr, t0) not in seen:
                starts.append((tr, t0))
                seen.add((tr, t0))
NR = len(starts)
RUN_SRC = [(int(tr), int(t0)) for tr, t0 in starts]
runs = data.build_runs(starts, skip, run_length)
gt = [torch.stack([runs[b][t] for b in range(NR)]).to(dev) for t in range(T)]
sampler = MaskSampler(
    cfg["image_size"],
    float(cfg["obs_ratio"]),
    cfg.get("mask_kind", "random"),
    int(cfg.get("mask_seed", 0)),
)
spin = int(cfg["spinup"])


def save(model_name, tru, bef, ana, meta):
    n = "" if NR <= len(all_starts) else f"_n{NR}"
    dens = "" if args.density == 30 else f"_obs{args.density}"
    tag = f"_{args.tag}" if args.tag else ""
    p = os.path.join(OUT, f"{args.dataset}_{model_name}{n}{dens}{tag}.npz")
    np.savez(
        p,
        tru=tru,
        bef=(bef if bef is not None else np.zeros(0)),
        ana=ana,
        meta=json.dumps(
            dict(
                meta,
                dataset=args.dataset,
                model=model_name,
                density=args.density,
                nr=NR,
                n_protocol_starts=len(all_starts),
                extended_members=bool(NR > len(all_starts)),
                T=T,
                spinup=spin,
                obs_ratio=float(cfg["obs_ratio"]),
                da_data_path=args.data_path or None,
                split_tag=args.tag or None,
                run_src=RUN_SRC,
            )
        ),
    )
    print(f"[saved] {p}  tru{tru.shape} ana{ana.shape}", flush=True)


@torch.no_grad()
def truth_and_before(m):
    tru = torch.stack([m.encode(gt[t]).float() for t in range(T)], dim=1)
    z = m.encode(gt[0]).float()
    rolls = [z]
    for _ in range(run_length):
        z = m.step(z).float()
        rolls.append(z)
    return tru.cpu().numpy(), torch.stack(rolls, dim=1).cpu().numpy()


@torch.no_grad()
def rrmse_of(decode, ana):
    ana = torch.as_tensor(ana, device=dev)
    errs = [
        _rrmse_batch(decode(ana[:, t]).reshape(gt[t].shape), gt[t]).cpu().numpy()
        for t in range(run_length)
    ]
    return float(np.stack(errs, 1)[:, spin:].mean())


def closed_leg(name, t, mode):
    m = t["m"]
    tru, bef = truth_and_before(m)
    _, ana = benchmark.tv_solve(
        t["solve"],
        m,
        runs,
        t["sampler"],
        t["Bi"],
        t["sq"],
        t["sr"],
        return_analysis=True,
        spinup=spin,
        **dict(t["kw"], run_length=run_length),
    )
    ana = ana.cpu().numpy() if torch.is_tensor(ana) else np.asarray(ana)
    save(
        name,
        tru,
        bef,
        ana,
        dict(mode=mode, sr=t["sr"], sq=t["sq"], rrmse=rrmse_of(m.decode, ana)),
    )


want = set(args.models.split(","))

for arch in ("kae", "wl_ae", "nl_ae"):
    if arch not in want:
        continue
    m = build_model(arch, benchmark.CKPT[(ds, arch)], device=dev, compile=False)
    tru, bef = truth_and_before(m)
    Bi = nmc_background(
        m,
        data.frame_batched,
        data.nmc_samples(
            int(cfg["nmc_lead"]),
            skip,
            int(cfg["nmc_samples"]),
            seed=int(cfg.get("mask_seed", 0)),
        ),
        int(cfg["nmc_lead"]),
        skip,
        diagonal=bool(cfg.get("b_diagonal", False)),
        ridge=float(cfg["b_ridge"]),
    ).to(dev)
    sr, sq, shard, _ = benchmark.sigmas(ds, arch, args.density)
    _, ana = assimilate_runs(
        m,
        runs,
        sampler,
        Bi,
        sq,
        sr,
        window_size=int(cfg["window_size"]),
        spinup=spin,
        run_length=run_length,
        obs_every=int(cfg["obs_every"]),
        iters=int(cfg["iters"]),
        lr=float(cfg["lr"]),
        constraint="weak",
        converged=True,
        return_analysis=True,
    )
    ana = ana.cpu().numpy() if torch.is_tensor(ana) else np.asarray(ana)
    save(
        benchmark.DISP[arch],
        tru,
        bef,
        ana,
        dict(sr=sr, sq=sq, sigma_source=shard, rrmse=rrmse_of(m.decode, ana)),
    )
    del m
    torch.cuda.empty_cache()

if "tv" in want:
    closed_leg("TensorVar", benchmark.tv_setup(ds, args.density, dev), "TV_DRIVER")
    torch.cuda.empty_cache()

if "dmd" in want:
    closed_leg("DMD", benchmark.dmd_setup(ds, args.density, dev), "DMD r=0")
    torch.cuda.empty_cache()

if "dbf" in want:
    from src.model.dbf.utils import compute_K

    dbf, ck = benchmark.dbf_load(ds, args.density, dev)
    nc = dbf.lambdas.shape[0] // 2
    F = compute_K(dbf.lambdas, steps=2, num_real=0, num_complex=nc)[1].to(dev).float()
    dsamp = MaskSampler(
        cfg["image_size"],
        float(cfg["obs_ratio"]),
        cfg.get("mask_kind", "random"),
        int(cfg.get("mask_seed", 0)),
        fixed=benchmark.DBF_MASK_FIXED[ds],
    )
    with torch.no_grad():
        tru = torch.stack(
            [dbf.f_network(gt[t].reshape(NR, -1).float()) for t in range(T)], dim=1
        )
        z = tru[:, 0]
        rolls = [z]
        for _ in range(run_length):
            z = z @ F.t()
            rolls.append(z)
        bef = torch.stack(rolls, dim=1).cpu().numpy()
        obs = [gt[0].reshape(NR, -1)]
        for t in range(1, T):
            masks = torch.stack([dsamp.mask(b, t) for b in range(NR)]).to(dev)
            obs.append((gt[t] * masks).reshape(NR, -1))
        obs = torch.stack(obs, dim=1).float()
        dbf.bench_obs_every = int(cfg.get("obs_every", 1))
        _, _, mu_t, _ = dbf._compute_mu_t_sigma_t(obs, T, NR, 1, spinup=spin)
        ana = mu_t.reshape(NR, T, -1)
        s = rrmse_of(dbf.h_network, ana)
    save(
        "DBF",
        tru.cpu().numpy(),
        bef,
        ana.cpu().numpy(),
        dict(ckpt=ck, fixed_mask=benchmark.DBF_MASK_FIXED[ds], rrmse=s),
    )
print("ALL DONE", flush=True)
