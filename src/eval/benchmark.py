import argparse
import json
import os
import sys
import time

REPO = os.environ.get(
    "REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
os.chdir(REPO)
sys.path.insert(0, REPO)

import numpy as np
import torch

from src.config import load_config
from src.assimilation.variational import assimilate_runs, nmc_background
from src.assimilation.data import DAData
from src.assimilation.models import build_model
from src.assimilation.observation import MaskSampler
from src.eval.covariance_selection import dmd_sigmas

_CFG = next(
    (
        os.path.join(REPO, d)
        for d in ("config", "config")
        if os.path.exists(os.path.join(REPO, d, "eval.json"))
    ),
    os.path.join(REPO, "config"),
)
REG = json.load(open(f"{_CFG}/eval.json"))
SWEEP = json.load(open(f"{_CFG}/sweep.json"))


EVAL_SEED = str(int(os.environ.get("EVAL_SEED", "1")))
if int(EVAL_SEED) < 1:
    raise ValueError("EVAL_SEED must be positive")


def _seed_swap(obj):
    n = EVAL_SEED
    if not n or str(n) == "1":
        return obj
    if isinstance(obj, str):
        return obj.replace("/seed1/", "/seed%s/" % n)
    if isinstance(obj, list):
        return [_seed_swap(x) for x in obj]
    return obj


if EVAL_SEED != "1":
    for _ds, _mm in REG["datasets"].items():
        for _m, _v in _mm.items():
            for _k in ("ckpt", "fwd", "noise"):
                if _k in _v:
                    _v[_k] = _seed_swap(_v[_k])

CKPT = {
    (ds, m): v["ckpt"]
    for ds, mm in REG["datasets"].items()
    for m, v in mm.items()
    if m in ("nl_ae", "wl_ae", "kae")
}
DBF_MASK_FIXED = {
    ds: mm["dbf"]["flags"]["mask_fixed"] for ds, mm in REG["datasets"].items()
}
TV_SIGMA = {
    ds: {int(k): tuple(v) for k, v in mm["tensorvar"]["sigma"].items()}
    for ds, mm in REG["datasets"].items()
}


def _ck_parts(ds):
    d = os.path.dirname(REG["datasets"][ds]["dbf"]["ckpt"])
    return os.path.dirname(os.path.dirname(d)), os.path.basename(d)


CKD = lambda ds, model: os.path.join(_ck_parts(ds)[0], model, _ck_parts(ds)[1])

OUT = os.path.join(os.environ.get("EVAL_OUTPUT_ROOT", "results/eval"), f"seed{EVAL_SEED}")
DISP = {
    "nl_ae": "NL-AE",
    "wl_ae": "WL-AE",
    "kae": "KAE",
    "tensorvar": "TensorVar",
    "dbf": "DBF",
    "cgkn": "CGKN",
    "dmd": "DMD",
}
COV_SCHEMES = {
    "calibration": dict(
        calib="none",
        calib_ridge=1e-3,
        ridge=0.1,
        nseg=600,
        seglen=40,
        timecorr=0,
        stats_batch=64,
    ),
    "paired_samples": dict(qr_n=8000, qr_ridge=0.1),
    "windows": dict(qr_n=400, qr_ridge=0.1),
}

TV_DRIVER = {
    "lorenz96": dict(
        kind="feature_space",
        fwd="forward.pt",
        inv="inverse_encoder@{d}.pt",
        calib="fwd",
        calib_ridge=1e-3,
        ridge=0.01,
        nseg=600,
        seglen=40,
        timecorr=0,
    ),
    "cylinder": dict(
        kind="closed_form",
        inpaint=True,
        fwd="forward.pt",
        inv="inverse_encoder@{d}.pt",
        cov="paired_samples",
        width=96,
        n_attn=4,
    ),
    "shallow_water": dict(
        kind="closed_form",
        inpaint=True,
        fwd="forward.pt",
        inv="inverse_encoder@{d}.pt",
        cov="calibration",
        width=96,
        n_attn=4,
    ),
    "era5": dict(
        kind="closed_form",
        inpaint=True,
        fwd="forward.pt",
        inv="inverse_encoder@{d}.pt",
        cov="windows",
        width=96,
        n_attn=4,
    ),
    "era5_high": dict(
        kind="closed_form",
        inpaint=True,
        fwd="forward.pt",
        inv="inverse_encoder@{d}.pt",
        cov="paired_samples",
        width=96,
        n_attn=4,
    ),
}

for _ds, _D in TV_DRIVER.items():
    if _D["kind"] == "closed_form":
        TV_DRIVER[_ds] = {**COV_SCHEMES[_D["cov"]], **_D}

for _ds, _mm in TV_SIGMA.items():
    _D = TV_DRIVER.get(_ds)
    if _D is None:
        continue
    _D["sig"] = {int(_d): tuple(_v) for _d, _v in _mm.items()}
del _ds, _mm, _D


def sigmas(ds, model, dens):
    key = f"{ds}:{model}:obs{dens}"
    c = SWEEP["cells"].get(key)
    if c is None:
        raise KeyError(key)
    return (
        float(c["sigma_r"]),
        float(c["sigma_q"]),
        f"config/sweep.json[{key}]",
        float(c["weak_best"]),
    )


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def setup(ds, model, dens, dev):
    cfg = load_config("config/da_sweep.yaml", f"{ds}_obs{dens}")
    cfg["model"] = model
    data = DAData(cfg)
    ev = cfg["eval"]
    run_length = int(ev["length"])
    ev["starts"] = data.eval_starts(ev, int(cfg["skip"]))
    runs = data.build_runs(ev["starts"], int(cfg["skip"]), run_length)
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
    m = build_model(model, CKPT[(ds, model)], device=dev, compile=False)
    Bi = nmc_background(
        m,
        data.frame_batched,
        samples,
        int(cfg["nmc_lead"]),
        int(cfg["skip"]),
        diagonal=bool(cfg.get("b_diagonal", False)),
        ridge=float(cfg["b_ridge"]),
    ).to(dev)
    return cfg, m, runs, sampler, Bi, run_length


def _common(ds, dens, cfgfile="config/da_sweep.yaml", data_path=None):
    cfg = load_config(cfgfile, f"{ds}_obs{dens}")
    if data_path:
        cfg["da_data_path"] = data_path
    data = DAData(cfg)
    ev = cfg["eval"]
    rl = int(ev["length"])
    starts = data.eval_starts(ev, int(cfg["skip"]), val=bool(data_path))
    ev["starts"] = starts
    runs = data.build_runs(starts, int(cfg["skip"]), rl)
    return cfg, data, runs, rl


def dmd_setup(ds, dens, dev, data_path=None, *, use_selected_sigmas=True):
    from src.assimilation.dmd import DMDDA, assimilate_runs_dmd, estimate_QR_dmd

    sr, sq, source = dmd_sigmas(ds, dens) if use_selected_sigmas else (1.0, 1.0, None)
    cfg, data, runs, rl = _common(ds, dens, data_path=data_path)
    m = DMDDA(cfg, f"{ds}_obs{dens}", dev, r=0)
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
        m,
        data.frame_batched,
        samples,
        int(cfg["nmc_lead"]),
        int(cfg["skip"]),
        diagonal=bool(cfg.get("b_diagonal", False)),
        ridge=float(cfg["b_ridge"]),
    ).to(dev)
    Qi, Ri = estimate_QR_dmd(m, cfg, sampler)
    kw = dict(
        window_size=int(cfg["window_size"]),
        run_length=rl,
        obs_every=int(cfg["obs_every"]),
        constraint=cfg.get("constraint", "weak"),
        Qi=Qi,
        Ri=Ri,
    )
    return dict(
        m=m,
        Bi=Bi,
        solve=assimilate_runs_dmd,
        kw=kw,
        runs=runs,
        sampler=sampler,
        cfg=cfg,
        rl=rl,
        spin=int(cfg["spinup"]),
        sr=sr,
        sq=sq,
        sigma_source=source,
    )


def run_dmd_cell(ds, dens, dev, forecast=False):
    t = dmd_setup(ds, dens, dev, use_selected_sigmas=not forecast)
    sp = t["rl"] if forecast else t["spin"]
    arr, secs = timed(
        lambda: t["solve"](
            t["m"],
            t["runs"],
            t["sampler"],
            t["Bi"],
            t["sq"],
            t["sr"],
            spinup=sp,
            **t["kw"],
        )
    )
    return arr, secs, t["cfg"]


def dbf_load(ds, dens, dev):
    from src.model.dbf.our_ae import build_dbf_for_dataset

    ckpt = f"{CKD(ds, 'dbf')}/dbf@{dens}.pt"
    if not os.path.exists(ckpt):
        ckpt = f"{CKD(ds, 'dbf')}/dbf.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    dbf = build_dbf_for_dataset(
        f"{ds}_obs{dens}" if ds == "lorenz96" else ds,
        load_ae=False,
        compile=False,
        device=dev,
    )
    clean = lambda sd: {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    dbf.f_network.load_state_dict(clean(ck["f_network"]))
    dbf.G_network.load_state_dict(clean(ck["G_network"]))
    dbf.h_network.load_state_dict(clean(ck["h_network"]))
    dbf.lambdas = ck["lambdas"].to(dev)
    return dbf, ckpt


def run_dbf_cell(ds, dens, dev, forecast=False):
    from src.assimilation.dbf import evaluate

    cfg, data, runs, rl = _common(ds, dens, "config/da.yaml")
    dbf, _ = dbf_load(ds, dens, dev)
    dbf.bench_obs_every = int(cfg.get("obs_every", 1))
    sampler = MaskSampler(
        cfg["image_size"],
        float(cfg["obs_ratio"]),
        cfg.get("mask_kind", "random"),
        int(cfg.get("mask_seed", 0)),
        fixed=DBF_MASK_FIXED[ds],
    )
    spin = int(cfg["spinup"])
    if not forecast:
        return timed(lambda: evaluate(dbf, runs, sampler, spin, rl, dev)) + (cfg,)

    from src.assimilation.variational import _rrmse_batch

    @torch.no_grad()
    def _fc():
        B = len(runs)
        gt = [
            torch.stack([runs[b][k] for b in range(B)]).to(dev) for k in range(rl + 1)
        ]
        obs0 = gt[0].reshape(B, -1).unsqueeze(1).float()
        _, _, mu, sg = dbf._compute_mu_t_sigma_t(obs0, 1, B, 1)
        mu0, s0 = mu[:, 0], sg[:, 0]
        mus, _ = dbf.predict_msteps(rl - 1, mu0, s0)
        z = torch.cat([mu0.unsqueeze(0), mus], 0)
        dec = dbf.h_network(z.reshape(rl * B, -1)).reshape(rl, B, -1)
        return np.stack(
            [
                _rrmse_batch(dec[k].reshape(B, *gt[k].shape[1:]), gt[k]).cpu().numpy()
                for k in range(rl)
            ],
            axis=1,
        )

    return timed(_fc) + (cfg,)


def cgkn_subset(ds):
    from src.data.cgkn_eval_subsets import build

    cache = os.environ.get("CGKN_SUBSET_CACHE", ".cgkn_subsets")
    os.makedirs(cache, exist_ok=True)
    p = os.path.join(cache, f"eval_runs_{ds}_test.npy")
    if not os.path.exists(p):
        arr, meta = build(ds)
        np.save(p, arr)
        print(f"  [cgkn] built {p} from {meta['source']}", flush=True)
    return p


def cgkn_env(ds, dens):
    cfg = load_config("config/da_sweep.yaml", f"{ds}_obs{dens}")
    script = (
        "src/eval/cgkn/cgkn_filter_lorenz96.py"
        if ds == "lorenz96"
        else "src/eval/cgkn/cgkn_filter_grid.py"
    )
    env = dict(
        OBS=str(dens),
        CKPT=REG["datasets"][ds]["cgkn"]["ckpt"].format(d=dens),
        EVAL_RUNS=cgkn_subset(ds),
        SPINUP=str(cfg["spinup"]),
        OBS_EVERY=str(cfg["obs_every"]),
        RUN_LENGTH=str(cfg["eval"]["length"]),
    )
    if ds == "lorenz96":
        env["FILTER"] = "discrete"
        env["NOISE"] = REG["datasets"][ds]["cgkn"].get(
            "noise", os.path.join(os.path.dirname(env["CKPT"]), "sigma@{d}.pt")
        ).format(d=dens)
    else:
        inputs = os.environ.get("CGKN_INPUTS", "ckpt_release/cgkn_inputs")
        name = {
            "era5": "normalization_era5.json",
            "era5_high": "normalization_era5_high.json",
        }.get(ds, "normalization.json")
        env.update(DS=ds, NORMALIZATION=os.path.join(inputs, name))
    return script, env


def run_cgkn_cell(ds, dens, dev, forecast=False):
    import subprocess, tempfile

    cfg = _common(ds, dens)[0]
    rl = int(cfg["eval"]["length"])
    if forecast:
        return cgkn_forecast(ds, dens, dev, rl)
    script, env = cgkn_env(ds, dens)
    out = tempfile.mktemp(suffix=".npz")
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, script],
        env=dict(os.environ, OUT=out, **env),
        capture_output=True,
        text=True,
    )
    secs = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-1500:], r.stderr[-1500:])
    errs = np.load(out)["errs"]
    os.remove(out)
    return errs, secs, cfg


def _merge_model_record(old_rec, new_rec):
    if not isinstance(old_rec, dict) or not isinstance(new_rec, dict):
        return new_rec
    out = dict(old_rec) | dict(new_rec)
    out.pop("error", None)
    err = new_rec.get("error")
    if err is None:
        out.pop("_last_run_error", None)
    else:
        kept = [k for k in ("forecast", "da") if old_rec.get(k)]
        out["_last_run_error"] = {"error": err, "survives": kept}
        for k in kept:
            out[k] = old_rec[k]
    for sub in ("da", "da_curve"):
        o, n = old_rec.get(sub), new_rec.get(sub)
        if isinstance(o, dict) and isinstance(n, dict):
            merged = dict(o)
            for dens, nv in n.items():
                if nv is None:
                    continue
                ov = merged.get(dens)
                if not (isinstance(ov, dict) and isinstance(nv, dict)):
                    merged[dens] = nv
                    continue
                keep = dict(ov)
                if any(
                    keep.get(k) != nv.get(k)
                    for k in ("per_window_s", "total_s")
                    if k in nv
                ):
                    for d in ("per_window_s_net", "floor_pct_of_total"):
                        keep.pop(d, None)
                merged[dens] = {**keep, **nv}
            out[sub] = merged
    if new_rec.get("forecast") is None and old_rec.get("forecast") is not None:
        out["forecast"] = old_rec["forecast"]

    def _finite(f):
        return isinstance(f, dict) and (
            "diverged_at_step" in f
            or any(isinstance(v, (int, float)) and v == v for v in f.values())
        )

    if not _finite(out.get("forecast")) and _finite(old_rec.get("forecast")):
        out["forecast"] = old_rec["forecast"]
        out["forecast_from_previous_run"] = True
    return out


def _check_metadata(old_meta, new_meta, path):
    for key in ("dataset", "seed", "protocol"):
        old, new = old_meta.get(key), new_meta.get(key)
        if key == "seed":
            old, new = str(old), str(new)
        if old != new:
            raise ValueError(f"{key} mismatch in {path}; use a separate output directory")


def _merge_meta(old_meta, new_meta):
    if not isinstance(old_meta, dict) or not isinstance(new_meta, dict):
        return new_meta or old_meta
    out = dict(old_meta)
    for k, v in new_meta.items():
        if v is None or v == {} or v == []:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_meta(out[k], v)
        else:
            out[k] = v
    return out


def _write_per_model(d, sub, obj, key="models"):
    outdir = os.path.join(d, sub)
    os.makedirs(outdir, exist_ok=True)
    meta = {k: v for k, v in obj.items() if k != key}
    meta["seed"] = EVAL_SEED
    mpath = os.path.join(outdir, "_meta.json")
    if meta:
        if os.path.exists(mpath):
            with open(mpath) as f:
                old_meta = json.load(f)
            _check_metadata(old_meta, meta, mpath)
            meta = _merge_meta(old_meta, meta)
        with open(mpath, "w") as f:
            json.dump(meta, f, indent=1)
    n = 0
    for model, rec in (obj.get(key) or {}).items():
        safe = str(model).replace("/", "_")
        fp = os.path.join(outdir, f"{safe}.json")
        if os.path.exists(fp):
            with open(fp) as f:
                rec = _merge_model_record(json.load(f), rec)
        with open(fp, "w") as f:
            json.dump(rec, f, indent=1)
        n += 1
    print(f"  wrote {outdir}/  ({n} model files)", flush=True)


def _write(d, fn, obj, key="models"):
    obj = dict(obj, seed=EVAL_SEED)
    path = f"{d}/{fn}"
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        _check_metadata(old, obj, path)
        if isinstance(old.get(key), dict):
            merged = dict(old[key])
            for _m, _rec in obj[key].items():
                merged[_m] = _merge_model_record(merged.get(_m), _rec)
            obj = dict(old) | dict(obj)
            obj[key] = merged
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"  wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)", flush=True)


def tv_checkpoint_paths(ds, dens, driver=None, root=None):
    if root is not None:
        return os.path.join(root, driver["fwd"]), os.path.join(root, driver["inv"].format(d=dens))
    config = REG["datasets"][ds]["tensorvar"]
    inverse = config["ckpt"]
    if isinstance(inverse, list):
        if len(inverse) != 1:
            raise ValueError("Tensor-Var requires one inverse checkpoint template")
        inverse = inverse[0]
    return config["fwd"], inverse.format(d=dens)


def tv_build(D, root, dens, cfg, data, sampler, samples, oev, W, dev, paths=None):
    from src.assimilation import tensorvar as RT

    fwd, inv = paths if paths is not None else (
        os.path.join(root, D["fwd"]), os.path.join(root, D["inv"].format(d=dens))
    )
    if D["kind"] == "feature_space":
        from src.assimilation.tensorvar_feature import TVDA

        m = TVDA(fwd, inv, dev)
    else:
        from src.assimilation.tensorvar_inpaint import TVDAInpaint

        m = TVDAInpaint(fwd, inv, dev, width=D["width"], n_attn=D["n_attn"])
    Bi = nmc_background(
        m,
        data.frame_batched,
        samples,
        int(cfg["nmc_lead"]),
        int(cfg["skip"]),
        diagonal=bool(cfg.get("b_diagonal", False)),
        ridge=float(cfg["b_ridge"]),
    ).to(dev)
    if D["kind"] == "feature_space" or D["cov"] == "calibration":
        from src.assimilation.tensorvar_feature import collect_stats, cov_inv, fit_calib

        st = collect_stats(
            m,
            cfg,
            float(cfg["obs_ratio"]),
            cfg.get("mask_kind", "random"),
            D["nseg"],
            D["seglen"],
            D["timecorr"],
            batch=D.get("stats_batch", 512),
        )
        A, b, H = fit_calib(st["Y"], st["Z"], D["calib"], D["calib_ridge"])
        Ri, _ = cov_inv((st["Y"] @ A.t() + b) - st["Z"] @ H.t(), D["ridge"])
        Qi, _ = cov_inv(st["EQ"], D["ridge"])
        if D["kind"] == "feature_space":
            from src.assimilation.tensorvar_feature import assimilate

            Sinv = {n: torch.block_diag(*([Ri] * n)) for n in range(2, W + 2)}
            return m, Bi, assimilate, dict(Qi=Qi, Ri=Ri, A=A, b=b, H=H, Sinv_full=Sinv)
    elif D["cov"] == "paired_samples":
        from src.assimilation.covariance import build_inv, collect_QR

        EQ, ER = collect_QR(m, cfg, sampler, oev, n_samples=D["qr_n"], yshift=0, gap=0)
        Qi = build_inv(EQ, D["qr_ridge"], "full").to(dev)
        Ri = build_inv(ER, D["qr_ridge"], "full").to(dev)
    else:
        Qi, Ri = RT.estimate_QR(
            m, cfg, sampler, oev, n_windows=D["qr_n"], ridge=D["qr_ridge"]
        )
    return m, Bi, RT.assimilate_runs_tv, dict(Qi=Qi, Ri=Ri)


def tv_solve(solve, m, runs, sampler, Bi, sq, sr, return_analysis=False, **kw):
    out = solve(m, runs, sampler, Bi, sq, sr, return_analysis=return_analysis, **kw)
    if not isinstance(out, tuple):
        return out, None
    return out[0], (out[-1] if return_analysis else None)


def tv_setup(ds, dens, dev, driver=None, root=None, data_path=None):
    D = driver or TV_DRIVER[ds]
    paths = tv_checkpoint_paths(ds, dens, D, root)
    cfgfile = (
        "config/da_sweep.yaml" if D["kind"] == "feature_space" else "config/da.yaml"
    )
    cfg, data, runs, rl = _common(ds, dens, cfgfile, data_path)
    sampler = MaskSampler(
        cfg["image_size"],
        float(cfg["obs_ratio"]),
        cfg.get("mask_kind", "random"),
        int(cfg.get("mask_seed", 0)),
        fixed=bool(cfg.get("mask_fixed", False)),
    )
    samples = data.nmc_samples(
        int(cfg["nmc_lead"]),
        int(cfg["skip"]),
        int(cfg["nmc_samples"]),
        seed=int(cfg.get("mask_seed", 0)),
    )
    W, oev = int(cfg["window_size"]), int(cfg["obs_every"])
    m, Bi, solve, cov = tv_build(
        D, root, dens, cfg, data, sampler, samples, oev, W, dev, paths=paths
    )
    kw = dict(
        window_size=W,
        run_length=rl,
        obs_every=oev,
        constraint=cfg.get("constraint", "weak"),
        **cov,
    )
    sr, sq = D["sig"][dens]
    return dict(
        m=m,
        Bi=Bi,
        solve=solve,
        kw=kw,
        runs=runs,
        sampler=sampler,
        cfg=cfg,
        rl=rl,
        spin=int(cfg["spinup"]),
        sr=sr,
        sq=sq,
    )


def run_tv_cell(ds, dens, dev, forecast=False):
    t = tv_setup(ds, dens, dev)
    sp = t["rl"] if forecast else t["spin"]
    (arr, _), secs = timed(
        lambda: tv_solve(
            t["solve"],
            t["m"],
            t["runs"],
            t["sampler"],
            t["Bi"],
            t["sq"],
            t["sr"],
            spinup=sp,
            **t["kw"],
        )
    )
    return arr, secs, t["cfg"]


@torch.no_grad()
def _cgkn_patch_forecast(H, ds, dens, dev, rl, return_fields=False):
    import numpy as _np

    model = H.CGKN().to(dev)
    model.load_state_dict(torch.load(os.environ["CKPT"], map_location=dev))
    model.eval()
    norms = json.load(
        open(
            H._input(
                "NORMALIZATION",
                {
                    "era5": "normalization_era5.json",
                    "era5_high": "normalization_era5_high.json",
                }.get(H.DS, "normalization.json"),
            )
        )
    )
    m_c, s_c = [_np.asarray(v, _np.float32) for v in norms["cgkn_norm"][H.DS]]
    m_b, s_b = [_np.asarray(v, _np.float32) for v in norms["bench_norm"][H.DS]]
    raw = _np.load(H._input("EVAL_RUNS", f"eval_runs_{H.DS}.npy")).astype(_np.float32)
    if H.LAYOUT == "nhwc":
        raw = raw.transpose(0, 1, 4, 2, 3)
    mc = torch.tensor(m_c, device=dev).view(1, H.C, 1)
    sc = torch.tensor(s_c, device=dev).view(1, H.C, 1)
    mb = torch.tensor(m_b, device=dev).view(1, H.C, 1)
    sb = torch.tensor(s_b, device=dev).view(1, H.C, 1)
    Idz = torch.eye(H.DZ, device=dev)

    n = raw.shape[0]
    errs = _np.zeros((n, rl), _np.float64)
    flds = [] if return_fields else None
    lats = [] if return_fields else None
    t0 = time.perf_counter()
    for r in range(n):
        xp = torch.tensor(raw[r, : rl + 1], device=dev).reshape(rl + 1, H.C, H.H * H.W)
        x = (xp - mc) / sc
        u1, u2 = H.split_field(x)
        z = model.ae.enc(u2[:1]).reshape(H.DZ)
        u1n = u1[0].reshape(H.DU1)
        zs, u1s = [z], [u1n]
        for _ in range(rl):
            f1, Hm, f2, Am = [
                e[0] for e in model.cgn.dense(u1n.reshape(1, H.NB, H.DU1_P))
            ]
            u1n = u1n + (Hm @ z + f1)
            z = (Idz + Am) @ z + f2
            zs.append(z)
            u1s.append(u1n)
        U1 = torch.stack(u1s).reshape(rl + 1, *u1.shape[1:])
        full = H.assemble_field(
            U1, model.ae.dec(torch.stack(zs).reshape(-1, H.NB, H.DZU), U1)
        )
        pb = ((full * sc + mc - mb) / sb).reshape(-1, H.C, H.H, H.W)
        gb = ((xp - mb) / sb).reshape(-1, H.C, H.H, H.W)
        num = ((pb - gb) ** 2).mean(dim=(1, 2, 3))
        den = (gb**2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
        errs[r] = torch.sqrt(num / den).cpu().numpy()[:rl]
        if flds is not None:
            flds.append(pb[:rl].detach().cpu().numpy())
            lats.append(torch.stack(zs).reshape(rl + 1, -1).detach().cpu().numpy())
        del xp, x, u1, u2, zs, u1s, U1, full, pb, gb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    secs = time.perf_counter() - t0
    cfg, _, _, _ = _common(ds, dens)
    if return_fields:
        return errs, secs, cfg, _np.stack(flds), _np.stack(lats)
    return errs, secs, cfg


@torch.no_grad()
def cgkn_forecast(ds, dens, dev, rl, return_fields=False):
    import importlib.util

    script, env = cgkn_env(ds, dens)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("cgkn_h", script)
    H = importlib.util.module_from_spec(spec)
    sys.modules["cgkn_h"] = H
    spec.loader.exec_module(H)

    if ds != "lorenz96":
        return _cgkn_patch_forecast(H, ds, dens, dev, rl, return_fields)
    ae = H.AutoEncoder(H.BW, H.dim_z_unit)
    cgn = H.CGN()
    m = H.CGKN(ae, cgn).to(dev)
    m.load_state_dict(torch.load(env["CKPT"], map_location=dev))
    m.eval()
    a = np.load(H._input("EVAL_RUNS", "eval_runs_lorenz96.npy")).astype(np.float32)
    u_all = torch.tensor(a.reshape(a.shape[0], a.shape[1], H.I), device=dev)
    norms = json.load(open(H._input("NORMALIZATION", "normalization.json")))
    mb, sb = [float(v[0]) for v in norms["bench_norm"]["lorenz96"]]
    RL, DT = 40, 0.05
    idx = H.indices_u1

    K = u_all.shape[0]
    t0 = time.perf_counter()
    z = H.encode_u2(ae, u_all[:, :1]).reshape(K, H.dim_z, 1)
    u1 = u_all[:, 0][:, idx].reshape(K, -1, 1)
    zs, u1s = [z], [u1]
    for _ in range(RL):
        f1, g1, f2, g2 = cgn(u1.squeeze(-1))
        u1 = u1 + (g1 @ z + f1) * DT
        z = z + (g2 @ z + f2) * DT
        zs.append(z)
        u1s.append(u1)
    Z = torch.stack(zs, 1).squeeze(-1)
    U1 = torch.stack(u1s, 1).squeeze(-1)
    full = H.assemble(U1, H.decode_blocks(ae, Z))
    gt = u_all[:, : RL + 1]
    pb = ((full - mb) / sb).reshape(K, RL + 1, 1, 8, 5)
    gb = ((gt - mb) / sb).reshape(K, RL + 1, 1, 8, 5)
    num = ((pb - gb) ** 2).mean(dim=(2, 3, 4))
    den = (gb**2).mean(dim=(2, 3, 4)).clamp(min=1e-12)
    errs = torch.sqrt(num / den)[:, :RL].cpu().numpy()
    secs = time.perf_counter() - t0
    cfg, _, _, _ = _common(ds, dens)
    if return_fields:
        return (
            errs,
            secs,
            cfg,
            pb[:, :RL].detach().cpu().numpy(),
            Z.detach().cpu().numpy(),
        )
    return errs, secs, cfg


def main(ds, models, densities, gpu, tol):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"dataset": ds, "models": {}}
    wall = {"dataset": ds, "models": {}}
    curves = {"dataset": ds, "models": {}}
    proto = None

    KNOWN = set(DISP) | {"nl_ae", "wl_ae", "kae", "tensorvar", "dbf", "cgkn", "dmd"}
    unknown = [m for m in models if m not in KNOWN]
    if unknown:
        raise SystemExit(str(unknown))

    for model in models:
        name = DISP.get(model, model)
        print(f"\n===== {ds} : {name} =====", flush=True)
        r = {"forecast": None, "da": {}}
        w = {"forecast": None, "da": {}}
        c = {"forecast": None, "da": {}}
        variational = model in ("nl_ae", "wl_ae", "kae")
        fc_s = None

        try:
            d0 = densities[0]
            if variational:
                cfg, m, runs, sampler, Bi, run_length = setup(ds, model, d0, dev)
                spin = int(cfg["spinup"])
                fc, fc_s = timed(
                    lambda: assimilate_runs(
                        m,
                        runs,
                        sampler,
                        Bi,
                        1.0,
                        1.0,
                        window_size=int(cfg["window_size"]),
                        spinup=run_length,
                        run_length=run_length,
                        obs_every=int(cfg["obs_every"]),
                        iters=int(cfg["iters"]),
                        lr=float(cfg["lr"]),
                        constraint="weak",
                        converged=True,
                    )
                )
            else:
                fn = {
                    "dmd": run_dmd_cell,
                    "tensorvar": run_tv_cell,
                    "dbf": run_dbf_cell,
                    "cgkn": run_cgkn_cell,
                }[model]
                fc, fc_s, cfg = fn(ds, d0, dev, forecast=True)
                run_length, spin = int(cfg["eval"]["length"]), int(cfg["spinup"])
            nr = fc.shape[0]
            r["forecast"] = {
                "rrmse": float(fc.mean()),
                "rrmse_post_spinup": float(fc[:, spin:].mean()),
                "rrmse_final_step": float(fc[:, -1].mean()),
            }
            _bad = ~np.isfinite(fc).all(axis=0)
            if _bad.any():
                r["forecast"]["diverged_at_step"] = int(np.argmax(_bad))
            w["forecast"] = {
                "total_s": round(fc_s, 4),
                "total_ms": round(fc_s * 1e3, 1),
                "per_run_ms": round(fc_s / nr * 1e3, 3),
                "per_step_ms": round(fc_s / (nr * run_length) * 1e3, 4),
            }
            c["forecast"] = fc.round(6).tolist()
            print(
                f"\n  forecast          rrmse {r['forecast']['rrmse']:.5f} "
                f"(post-spinup {r['forecast']['rrmse_post_spinup']:.5f})  "
                f"{fc_s:.2f}s",
                flush=True,
            )

            if proto is None:
                proto = {
                    "starts": [list(s) for s in cfg["eval"]["starts"]],
                    "n_test_runs": len(cfg["eval"]["starts"]),
                    "run_length": run_length,
                    "spinup": spin,
                    "window_size": int(cfg["window_size"]),
                    "obs_every": int(cfg["obs_every"]),
                    "skip": int(cfg["skip"]),
                    "image_size": cfg["image_size"],
                    "data": cfg.get("da_data_path")
                    or cfg.get("val_path")
                    or cfg["data_path"],
                }
                res["protocol"] = wall["protocol"] = curves["protocol"] = proto

            for dens in densities:
                sr = sq = src = ref = None
                if variational:
                    if dens != densities[0]:
                        cfg, m, runs, sampler, Bi, run_length = setup(
                            ds, model, dens, dev
                        )
                    sr, sq, src, ref = sigmas(ds, model, dens)
                    da, da_s = timed(
                        lambda: assimilate_runs(
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
                        )
                    )
                elif model == "dmd":
                    da, da_s, cfg = run_dmd_cell(ds, dens, dev)
                elif model == "dbf":
                    da, da_s, cfg = run_dbf_cell(ds, dens, dev)
                elif model == "cgkn":
                    da, da_s, cfg = run_cgkn_cell(ds, dens, dev)
                elif model == "tensorvar":
                    da, da_s, cfg = run_tv_cell(ds, dens, dev)
                else:
                    raise NotImplementedError(model)

                got = float(da[:, spin:].mean())
                nr = da.shape[0]
                cell = {"rrmse": got}
                if ref is not None:
                    cell.update(
                        resweep_weak_best=ref,
                        deviation_pct=round((got - ref) / ref * 100, 3),
                        within_tol=bool(abs(got - ref) / ref <= tol),
                        sigma_r=sr,
                        sigma_q=sq,
                        sigma_source=src,
                    )
                if model == "dmd":
                    sr, sq, source = dmd_sigmas(ds, dens)
                    cell.update(sigma_r=sr, sigma_q=sq, sigma_source=source)
                r["da"][f"obs{dens}"] = cell
                nwin = int(np.ceil((run_length - spin) / int(cfg["window_size"])))
                wcell = {
                    "total_s": round(da_s, 3),
                    "per_run_s": round(da_s / nr, 4),
                    "per_window_s": round(da_s / (nr * nwin), 4),
                    "n_windows_per_run": nwin,
                }
                if fc_s:
                    wcell["cost_vs_forecast_x"] = round(da_s / fc_s, 1)
                if model == "cgkn":
                    wcell["subprocess"] = True
                w["da"][f"obs{dens}"] = wcell
                c["da"][f"obs{dens}"] = da.round(6).tolist()
                dv = ""
                print(
                    f"\n  DA obs{dens:<3d}         rrmse {got:.5f}{dv}  {da_s:.1f}s",
                    flush=True,
                )

        except Exception as e:
            import traceback

            print(
                f"  [{name}] FAILED: {e}\n{traceback.format_exc()[-1200:]}", flush=True
            )
            r["error"] = str(e)

        res["models"][name], wall["models"][name], curves["models"][name] = r, w, c
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    d = f"{OUT}/{ds}"
    os.makedirs(d, exist_ok=True)
    _write_per_model(d, "result", res)
    _write_per_model(d, "wall_time", wall)
    _write_per_model(d, "curve", curves)
    failed = [name for name, record in res["models"].items() if record.get("error")]
    if failed:
        raise SystemExit(f"Evaluation failed for: {failed}")


def _timed_sync(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _per_traj(fn, n, warmup=1):
    for _ in range(warmup):
        fn(0)
    ts = [_timed_sync(lambda i=i: fn(i)) for i in range(n)]
    return float(np.mean(ts)), float(np.std(ts)), ts


@torch.no_grad()
def _latent_fc(m, run0, rl, dev):
    z = m.encode(run0[0].unsqueeze(0).to(dev))
    for _ in range(rl):
        z = m.step(z)
        m.decode(z)


def speed_suite(ds, models, dev, tol):
    cfg = load_config("config/da.yaml", f"{ds}_obs30")
    data = DAData(cfg)
    ev = cfg["eval"]
    rl = int(ev["length"])
    runs = data.build_runs(data.eval_starts(ev, int(cfg["skip"])), int(cfg["skip"]), rl)
    NR = len(runs)
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
    spin, oev, wsz = int(cfg["spinup"]), int(cfg["obs_every"]), int(cfg["window_size"])
    out = {
        "dataset": ds,
        "protocol": {
            "batch": 1,
            "warmup_traj": 1,
            "n_traj": NR,
            "run_length": rl,
            "obs_ratio": float(cfg["obs_ratio"]),
            "obs_every": oev,
            "window": wsz,
            "spinup": spin,
        },
        "models": {},
    }

    _stamp = dict(
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        n_traj=NR,
    )

    def rec(name, fc, da, **extra):
        row = dict(
            forecast=fc[0], forecast_sd=fc[1], inference=da[0], inference_sd=da[1]
        )
        row.update(_stamp)
        row.update(extra)
        out["models"][name] = row
        print(
            f"  {name:10s} forecast {fc[0]*1e3:8.1f}+-{fc[1]*1e3:5.1f} ms   "
            f"inference {da[0]*1e3:9.1f}+-{da[1]*1e3:6.1f} ms",
            flush=True,
        )

    for model in models:
        name = DISP.get(model, model)
        try:
            if model in ("nl_ae", "wl_ae", "kae"):
                m = build_model(model, CKPT[(ds, model)], device=dev, compile=False)
                Bi = nmc_background(
                    m,
                    data.frame_batched,
                    samples,
                    int(cfg["nmc_lead"]),
                    int(cfg["skip"]),
                    diagonal=bool(cfg.get("b_diagonal", False)),
                    ridge=float(cfg["b_ridge"]),
                ).to(dev)
                sr, sq, src, _ = sigmas(ds, model, 30)
                fc = _per_traj(lambda i: _latent_fc(m, runs[i], rl, dev), NR)
                da = _per_traj(
                    lambda i: assimilate_runs(
                        m,
                        [runs[i]],
                        sampler,
                        Bi,
                        sq,
                        sr,
                        window_size=wsz,
                        spinup=spin,
                        run_length=rl,
                        obs_every=oev,
                        iters=int(cfg["iters"]),
                        lr=float(cfg["lr"]),
                        constraint="weak",
                        converged=True,
                    ),
                    NR,
                )
                rec(
                    name,
                    fc,
                    da,
                    sigma_r=sr,
                    sigma_q=sq,
                    sigma_source=src,
                    ckpt=CKPT[(ds, model)],
                )
                del m
            elif model == "dmd":
                from src.assimilation.dmd import (
                    DMDDA,
                    assimilate_runs_dmd,
                    estimate_QR_dmd,
                )

                sr, sq, source = dmd_sigmas(ds, 30)
                m = DMDDA(cfg, f"{ds}_obs30", dev, r=0)
                Bi = nmc_background(
                    m,
                    data.frame_batched,
                    samples,
                    int(cfg["nmc_lead"]),
                    int(cfg["skip"]),
                    diagonal=bool(cfg.get("b_diagonal", False)),
                    ridge=float(cfg["b_ridge"]),
                ).to(dev)
                Qi, Ri = estimate_QR_dmd(m, cfg, sampler)
                fc = _per_traj(lambda i: _latent_fc(m, runs[i], rl, dev), NR)
                da = _per_traj(
                    lambda i: assimilate_runs_dmd(
                        m,
                        [runs[i]],
                        sampler,
                        Bi,
                        sq,
                        sr,
                        window_size=wsz,
                        spinup=spin,
                        run_length=rl,
                        obs_every=oev,
                        constraint=cfg.get("constraint", "weak"),
                        Qi=Qi,
                        Ri=Ri,
                    ),
                    NR,
                )
                rec(name, fc, da, sigma_r=sr, sigma_q=sq, sigma_source=source)
                del m
            elif model == "tensorvar":
                fc, da, extra = _tv_speed(
                    ds, cfg, data, runs, sampler, samples, dev, NR, rl, spin, oev, wsz
                )
                rec(name, fc, da, **extra)
            elif model == "dbf":
                from src.assimilation.dbf import evaluate as dbf_eval

                dbf, ckpt = dbf_load(ds, 30, dev)

                def _dbf(i, oe):
                    dbf.bench_obs_every = oe
                    dbf_eval(dbf, [runs[i]], sampler, spin, rl, dev)

                fc = _per_traj(lambda i: _dbf(i, 10**6), NR)
                da = _per_traj(lambda i: _dbf(i, oev), NR)
                rec(name, fc, da, ckpt=ckpt)
                del dbf
            elif model == "cgkn":
                fc, da, extra = _cgkn_speed(ds, dev, NR, rl)
                rec(name, fc, da, **extra)
            else:
                raise NotImplementedError(model)
        except Exception as e:
            import traceback

            print(
                f"  {name:10s} FAILED: {e}\n{traceback.format_exc()[-900:]}", flush=True
            )
            out["models"][name] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    d = f"{OUT}/{ds}"
    os.makedirs(d, exist_ok=True)
    _write(d, "speed.json", out)
    failed = [name for name, record in out["models"].items() if record.get("error")]
    if failed:
        raise SystemExit(f"Speed evaluation failed for: {failed}")


@torch.no_grad()
def _cgkn_speed(ds, dev, NR, rl):
    import importlib.util

    script, env = cgkn_env(ds, 30)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("cgkn_s", script)
    H = importlib.util.module_from_spec(spec)
    sys.modules["cgkn_s"] = H
    spec.loader.exec_module(H)
    if ds != "lorenz96":
        return _cgkn_speed_patch(H, dev, rl)
    ae = H.AutoEncoder(H.BW, H.dim_z_unit)
    cgn = H.CGN()
    m = H.CGKN(ae, cgn).to(dev)
    m.load_state_dict(torch.load(os.environ["CKPT"], map_location=dev))
    m.eval()
    sigma = torch.load(
        H._input("NOISE", f"sigma@{os.environ['OBS']}.pt"), map_location=dev
    )
    a = np.load(H._input("EVAL_RUNS", "eval_runs_lorenz96.npy")).astype(np.float32)
    u_all = torch.tensor(a.reshape(a.shape[0], a.shape[1], H.I), device=dev)
    idx, DT = H.indices_u1, 0.05
    R0 = 0.01 * torch.eye(H.dim_z, device=dev)

    def _da(i):
        u = u_all[i]
        mu0 = H.encode_u2(ae, u[:1]).reshape(H.dim_z, 1)
        mu, u1_used = H.CGFilter(
            m, sigma, u[:, idx].unsqueeze(-1), mu0, R0, DT,
            spinup=H.SPINUP, return_observed=True,
        )
        sel = torch.arange(0, rl + 1, device=dev)
        H.assemble(u1_used.squeeze(-1)[sel], H.decode_blocks(ae, mu.squeeze(-1)[sel]))

    def _fc(i):
        u = u_all[i]
        z = H.encode_u2(ae, u[:1]).reshape(H.dim_z, 1)
        u1 = u[0, idx].reshape(-1, 1)
        zs, u1s = [z], [u1]
        for _ in range(rl):
            f1, g1, f2, g2 = [e.squeeze(0) for e in cgn(u1.T)]
            u1 = u1 + (g1 @ z + f1) * DT
            z = z + (g2 @ z + f2) * DT
            zs.append(z)
            u1s.append(u1)
        H.assemble(
            torch.stack(u1s).squeeze(-1),
            H.decode_blocks(ae, torch.stack(zs).squeeze(-1)),
        )

    n = min(NR, u_all.shape[0])
    return (
        _per_traj(_fc, n),
        _per_traj(_da, n),
        {"ckpt": os.environ["CKPT"], "n_traj": n},
    )


def _tv_speed(ds, cfg, data, runs, sampler, samples, dev, NR, rl, spin, oev, wsz):
    D = TV_DRIVER[ds]
    paths = tv_checkpoint_paths(ds, 30)
    sr, sq = TV_SIGMA[ds][30]
    m, Bi, solve, cov = tv_build(
        D, None, 30, cfg, data, sampler, samples, oev, wsz, dev, paths=paths
    )
    kw = dict(
        window_size=wsz,
        spinup=spin,
        run_length=rl,
        obs_every=oev,
        constraint=cfg.get("constraint", "weak"),
        **cov,
    )
    fc = _per_traj(lambda i: _latent_fc(m, runs[i], rl, dev), NR)
    da = _per_traj(lambda i: solve(m, [runs[i]], sampler, Bi, sq, sr, **kw), NR)
    return (
        fc,
        da,
        {
            "driver": solve.__module__,
            "ckpt": paths[1],
        },
    )


@torch.no_grad()
def _cgkn_speed_patch(H, dev, rl):
    import numpy as _np

    model = H.CGKN().to(dev)
    model.load_state_dict(torch.load(os.environ["CKPT"], map_location=dev))
    model.eval()
    norms = json.load(
        open(
            H._input(
                "NORMALIZATION",
                {
                    "era5": "normalization_era5.json",
                    "era5_high": "normalization_era5_high.json",
                }.get(H.DS, "normalization.json"),
            )
        )
    )
    m_c, s_c = [_np.asarray(v, _np.float32) for v in norms["cgkn_norm"][H.DS]]
    raw = _np.load(H._input("EVAL_RUNS", f"eval_runs_{H.DS}.npy")).astype(_np.float32)
    if H.LAYOUT == "nhwc":
        raw = raw.transpose(0, 1, 4, 2, 3)
    mc = torch.tensor(m_c, device=dev).view(1, H.C, 1)
    sc = torch.tensor(s_c, device=dev).view(1, H.C, 1)

    def _da(r):
        xp = torch.tensor(raw[r, : rl + 1], device=dev).reshape(rl + 1, H.C, H.H * H.W)
        u1, u2 = H.split_field((xp - mc) / sc)
        u1f = u1.reshape(rl + 1, H.DU1)
        mu0 = model.ae.enc(u2[:1]).reshape(H.DZ)
        mu, u1f_eff = H.kalman_window(
            model, u1f, s1sq=0.05, s2sq=1.0, mu0=mu0,
            spinup=H.SPINUP, return_observed=True,
        )
        u1_eff = u1f_eff.reshape_as(u1)
        H.assemble_field(u1_eff, model.ae.dec(mu.reshape(-1, H.NB, H.DZU), u1_eff))

    Idz = torch.eye(H.DZ, device=dev)

    def _fc(r):
        xp = torch.tensor(raw[r, : rl + 1], device=dev).reshape(rl + 1, H.C, H.H * H.W)
        u1, u2 = H.split_field((xp - mc) / sc)
        z = model.ae.enc(u2[:1]).reshape(H.DZ)
        u1n = u1[0].reshape(H.DU1)
        zs, u1s = [z], [u1n]
        for _ in range(rl):
            f1, Hm, f2, Am = [
                e[0] for e in model.cgn.dense(u1n.reshape(1, H.NB, H.DU1_P))
            ]
            u1n = u1n + (Hm @ z + f1)
            z = (Idz + Am) @ z + f2
            zs.append(z)
            u1s.append(u1n)
        U1 = torch.stack(u1s).reshape(rl + 1, *u1.shape[1:])
        H.assemble_field(U1, model.ae.dec(torch.stack(zs).reshape(-1, H.NB, H.DZU), U1))

    n = raw.shape[0]
    return (
        _per_traj(_fc, n),
        _per_traj(_da, n),
        {"ckpt": os.environ["CKPT"], "n_traj": n},
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", default="nl_ae,wl_ae,kae,tensorvar,dbf,cgkn,dmd")
    ap.add_argument(
        "--densities", default=",".join(str(d) for d in sorted(REG["densities"]))
    )
    ap.add_argument("--gpu", default="0")
    ap.add_argument(
        "--speed",
        action="store_true",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=0.005,
    )
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    if a.speed:
        import torch as _t

        speed_suite(
            a.dataset,
            a.models.split(","),
            _t.device("cuda" if _t.cuda.is_available() else "cpu"),
            a.tol,
        )
    else:
        main(
            a.dataset,
            a.models.split(","),
            [int(x) for x in a.densities.split(",")],
            a.gpu,
            a.tol,
        )
