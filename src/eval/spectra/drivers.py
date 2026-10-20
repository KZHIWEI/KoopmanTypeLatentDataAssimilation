from __future__ import annotations

import os

import numpy as np
import torch


def _gt_stack(runs, rl):
    B = len(runs)
    return np.stack(
        [torch.stack([runs[b][t] for b in range(B)]).numpy() for t in range(rl)], axis=1
    )


def _decode_all(model, lat, dev, batch=32):
    lat = torch.as_tensor(lat)
    B, T = lat.shape[0], lat.shape[1]
    out = []
    with torch.no_grad():
        flat = lat.reshape(-1, lat.shape[-1]).to(dev)
        for i in range(0, flat.shape[0], batch):
            out.append(model.decode(flat[i : i + batch]).float().cpu().numpy())
    f = np.concatenate(out, 0)
    return f.reshape(B, T, *f.shape[1:])


def _variational(ds, mk, dens, dev, forecast):
    from src.eval.benchmark import setup, sigmas
    from src.assimilation.variational import assimilate_runs

    cfg, m, runs, sampler, Bi, rl = setup(ds, mk, dens, dev)
    sr, sq, shard, _ = sigmas(ds, mk, dens)
    if forecast:
        sr = sq = 1.0
    sp = rl if forecast else int(cfg["spinup"])
    _, lat = assimilate_runs(
        m,
        runs,
        sampler,
        Bi,
        sq,
        sr,
        window_size=int(cfg["window_size"]),
        spinup=sp,
        run_length=rl,
        obs_every=int(cfg["obs_every"]),
        iters=int(cfg["iters"]),
        lr=float(cfg["lr"]),
        constraint="weak",
        converged=True,
        return_analysis=True,
    )
    return (
        _decode_all(m, lat[:, :rl], dev),
        _gt_stack(runs, rl),
        {"sigma": [sr, sq], "sigma_source": shard, "spinup": int(cfg["spinup"])},
    )


def _tensorvar(ds, mk, dens, dev, forecast):
    from src.eval.benchmark import TV_DRIVER, tv_setup, tv_solve

    t = tv_setup(ds, dens, dev)
    rl = t["rl"]
    _, lat = tv_solve(
        t["solve"],
        t["m"],
        t["runs"],
        t["sampler"],
        t["Bi"],
        t["sq"],
        t["sr"],
        return_analysis=True,
        spinup=rl if forecast else t["spin"],
        **t["kw"],
    )
    meta = {
        "tv_kind": TV_DRIVER[ds]["kind"],
        "sigma": [t["sr"], t["sq"]],
        "spinup": t["spin"],
    }
    return _decode_all(t["m"], lat[:, :rl], dev), _gt_stack(t["runs"], rl), meta


def _dmd(ds, mk, dens, dev, forecast):
    from src.eval.benchmark import dmd_setup

    t = dmd_setup(ds, dens, dev)
    m, runs, cfg, rl = t["m"], t["runs"], t["cfg"], t["rl"]
    _, lat = t["solve"](
        m,
        runs,
        t["sampler"],
        t["Bi"],
        t["sq"],
        t["sr"],
        spinup=rl if forecast else t["spin"],
        return_analysis=True,
        **t["kw"],
    )
    return (
        _decode_all(m, lat[:, :rl], dev),
        _gt_stack(runs, rl),
        {
            "sigma": [0.05, 1.0],
            "spinup": int(cfg["spinup"]),
        },
    )


def _dbf(ds, mk, dens, dev, forecast):
    from src.eval.benchmark import DBF_MASK_FIXED, _common, dbf_load
    from src.assimilation.observation import MaskSampler
    from src.assimilation.dbf import evaluate

    cfg, data, runs, rl = _common(ds, dens, "config/da.yaml")
    dbf, ckpt = dbf_load(ds, dens, dev)
    dbf.bench_obs_every = int(cfg.get("obs_every", 1))
    sampler = MaskSampler(
        cfg["image_size"],
        float(cfg["obs_ratio"]),
        cfg.get("mask_kind", "random"),
        int(cfg.get("mask_seed", 0)),
        fixed=DBF_MASK_FIXED[ds],
    )
    spin = int(cfg["spinup"])
    meta = {
        "mask_fixed": bool(DBF_MASK_FIXED[ds]),
        "obs_every": dbf.bench_obs_every,
        "spinup": spin,
    }
    if not forecast:
        _, fld = evaluate(dbf, runs, sampler, spin, rl, dev, return_fields=True)
        return fld, _gt_stack(runs, rl), meta

    B = len(runs)
    with torch.no_grad():
        gt0 = torch.stack([runs[b][0] for b in range(B)]).to(dev)
        obs0 = gt0.reshape(B, -1).unsqueeze(1).float()
        _, _, mu, sg = dbf._compute_mu_t_sigma_t(obs0, 1, B, 1)
        mu0, s0 = mu[:, 0], sg[:, 0]
        mus, _ = dbf.predict_msteps(rl - 1, mu0, s0)
        z = torch.cat([mu0.unsqueeze(0), mus], 0)
        dec = dbf.h_network(z.reshape(rl * B, -1)).reshape(rl, B, *gt0.shape[1:])
    return (
        dec.permute(1, 0, *range(2, dec.dim())).float().cpu().numpy(),
        _gt_stack(runs, rl),
        meta,
    )


def _cgkn_to_common_units(fld, ds, cfg, norms=None):
    import json as _json
    from src.data import channel_norm

    src = norms or os.path.join(
        os.environ.get("CGKN_INPUTS", "ckpt_release/cgkn_inputs"), "normalization.json"
    )
    bn = _json.load(open(src))["bench_norm"][ds]
    if "norm_mean" not in cfg:
        return fld, None
    mb, sb = (np.asarray(v, np.float64).reshape(1, 1, -1, 1, 1) for v in bn)
    mc, sc = (
        np.asarray(v, np.float64).reshape(1, 1, -1, 1, 1)
        for v in channel_norm(np.zeros((1, 1, len(bn[0]), 1, 1), np.float32), cfg)
    )
    if np.allclose(mb, mc, rtol=1e-6, atol=1e-6) and np.allclose(sb, sc, rtol=1e-6):
        return fld, None
    out = ((np.asarray(fld, np.float64) * sb + mb) - mc) / sc
    return out.astype(np.asarray(fld).dtype), {
        "max_mean_gap": float(np.abs(mb - mc).max()),
        "max_std_ratio": float(np.abs(sb / sc - 1).max()),
    }


def _cgkn(ds, mk, dens, dev, forecast):
    import subprocess, sys, tempfile
    from src.eval.benchmark import _common, cgkn_env

    cfg, data, runs, rl = _common(ds, dens)
    script, env = cgkn_env(ds, dens)
    out = tempfile.mktemp(suffix=".npz")
    env = dict(os.environ, OUT=out, FIELDS=str(len(runs)), **env)
    if forecast:
        from src.eval.benchmark import cgkn_forecast

        _errs, _secs, _cfg, fld, _lat = cgkn_forecast(
            ds, dens, dev, rl, return_fields=True
        )
        os.path.exists(out) and os.remove(out)
        gt = _gt_stack(runs, rl)
        fld, conv = _cgkn_to_common_units(
            fld[:, :rl], ds, cfg, env.get("NORMALIZATION")
        )
        return (
            fld,
            gt[: fld.shape[0]],
            {
                "spinup": int(cfg["spinup"]),
                "n_runs_cgkn": int(fld.shape[0]),
                "n_runs_config": len(runs),
                "forecast": True,
                "units_converted": conv,
            },
        )
    r = subprocess.run(
        [sys.executable, script], env=env, capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-1200:], r.stderr[-1200:])
    z = np.load(out)
    if "fields" not in z:
        os.remove(out)
        raise RuntimeError(out)
    fld = z["fields"][:, :rl]
    os.remove(out)
    gt = _gt_stack(runs, rl)
    fld, conv = _cgkn_to_common_units(fld, ds, cfg, env.get("NORMALIZATION"))
    return (
        fld,
        gt[: fld.shape[0]],
        {
            "spinup": int(cfg["spinup"]),
            "n_runs_cgkn": int(fld.shape[0]),
            "n_runs_config": len(runs),
            "units_converted": conv,
        },
    )


FAMILY = {
    "nl_ae": _variational,
    "wl_ae": _variational,
    "kae": _variational,
    "tensorvar": _tensorvar,
    "dmd": _dmd,
    "dbf": _dbf,
    "cgkn": _cgkn,
}


def analysis_fields(ds, model_key, dens, dev, forecast=False):
    mk = model_key.lower().replace("-", "_")
    if mk not in FAMILY:
        raise KeyError(model_key)
    f, gt, meta = FAMILY[mk](ds, mk, dens, dev, forecast)
    f, gt = np.asarray(f, np.float32), np.asarray(gt, np.float32)
    if f.shape != gt.shape:
        raise RuntimeError(f.shape, gt.shape)
    meta.update({"model": model_key, "density": dens, "forecast": bool(forecast)})
    return f, gt, meta
