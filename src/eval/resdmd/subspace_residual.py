from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re

import numpy as np
import torch

from src.eval.resdmd import resdmd

_REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
CACHE = os.path.join(_REPO, "results", "latent", "cache")
EVALJSON = os.path.join(_REPO, "config", "eval.json")
OUT = "results/resdmd/residual_ratio.json"
EPS = (0.05, 0.1, 0.2, 0.3, 0.5)

GATE = 1e-5
KHAT_PAIRS = 8000

FITTED = ("NL-AE", "WL-AE")
CROSSFIT = ("CGKN",)

CFG_KEY = {
    "lorenz96": "lorenz96_obs30",
    "cylinder": "cylinder_obs30",
    "shallow_water": "shallow_water_obs30",
    "era5": "era5_obs30",
    "era5_high": "era5_high_obs30",
}
NTRAJ = {
    ("lorenz96", None): 16,
    ("cylinder", None): 5,
    ("shallow_water", None): 168,
    ("era5", None): 1,
}


def _patch_geometry(ds, dens):
    from src.eval.benchmark import cgkn_env

    _, env = cgkn_env(ds, dens)
    os.environ.update(env, OUT=os.devnull, DEVICE="cpu")
    spec = importlib.util.spec_from_file_location(
        "_cgkn_patch", os.path.join(_REPO, "src/eval/cgkn/cgkn_filter_grid.py")
    )
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return np.asarray(m.NBR), m.NB, m.DZU


def legs(path, ds, dens, model):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    out = {k: np.asarray(d[k], np.float64) for k in ("tru", "bef", "ana")}
    R0 = out["tru"].shape[0]
    if model == "CGKN" and ds != "lorenz96":
        NBR, NB, DZU = _patch_geometry(ds, dens)
        for k in ("tru", "ana"):
            Z = out[k]
            Z = Z.reshape(Z.shape[0], Z.shape[1], NB, DZU)[
                :, :, NBR.reshape(NB, 9), :
            ].reshape(Z.shape[0], Z.shape[1], NB, 9 * DZU)
            out[k] = Z.transpose(0, 2, 1, 3).reshape(-1, Z.shape[1], Z.shape[3])
        out["bef"] = np.zeros(0)
        meta["dictionary"] = "patchwise 9-neighbourhood"
    else:
        meta["dictionary"] = "stacked" if model == "CGKN" else "latent"
    return out, meta, R0


def _pairs(Z):
    D = Z.shape[-1]
    return Z[:, :-1].reshape(-1, D).copy(), Z[:, 1:].reshape(-1, D).copy()


def _finite_pairs(Z):
    D = Z.shape[-1]
    xs, ys, dropped = [], [], 0
    for r in Z:
        ok = np.isfinite(r).all(axis=1)
        n = len(r) if ok.all() else int(np.argmin(ok))
        dropped += len(r) - n
        if n >= 2:
            xs.append(r[: n - 1])
            ys.append(r[1:n])
    if not xs:
        return np.zeros((0, D)), np.zeros((0, D)), dropped
    return np.concatenate(xs).copy(), np.concatenate(ys).copy(), dropped


def _eval_entry(ds, model):
    return json.load(open(EVALJSON))["datasets"][ds][model.lower().replace("-", "_")]


_KHAT_MEMO = {}


TRAIN_FIT_RATIO = 2
DMD_POD_PAIRS = 6000


def _lazy_trajs(path):
    from src.data import _abspath, _load_trajs
    from src.assimilation.dmd import _LazyH5

    p = _abspath(path)
    if not p.endswith(".h5"):
        return (_load_trajs(path), lambda: None)
    v = _LazyH5(p)
    return (v, v._f.close)


def _train_encoder(ds, model, dens, dev):
    import torch
    from src.config import load_config

    e = _eval_entry(ds, model)
    if model == "DMD":
        from src.config import build_ae_config
        from src.data import channel_norm

        cfg = load_config("config/da.yaml", CFG_KEY[ds])
        r = build_ae_config(load_config("config/kae.yaml", CFG_KEY[ds])).latent_dim
        tr, closer = _lazy_trajs(cfg["data_path"])
        r = min(r, int(np.prod(tr.shape[2:])))
        mean, std = channel_norm(np.asarray(tr[:, :: max(1, tr.shape[1] // 200)]), cfg)
        ntraj, T = (tr.shape[0], tr.shape[1])
        skip = int(cfg.get("skip", 1))
        per = max(2, DMD_POD_PAIRS // ntraj)
        ts = np.linspace(0, T - skip - 1, per).astype(int)
        cols = []
        for i in range(ntraj):
            for k in range(0, len(ts), 128):
                j = ts[k : k + 128]
                A = (
                    np.asarray(tr[i, j], np.float32) - mean.reshape(1, -1, 1, 1)
                ) / std.reshape(1, -1, 1, 1)
                cols.append(A.reshape(A.shape[0], -1).T)
        closer()
        X = np.concatenate(cols, axis=1)
        del cols
        G = np.zeros((X.shape[1], X.shape[1]))
        for k in range(0, X.shape[0], 4096):
            B = X[k : k + 4096].astype(np.float64)
            G += B.T @ B
        evals, V = np.linalg.eigh(G)
        idx = np.argsort(evals)[::-1][:r]
        S = np.sqrt(np.clip(evals[idx], 1e-10, None))
        Vk = V[:, idx]
        U = np.empty((X.shape[0], len(idx)), np.float32)
        for k in range(0, X.shape[0], 4096):
            U[k : k + 4096] = ((X[k : k + 4096].astype(np.float64) @ Vk) / S).astype(
                np.float32
            )
        del X, G, V, Vk
        U = torch.tensor(U, device=dev)
        return (
            lambda A: (torch.tensor(A, device=dev).reshape(A.shape[0], -1) @ U)
            .double()
            .cpu()
            .numpy(),
            f"pod r={r} n={len(ts) * ntraj}",
        )
    if model == "DBF":
        from src.eval.benchmark import dbf_load

        dbf, ck = dbf_load(CFG_KEY[ds].rsplit("_obs", 1)[0], dens, dev)
        return (
            lambda A: dbf.f_network(
                torch.tensor(A, device=dev).reshape(A.shape[0], -1).float()
            )
            .double()
            .cpu()
            .numpy(),
            ck,
        )
    if model == "TensorVar":
        from src.eval.benchmark import TV_DRIVER, tv_checkpoint_paths

        D = TV_DRIVER[CFG_KEY[ds].rsplit("_obs", 1)[0]]
        fwd, inv = tv_checkpoint_paths(CFG_KEY[ds].rsplit("_obs", 1)[0], dens)
        if D.get("inpaint"):
            from src.assimilation.tensorvar_inpaint import TVDAInpaint

            m = TVDAInpaint(
                fwd,
                inv,
                dev,
                width=D.get("width", 96),
                n_attn=D.get("n_attn", 4),
            )
        else:
            from src.assimilation.tensorvar import TVDA

            m = TVDA(fwd, inv, dev)
        return (
            lambda A: m.encode(torch.tensor(A, device=dev)).double().cpu().numpy(),
            f"{fwd} + {inv}",
        )
    from src.assimilation.models import build_model

    ck = os.path.join(_REPO, e["ckpt"].format(d=dens))
    m = build_model(model.lower().replace("-", "_"), ck, dev)
    return (lambda A: m.encode(torch.tensor(A, device=dev)).double().cpu().numpy(), ck)


CGKN_ENV = {
    "cylinder": (
        "src/model/cgkn/cgkn_grid.py",
        {
            "H": "64",
            "W": "64",
            "C": "2",
            "DATA_DIR": "data/cylinder",
            "SKIP": "5",
            "DZU": "32",
            "OBS_EVERY": "1",
        },
    ),
    "shallow_water": (
        "src/model/cgkn/cgkn_grid.py",
        {
            "H": "96",
            "W": "192",
            "C": "3",
            "DATA_DIR": "data/shallow_water",
            "SKIP": "2",
            "DZU": "8",
            "OBS_EVERY": "1",
        },
    ),
    "era5": (
        "src/model/cgkn/cgkn_era5.py",
        {
            "H": "64",
            "W": "32",
            "DATA_DIR": "data/era5",
            "SKIP": "4",
            "DZU": "32",
        },
    ),
    "era5_high": (
        "src/model/cgkn/cgkn_era5.py",
        {
            "H": "240",
            "W": "120",
            "DATA_DIR": "data/era5_high",
            "SKIP": "4",
            "DZU": "12",
        },
    ),
}
L96_OBS_N = {5: 2, 15: 6, 30: 12, 50: 20}


def _with_env(env, fn):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return fn()
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, path))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    return M


def _normal_eq(pairs, D):
    G = np.zeros((D, D))
    C = np.zeros((D, D))
    ss = np.zeros(D)
    n = 0
    for X, Y in pairs:
        G += X.T @ X
        C += X.T @ Y
        ss += (Y * Y).sum(0)
        n += len(X)
    W = np.linalg.solve(G + 1e-10 * (np.trace(G) / max(D, 1)) * np.eye(D), C)
    resid = float(
        np.sqrt(max(ss.sum() - 2 * (C * W).sum() + (W.T @ G @ W).trace(), 0.0))
    )
    return W, n, resid / max(np.sqrt(np.trace(G)), 1e-30)


def _cgkn_l96_train_fit(ds, dens, dev, D, batch=256):
    ck = os.path.join(_REPO, f"ckpt_release/lorenz96/cgkn/seed1/cgkn@{dens}.pt")
    sd = torch.load(ck, map_location="cpu")
    ek = sorted(
        [k for k in sd if k.startswith("autoencoder.encoder") and k.endswith("weight")],
        key=lambda k: int(k.split(".")[2]),
    )
    env = dict(DZ_UNIT=str(sd[ek[-1]].shape[0]), OBS_N=str(L96_OBS_N[dens]))

    def run():
        M = _load_module("l96mod_tf", "src/model/cgkn/cgkn_lorenz96.py")
        model = M.CGKN(M.AutoEncoder(M.BW, M.dim_z_unit), M.CGN()).to(dev)
        model.load_state_dict(sd)
        model.eval()
        u = M.load_flat("train")
        ntraj, T = u.shape[0], u.shape[1]
        stride = max(1, ntraj * max(0, T - 1) // max(KHAT_PAIRS, TRAIN_FIT_RATIO * D))

        def pairs():
            with torch.no_grad():
                for i in range(ntraj):
                    idx = np.arange(0, T - 1, stride)
                    for k in range(0, len(idx), batch):
                        j = idx[k : k + batch]
                        X = M.encode_u2(model.autoencoder, u[i, j].to(dev))
                        Y = M.encode_u2(model.autoencoder, u[i, j + 1].to(dev))
                        yield X.double().cpu().numpy(), Y.double().cpu().numpy()

        return _normal_eq(pairs(), D), stride

    (W, n, fit), stride = _with_env(env, run)
    return (
        W,
        "khat_train_dense",
        f"{ck} n={n} stride={stride} D={D} fit={fit:.4f}",
    )


def _cgkn_train_fit(ds, dens, dev, D, batch=32):
    mod, env = CGKN_ENV[ds]
    env = dict(env, OBS_R=str(dens / 100))
    ck = os.path.join(_REPO, f"ckpt_release/{ds}/cgkn/seed1/cgkn@{dens}.pt")

    def run():
        M = _load_module(f"cgkn_tf_{ds}", mod)
        model = M.CGKN().to(dev)
        model.load_state_dict(torch.load(ck, map_location=dev))
        model.eval()
        tr = M.load_frames("train")
        nrm = M.Norm(tr)
        a = np.asarray(tr)
        frames = a.reshape(-1, *a.shape[-3:]) if a.ndim == 5 else a
        ntraj, T = (a.shape[0], a.shape[1]) if a.ndim == 5 else (1, a.shape[0])
        NBR, NB, DZU = _patch_geometry(ds, dens)
        nbr = torch.as_tensor(NBR.reshape(NB, 9), device=dev, dtype=torch.long)

        def enc(x):
            with torch.no_grad():
                _u1, u2 = M.split_field(nrm(x).to(dev))
                z = model.ae.enc(u2).reshape(len(x), NB, DZU)[:, nbr, :]
                z = z.reshape(len(x), NB, 9 * DZU).transpose(0, 1)
                return z.reshape(-1, 9 * DZU).double().cpu().numpy()

        want = max(KHAT_PAIRS, TRAIN_FIT_RATIO * D)
        stride = max(1, ntraj * max(0, T - 1) // max(want // max(NB, 1), 1))

        def pairs():
            for i in range(ntraj):
                idx = np.arange(0, T - 1, stride)
                for k in range(0, len(idx), batch):
                    j = idx[k : k + batch]
                    yield enc(frames[i * T + j]), enc(frames[i * T + j + 1])

        return _normal_eq(pairs(), D), stride, NB

    (W, n, fit), stride, NB = _with_env(env, run)
    return (
        W,
        "khat_train_dense",
        f"{ck} n={n} stride={stride} NB={NB} D={D} fit={fit:.4f}",
    )


def _train_fit(ds, model, dens, dev, D):
    import torch
    from src.config import load_config
    from src.data import channel_norm

    if model == "CGKN":
        if ds == "lorenz96":
            return _cgkn_l96_train_fit(ds, dens, dev, D)
        return _cgkn_train_fit(ds, dens, dev, D)

    enc, prov = _train_encoder(ds, model, dens, dev)
    cfg = load_config("config/da.yaml", CFG_KEY[ds])
    skip = int(cfg.get("skip", 1))
    tr, closer = _lazy_trajs(cfg["data_path"])
    mean, std = channel_norm(np.asarray(tr[:, :: max(1, tr.shape[1] // 200)]), cfg)
    ntraj, T = (tr.shape[0], tr.shape[1])
    want = max(KHAT_PAIRS, TRAIN_FIT_RATIO * D)
    avail = ntraj * max(0, T - skip)
    stride = max(1, avail // max(want, 1))
    G = np.zeros((D, D))
    C = np.zeros((D, D))
    n = 0
    ss = np.zeros(D)
    for i in range(ntraj):
        idx = np.arange(0, T - skip, stride)
        for k in range(0, len(idx), 256):
            j = idx[k : k + 256]
            A0 = (
                np.asarray(tr[i, j], np.float32) - mean.reshape(1, -1, 1, 1)
            ) / std.reshape(1, -1, 1, 1)
            A1 = (
                np.asarray(tr[i, j + skip], np.float32) - mean.reshape(1, -1, 1, 1)
            ) / std.reshape(1, -1, 1, 1)
            with torch.no_grad():
                X, Y = (enc(A0), enc(A1))
            G += X.T @ X
            C += X.T @ Y
            ss += (Y * Y).sum(0)
            n += len(X)
    closer()
    ridge = 1e-10 * (np.trace(G) / max(D, 1))
    W = np.linalg.solve(G + ridge * np.eye(D), C)
    resid = float(
        np.sqrt(max(ss.sum() - 2 * (C * W).sum() + (W.T @ G @ W).trace(), 0.0))
    )
    fit = resid / max(np.sqrt(np.trace(G)), 1e-30)
    return (
        W,
        "khat_train_dense",
        f"{prov} n={n} stride={stride} D={D} fit={fit:.4f}",
    )


def operator(ds, model, dens, dev, L, R0, force_fitted=False, train_fit=False):
    if train_fit:
        return _train_fit(ds, model, dens, dev, L["tru"].shape[-1])
    if model in CROSSFIT or force_fitted:
        T = L["tru"]
        per = T.shape[0] // R0
        h = max(1, R0 // 2) * per
        X, Y = _pairs(T[:h])
        W = np.linalg.lstsq(X, Y, rcond=None)[0]
        fit = float(np.linalg.norm(Y - X @ W) / max(np.linalg.norm(X), 1e-30))
        return (
            W,
            "khat_crossfit",
            (
                f"lstsq on {len(X)} pairs from {h // per} of "
                f"{R0} runs (D={X.shape[1]}), fit={fit:.4f}"
            ),
        )

    e = _eval_entry(ds, model)
    if model in FITTED:
        ck = e["ckpt"].format(d=dens)
        if ck in _KHAT_MEMO:
            return _KHAT_MEMO[ck]
        from src.config import load_config
        from src.assimilation.models import build_model
        from src.data import _load_trajs, channel_norm

        m = build_model(model.lower().replace("-", "_"), os.path.join(_REPO, ck), dev)
        cfg = load_config("config/da.yaml", CFG_KEY[ds])
        skip = int(cfg.get("skip", 1))
        tr = _load_trajs(cfg["data_path"])
        mean, std = channel_norm(np.asarray(tr[:, :: max(1, tr.shape[1] // 200)]), cfg)
        ntraj, T = tr.shape[0], tr.shape[1]
        idx = np.unique(
            np.linspace(0, T - skip - 1, max(2, KHAT_PAIRS // ntraj)).astype(int)
        )

        def enc(frames):
            A = (
                np.asarray(frames, np.float32) - mean.reshape(1, -1, 1, 1)
            ) / std.reshape(1, -1, 1, 1)
            return m.encode(torch.tensor(A, device=dev)).double().cpu().numpy()

        with torch.no_grad():
            X = np.concatenate([enc(tr[i, idx]) for i in range(ntraj)])
            Y = np.concatenate([enc(tr[i, idx + skip]) for i in range(ntraj)])
        W = np.linalg.lstsq(X, Y, rcond=None)[0]
        fit = float(np.linalg.norm(Y - X @ W) / max(np.linalg.norm(X), 1e-30))
        out = (
            W,
            "khat_train",
            f"{ck} | lstsq on {len(X)} train pairs "
            f"(D={X.shape[1]}), train_fit_residual={fit:.4f}",
        )
        _KHAT_MEMO[ck] = out
        return out

    if model == "DBF":
        from src.model.dbf.utils import compute_K_sigma_block_diag

        ck = e["ckpt"].format(d=dens)
        st = torch.load(os.path.join(_REPO, ck), map_location="cpu", weights_only=False)
        lam, zd = st["lambdas"].double(), int(st["z_dim"])
        B = compute_K_sigma_block_diag(
            lam, steps=1, num_real=0, num_complex=zd // 2
        ).double()
        F = torch.zeros(zd, zd, dtype=torch.float64)
        for i in range(B.shape[0]):
            F[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = B[i]
        return F.T.numpy(), "exact", ck

    if model == "DMD":
        from src.config import load_config
        from src.assimilation.dmd import DMDDA

        if dev.type != "cuda":
            raise SystemExit(dev.type)
        key = CFG_KEY[ds]
        return (
            DMDDA(load_config("config/da.yaml", key), key, dev)
            .C.double()
            .cpu()
            .numpy()
            .T,
            "exact",
            f"refit@{key}",
        )

    from src.assimilation.models import build_model

    ck = (e["fwd"] if model == "TensorVar" else e["ckpt"]).format(d=dens)
    W = build_model(
        model.lower().replace("-", "_"), os.path.join(_REPO, ck), dev
    ).linear_operator()
    return W.detach().double().cpu().numpy(), "exact", ck


def blocks(meta, n_rows, R0):
    if n_rows > R0 and n_rows % R0 == 0:
        per = n_rows // R0
        return (
            [np.arange(k * per, (k + 1) * per) for k in range(R0)],
            f"patch_blocks={R0}",
        )
    src = meta.get("run_src")
    if src:
        by = {}
        for i, (tr, _t0) in enumerate(src[:n_rows]):
            by.setdefault(int(tr), []).append(i)
        if len(by) > 1:
            return [np.array(v) for v in by.values()], f"run_src={len(by)}"
    ntraj = NTRAJ.get((meta.get("dataset"), meta.get("split_tag")))
    if ntraj and ntraj > 1:
        npro = int(meta.get("n_protocol_starts") or 0)
        by = {}
        for i in range(n_rows):
            by.setdefault(f"p{i}" if i < npro else (i - npro) % ntraj, []).append(i)
        return (
            [np.array(v) for v in by.values()],
            f"reconstructed={len(by)}",
        )
    L = int(np.ceil(np.sqrt(max(n_rows, 1))))
    return (
        [np.arange(i, min(i + L, n_rows)) for i in range(0, n_rows, L)],
        f"moving_block={L}",
    )


def smoothing(L):
    def mv(Z):
        return np.linalg.norm(Z[:, 1:] - Z[:, :-1]) / max(
            np.linalg.norm(Z[:, :-1]), 1e-30
        )

    return float(mv(L["tru"]) / max(mv(L["ana"]), 1e-30))


def cell(
    path,
    ds,
    model,
    dens,
    dev,
    n_boot=40,
    gate_only=False,
    force_fitted=False,
    train_fit=False,
):
    L, meta, R0 = legs(path, ds, dens, model)
    D = L["tru"].shape[-1]
    W, kind, prov = operator(ds, model, dens, dev, L, R0, force_fitted, train_fit)
    rec = {
        "dataset": ds,
        "model": model,
        "density": dens,
        "D": int(D),
        "dictionary": meta["dictionary"],
        "operator_kind": kind,
        "operator_provenance": prov,
        "n_runs": R0,
        "n_rows": int(L["tru"].shape[0]),
        "split_tag": meta.get("split_tag"),
        "sigma_q": meta.get("sq"),
        "sigma_r": meta.get("sr"),
        "smoothing": smoothing(L),
        "legs": {},
    }
    if W.shape != (D, D):
        rec["verdict"] = "shape_mismatch"
        return rec

    g = None
    n_bad = 0
    if L["bef"].size:
        X, Y, n_bad = _finite_pairs(L["bef"])
        if len(X):
            g = float(np.abs(Y - X @ W).max() / max(np.abs(Y).max(), 1e-30))
            rec["free_forecast_deviation_frobenius"] = float(
                np.linalg.norm(Y - X @ W) / max(np.linalg.norm(X), 1e-30)
            )
    rec["free_forecast_steps_dropped_nonfinite"] = int(n_bad)
    rec["gate_free_forecast"] = g
    rec["gate_is_a_check"] = kind == "exact"
    rec["gate_passed"] = (g is not None and g < GATE) if kind == "exact" else None
    if kind == "exact":
        if g is None:
            rec["verdict"] = "no_free_forecast"
            return rec
        if g >= GATE:
            rec["verdict"] = "gate_failed"
            return rec
    if gate_only:
        return rec

    lam, V = np.linalg.eig(W.astype(np.complex128))
    rec["near_unit_eigenvalue_fraction"] = float(
        (np.abs(np.abs(lam) - 1.0) < 0.01).mean()
    )
    per = L["tru"].shape[0] // R0
    lo = max(1, R0 // 2) * per if kind == "khat_crossfit" else 0
    raw, drops = {}, {}
    for name, key in (("truth", "tru"), ("analysis", "ana"), ("free_forecast", "bef")):
        Z = L[key][lo:] if L[key].size else L[key]
        if Z.size == 0:
            rec["legs"][name] = {"absent": True}
            continue
        X, Y, nb = _finite_pairs(Z)
        drops[name] = int(nb)
        if len(X) == 0:
            rec["legs"][name] = {
                "absent": True,
                "steps_dropped_nonfinite": int(nb),
            }
            continue
        raw[name] = (X,) + resdmd.residuals_given(lam, V, X, Y)
    if "truth" not in raw or "analysis" not in raw:
        rec["verdict"] = "no_finite_pairs"
        return rec

    common = raw["truth"][3] & raw["analysis"][3]
    for name, (X, res, w, kept) in raw.items():
        sel = kept & common
        rs, ws = res[sel], w[sel] / max(w[sel].sum(), 1e-30)
        leg = {
            "n_pairs": int(len(X)),
            "n_kept": int(kept.sum()),
            "n_common": int(sel.sum()),
            "n_this_leg_only": int((kept & ~sel).sum()),
            "steps_dropped_nonfinite": drops.get(name, 0),
            "res_median": float(np.median(rs)) if rs.size else None,
            "res_energy_weighted": float(np.sum(rs * ws)) if rs.size else None,
            "res_p10": float(np.percentile(rs, 10)) if rs.size else None,
            "res_p90": float(np.percentile(rs, 90)) if rs.size else None,
            "in_eps_pseudospectrum": {str(e): int((rs <= e).sum()) for e in EPS},
            "res_median_this_leg_only": (
                float(np.median(res[kept & ~sel])) if (kept & ~sel).any() else None
            ),
        }
        if not rs.size:
            leg["absent"] = True
        rec["legs"][name] = leg
    if not n_boot:
        return rec

    nrows = L["tru"].shape[0] - lo
    bl, kindb = blocks(meta, nrows, max(1, R0 - lo // max(per, 1)) if lo else R0)
    rec["bootstrap_unit"] = kindb
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(bl), len(bl))
        sel = np.concatenate([bl[b] for b in pick])
        v = {}
        for name, key in (("truth", "tru"), ("analysis", "ana")):
            Xb, Yb = _pairs(L[key][lo:][sel])
            r, _w, _k = resdmd.residuals_given(lam, V, Xb, Yb)
            v[name] = float(np.nanmedian(r[common]))
        pairs.append((v["truth"], v["analysis"]))
    a = np.array(pairs)
    rec["boot_pairs"] = [[round(x, 6), round(y, 6)] for x, y in pairs]
    d = a[:, 1] - a[:, 0]
    rec["delta_bootstrap"] = {
        "n": len(d),
        "mean": float(d.mean()),
        "std": (float(d.std(ddof=1)) if len(d) > 1 else None),
        "p10": float(np.percentile(d, 10)),
        "p50": float(np.percentile(d, 50)),
        "p90": float(np.percentile(d, 90)),
    }
    ok = a[:, 0] > 1e-12
    r = a[ok, 1] / a[ok, 0]
    rec["ratio_bootstrap"] = {
        "n": int(ok.sum()),
        "mean": float(r.mean()) if r.size else None,
        "std": (float(r.std(ddof=1)) if r.size > 1 else None),
        "p10": float(np.percentile(r, 10)),
        "p50": float(np.percentile(r, 50)),
        "p90": float(np.percentile(r, 90)),
    }
    for i, name in enumerate(("truth", "analysis")):
        rec["legs"][name]["res_std"] = float(a[:, i].std(ddof=1))
    return rec


def caches():
    out = {}
    pat = re.compile(
        r"^(lorenz96|cylinder|shallow_water|era5_high|era5)"
        r"_(NL-AE|WL-AE|KAE|TensorVar|DMD|DBF|CGKN)"
        r"(?:_n(\d+))?(?:_obs(\d+))?(?:_(\w+))?$"
    )
    for p in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        m = pat.match(os.path.basename(p)[:-4])
        if not m or m.group(5) == "convref":
            continue
        ds, model, tag = m.group(1), m.group(2), m.group(5)
        n, d = int(m.group(3) or 0), int(m.group(4) or 30)
        base = f"{tag}_{ds}_{model}" if tag else f"{ds}_{model}"
        k = base if d == 30 else f"{base}@obs{d}"
        if k not in out or n > out[k][0]:
            out[k] = (n, p, d)
    return {k: (v[1], v[2]) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="")
    ap.add_argument("--n-boot", type=int, default=40)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument(
        "--train-fit",
        action="store_true",
    )
    ap.add_argument(
        "--fitted-all",
        action="store_true",
    )
    ap.add_argument("--only", default="")
    ap.add_argument(
        "--resume",
        action="store_true",
    )
    ap.add_argument(
        "--obs30-only",
        action="store_true",
    )
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for key, (path, dens) in sorted(caches().items()):
        if a.cells and a.cells not in key:
            continue
        model = key.split("@")[0].rsplit("_", 1)[1]
        if a.only and model not in a.only.split(","):
            continue
        if a.resume and key in all_:
            continue
        if a.obs30_only and re.search(r"@obs\d+$", key):
            continue
        ds = json.loads(str(np.load(path, allow_pickle=True)["meta"])).get("dataset")
        try:
            rec = cell(
                path,
                ds,
                model,
                dens,
                dev,
                0 if a.gate_only else a.n_boot,
                a.gate_only,
                a.fitted_all,
                a.train_fit,
            )
        except Exception as e:
            print(f"  {key:40s} FAILED ({type(e).__name__}: {str(e)[:60]})", flush=True)
            continue
        all_[key] = rec
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(all_, open(a.out, "w"), indent=1)
        g = rec.get("gate_free_forecast")
        gs = "n/a" if g is None else f"{g:.2e}"
        if "verdict" in rec:
            print(f"  {key:40s} gate={gs:>9s}  {rec['verdict'][:60]}", flush=True)
        elif not rec["legs"]:
            print(f"  {key:40s} gate={gs:>9s}  PASS", flush=True)
        else:
            t, an = rec["legs"]["truth"], rec["legs"]["analysis"]
            print(
                f"  {key:40s} gate={gs:>9s}  {t['res_median']:.4f} -> "
                f"{an['res_median']:.4f}  d={an['res_median'] - t['res_median']:+.4f}"
                f"  smooth={rec['smoothing']:.2f}x",
                flush=True,
            )
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
