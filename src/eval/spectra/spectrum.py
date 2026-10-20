import argparse
import json
import os

import numpy as np
import torch
from scipy import stats as sst

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
os.chdir(REPO)
OUT = "results/spectra"


def radial_spectrum(img: np.ndarray):
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    npix = min(h, w)
    fourier = np.fft.fftn(img)
    amps = np.abs(fourier) ** 2
    kx = np.fft.fftfreq(h) * h
    ky = np.fft.fftfreq(w) * w
    kx2, ky2 = np.meshgrid(kx, ky, indexing="ij")
    knrm = np.sqrt(kx2**2 + ky2**2).ravel()
    amps = amps.ravel()
    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = sst.binned_statistic(knrm, amps, statistic="mean", bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return kvals, Abins


def ring_spectrum(vec: np.ndarray):
    v = np.asarray(vec, dtype=np.float64).ravel()
    n = v.size
    amps = np.abs(np.fft.rfft(v)) ** 2
    return np.arange(1, n // 2 + 1, dtype=float), amps[1 : n // 2 + 1]


ONE_D = {"lorenz96"}


def spectrum_of_batch(fields: np.ndarray, one_d: bool = False):
    fn = ring_spectrum if one_d else radial_spectrum
    ks, acc, n = None, None, 0
    for i in range(fields.shape[0]):
        for c in range(fields.shape[1]):
            k, p = fn(fields[i, c])
            if acc is None:
                ks, acc = k, np.zeros_like(p)
            acc += p
            n += 1
    return ks, acc / max(n, 1), int(np.prod(fields.shape[2:]))


RING_WIN = 7


def _ssim_ring(pred, truth):
    from skimage.metrics import structural_similarity as _ssim

    B, T, Cc = pred.shape[0], pred.shape[1], pred.shape[2]
    h = RING_WIN // 2
    out = np.zeros((B, T))
    for t in range(T):
        for b in range(B):
            acc = 0.0
            for c in range(Cc):
                g = np.asarray(truth[b, t, c]).ravel()
                p = np.asarray(pred[b, t, c]).ravel()
                gw = np.concatenate([g[-h:], g, g[:h]])
                pw = np.concatenate([p[-h:], p, p[:h]])
                rng = float(g.max() - g.min())
                _, m = _ssim(
                    gw, pw, data_range=max(rng, 1e-12), win_size=RING_WIN, full=True
                )
                acc += float(m[h : h + g.size].mean())
            out[b, t] = acc / Cc
    return out


def ssim_curve(pred, truth, one_d=False):
    from skimage.metrics import structural_similarity as _ssim

    if one_d:
        return _ssim_ring(pred, truth)
    B, T, Cc = pred.shape[0], pred.shape[1], pred.shape[2]
    out = np.zeros((B, T))
    for t in range(T):
        for b in range(B):
            acc = 0.0
            for c in range(Cc):
                g = truth[b, t, c]
                rng = float(g.max() - g.min())
                acc += _ssim(g, pred[b, t, c], data_range=max(rng, 1e-12))
            out[b, t] = acc / Cc
    return out


def save_field_snapshot(ds, model, fields, truth, spin, forecast=False):
    folder = os.path.join("results", "fields", ds)
    os.makedirs(folder, exist_ok=True)
    selection = os.path.join(folder, "selection.npz")
    channel = 4 if ds in ("era5", "era5_high") else 0
    if model == "nl_ae" and not forecast:
        f, g = fields[:, spin:], truth[:, spin:]
        errors = np.sqrt(np.mean((f - g) ** 2, axis=(2, 3, 4)) / np.maximum(np.mean(g ** 2, axis=(2, 3, 4)), 1e-12)).mean(1)
        if not np.isfinite(errors).all():
            raise ValueError("Non-finite errors in field selection")
        run = int(np.argmin(np.abs(errors - np.median(errors))))
        step = max(spin, (fields.shape[1] - 1) // 2)
        from src.eval.benchmark import _common
        _, data, _, _ = _common(ds, 30)
        np.savez_compressed(selection, run=run, step=step, truth=truth[run, step, channel], mean=float(data.mean[channel, 0, 0]), std=float(data.std[channel, 0, 0]))
    if not os.path.isfile(selection):
        raise FileNotFoundError("Run NL-AE at 30% observations before exporting other field panels")
    with np.load(selection) as z:
        run, step = int(z["run"]), int(z["step"])
        if not np.allclose(z["truth"], truth[run, step, channel], rtol=1e-6, atol=1e-6):
            raise ValueError("Field panels must use the same truth and trajectory")
    leg = "forecast" if forecast else "analysis"
    np.savez_compressed(os.path.join(folder, f"{model}_{leg}.npz"), run=run, step=step, field=fields[run, step, channel])


def run_cell(ds, model_key, density, dev, spinup_only=True, forecast=False, save_fields=False):
    from src.eval.benchmark import DISP, OUT as BENCHMARK_OUT
    from src.eval.spectra.drivers import analysis_fields

    fields, gt, meta = analysis_fields(ds, model_key, density, dev, forecast=forecast)
    spin = int(meta["spinup"])
    f = torch.as_tensor(fields[:, spin:])
    g = torch.as_tensor(gt[:, spin:])
    num = ((f - g) ** 2).mean(dim=(2, 3, 4))
    den = (g**2).mean(dim=(2, 3, 4)).clamp(min=1e-12)
    meta["rrmse_measured"] = float(torch.sqrt(num / den).mean())
    p = f"{BENCHMARK_OUT}/{ds}/result/{DISP[model_key.lower().replace('-', '_')]}.json"
    if not forecast and os.path.exists(p):
        ref = json.load(open(p))["da"][f"obs{density}"]["rrmse"]
        meta["rrmse_published"] = ref
        meta["rrmse_rel_error"] = (meta["rrmse_measured"] - ref) / abs(ref)

    if save_fields and density == 30:
        save_field_snapshot(ds, model_key, fields, gt, spin, forecast)
    t0 = spin if spinup_only else 0
    fields, gt = fields[:, t0:], gt[:, t0:]
    flat = fields.reshape(-1, *fields.shape[2:])
    k, P, npix = spectrum_of_batch(flat, one_d=ds in ONE_D)
    ss = ssim_curve(fields, gt, one_d=ds in ONE_D)
    out = {
        "k": k.tolist(),
        "P": P.tolist(),
        "n_fields": int(flat.shape[0]),
        "n_pixels": npix,
        "ssim_t0": int(t0),
        "rrmse_per_run": np.sqrt(np.mean((fields - gt) ** 2, axis=(2, 3, 4)) / np.maximum(np.mean(gt ** 2, axis=(2, 3, 4)), 1e-12)).tolist(),
        "meta": meta,
    }
    if ss is not None:
        out["ssim"] = ss.mean(axis=0).tolist()
        out["ssim_per_run"] = ss.tolist()
        out["n_runs"] = int(ss.shape[0])
    out.update(bias_and_variance(fields, gt))
    return out


def bias_and_variance(fields, gt):
    b = np.asarray(fields, np.float64)
    t = np.asarray(gt, np.float64)
    ax = tuple(range(1, b.ndim))
    ok = np.isfinite(b).all(axis=ax) & np.isfinite(t).all(axis=ax)
    rms_t = np.sqrt((t**2).mean(axis=ax))
    var_t = t.var(axis=ax)
    with np.errstate(invalid="ignore", divide="ignore"):
        nmb = (b - t).mean(axis=ax) / np.maximum(rms_t, 1e-12)
        vr = b.var(axis=ax) / np.maximum(var_t, 1e-12)
    nmb = np.where(ok, nmb, np.nan)
    vr = np.where(ok, vr, np.nan)

    def _l(x):
        return [None if not np.isfinite(v) else float(v) for v in x]

    def _m(x):
        f = x[np.isfinite(x)]
        return float(f.mean()) if f.size else None

    return {
        "nmb": _m(nmb),
        "nmb_per_run": _l(nmb),
        "vr": _m(vr),
        "vr_per_run": _l(vr),
        "n_runs_bv": int(np.isfinite(nmb).sum()),
    }


def truth_spectrum(ds, density, dev):
    from src.eval.benchmark import _common

    cfg, data, runs, rl = _common(ds, density)
    spin = int(cfg["spinup"])
    gt = np.stack([torch.stack([r[t] for r in runs]).numpy() for t in range(spin, rl)])
    gt = gt.reshape(-1, *gt.shape[2:])
    k, P, npix = spectrum_of_batch(gt, one_d=ds in ONE_D)
    return {
        "k": k.tolist(),
        "P": P.tolist(),
        "n_fields": int(gt.shape[0]),
        "n_pixels": npix,
    }


def forecast_spectrum(ds, model_key, density, dev, save_fields=False):
    return run_cell(ds, model_key, density, dev, spinup_only=False, forecast=True, save_fields=save_fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--models",
        nargs="*",
        default=["nl_ae", "wl_ae", "kae", "tensorvar", "dmd", "dbf", "cgkn"],
    )
    ap.add_argument("--densities", default=None)
    ap.add_argument("--fields", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dens = (
        [int(x) for x in a.densities.split(",")]
        if a.densities
        else sorted(json.load(open("config/eval.json"))["densities"])
    )
    dev = torch.device(a.device)
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{a.dataset}.json"
    rec = (
        json.load(open(path))
        if os.path.exists(path)
        else {"dataset": a.dataset, "cells": {}}
    )
    if "truth" not in rec["cells"]:
        rec["cells"]["truth"] = truth_spectrum(a.dataset, dens[0], dev)
        print(f"  truth: {rec['cells']['truth']['n_fields']} fields", flush=True)
    failures = []
    for mk in a.models:
        for d in dens:
            key = f"{mk}:obs{d}"
            try:
                rec["cells"][key] = run_cell(a.dataset, mk, d, dev, save_fields=a.fields)
                print(f"  {key}: {rec['cells'][key]['n_fields']} fields", flush=True)
            except Exception as e:
                failures.append(key)
                rec["cells"].pop(key, None)
                print(f"  {key}: FAILED {type(e).__name__}: {e}", flush=True)
        try:
            rec["cells"][f"{mk}:forecast"] = forecast_spectrum(
                a.dataset, mk, dens[0], dev, save_fields=a.fields
            )
            print(
                f"  {mk}:forecast: {rec['cells'][f'{mk}:forecast']['n_fields']} fields",
                flush=True,
            )
            if a.fields and 30 in dens and dens[0] != 30:
                from src.eval.spectra.drivers import analysis_fields
                fields, truth, meta = analysis_fields(a.dataset, mk, 30, dev, forecast=True)
                save_field_snapshot(a.dataset, mk, fields, truth, int(meta["spinup"]), forecast=True)
        except NotImplementedError as e:
            failures.append(f"{mk}:forecast")
            rec["cells"].pop(f"{mk}:forecast", None)
            print(f"  {mk}:forecast unavailable {e}", flush=True)
        except Exception as e:
            failures.append(f"{mk}:forecast")
            rec["cells"].pop(f"{mk}:forecast", None)
            print(f"  {mk}:forecast: FAILED {type(e).__name__}: {e}", flush=True)
        with open(path, "w") as f:
            json.dump(rec, f)
    print("wrote", path)
    if failures:
        raise SystemExit(f"Failed spectra cells: {failures}")


if __name__ == "__main__":
    main()
