from __future__ import annotations

import importlib.util
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.eval.gaussianity.ensembles import common as C


def _module(ds: str, density: int, cfg):
    from src.eval.benchmark import cgkn_env

    stem, env = cgkn_env(ds, density)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(f"_gauss_cgkn_{ds}", stem)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, dict(env, script=stem)


def _assert_batch_equiv(ref, got, tol=1e-2):
    d = float((ref - got).abs().max())
    s = float(ref.abs().max()) + 1e-12
    if d / s > tol:
        raise RuntimeError(d, s)


def _kalman_patch(mod, model, u1_seq, mu0, obs_steps, s1sq=0.05, s2sq=1.0, *, time_offset=0):
    if any(n not in obs_steps for n in range(1, u1_seq.shape[0])):
        return torch.stack(
            [
                mod.kalman_window(
                    model, u1_seq, s1sq, s2sq, mu0[:, j], time_offset=time_offset
                )
                for j in range(mu0.shape[1])
            ],
            dim=-1,
        )
    DZ = mod.DZ
    Idz = torch.eye(DZ, device=u1_seq.device)
    m, Rc = mu0, 0.01 * Idz
    d1inv = 1.0 / s1sq
    mus = [m]
    for n in range(1, u1_seq.shape[0]):
        up, un = u1_seq[n - 1], u1_seq[n]
        f1, Hm, f2, Am = model.cgn.dense(up.reshape(1, mod.NB, mod.DU1_P))
        f1, Hm, f2, Am = f1[0], Hm[0], f2[0], Am[0]
        F = Idz + Am
        mp = F @ m + f2.unsqueeze(1)
        Rp = F @ Rc @ F.T + s2sq * Idz
        if n in obs_steps:
            y = (un - up).unsqueeze(1)
            v = y - (Hm @ m + f1.unsqueeze(1))
            G = Hm.T @ (d1inv * Hm)
            Rcond = torch.linalg.inv(torch.linalg.inv(Rc) + G)
            m = mp + F @ Rcond @ (Hm.T @ (d1inv * v))
            Ra = F @ Rcond @ F.T + s2sq * Idz
            Rc = 0.5 * (Ra + Ra.T)
        else:
            m = mp
            Rc = 0.5 * (Rp + Rp.T)
        mus.append(m)
        del Hm, Am
    return torch.stack(mus)


def _kalman_l96(mod, cgkn, sigma, u1, mu0, R0, dt, obs_steps, *, time_offset=0):
    if any(n not in obs_steps for n in range(1, u1.shape[0])):
        return torch.stack(
            [
                mod.CGFilter(
                    cgkn, sigma, u1, mu0[:, j : j + 1], R0, dt,
                    time_offset=time_offset,
                )[:, :, 0]
                for j in range(mu0.shape[1])
            ],
            dim=-1,
        )
    dev = u1.device
    d1, dz = u1.shape[1], mu0.shape[0]
    s1sq = torch.diag(sigma[:d1] ** 2) * dt
    s2sq = torch.diag(sigma[d1:] ** 2) * dt
    Idz = torch.eye(dz, device=dev)
    m, Rc = mu0, R0
    mus = [m]
    for n in range(1, u1.shape[0]):
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(u1[n - 1].T)]
        F = Idz + g2 * dt
        c = f2 * dt
        Hm = g1 * dt
        off = f1 * dt
        mp = F @ m + c
        Rp = F @ Rc @ F.T + s2sq
        if n in obs_steps:
            y = u1[n] - u1[n - 1]
            S = Hm @ Rc @ Hm.T + s1sq
            cross = F @ Rc @ Hm.T
            K = cross @ torch.linalg.inv(S)
            m = mp + K @ (y - (Hm @ m + off))
            Ra = Rp - K @ cross.T
            Rc = 0.5 * (Ra + Ra.T)
        else:
            m = mp
            Rc = 0.5 * (Rp + Rp.T)
        mus.append(m)
    return torch.stack(mus)


def run(
    ds: str,
    *,
    members: int,
    seed: int,
    density: int,
    **_,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict]:
    dev = C.device()
    cfg, _data, _runs, _sampler, geom = C.load_protocol(ds, density)
    mod, env = _module(ds, density, cfg)
    spin, Wn, oev = geom["spinup"], geom["window"], geom["obs_every"]
    assert spin + Wn <= geom["run_length"]
    obs_steps = {n for n in range(1, Wn + 1) if (spin + n) % oev == 0}
    if not obs_steps:
        raise ValueError("CGKN Gaussianity window contains no observation update")
    analysis_step = min(obs_steps)
    stop = spin + analysis_step + 1
    DZ = mod.dim_z if ds == "lorenz96" else mod.DZ
    Bnp = 0.01 * np.eye(DZ)
    draw = C.Draw(seed)
    info: Dict = {}
    out, refs = [], []

    if ds == "lorenz96":
        ae = mod.AutoEncoder(mod.BW, mod.dim_z_unit)
        cgkn = mod.CGKN(ae, mod.CGN()).to(dev)
        cgkn.load_state_dict(torch.load(os.environ["CKPT"], map_location=dev))
        cgkn.eval()
        sigma = torch.load(
            mod._input("NOISE", f"sigma@{density}.pt"), map_location=dev
        )
        a = np.load(os.environ["EVAL_RUNS"]).astype(np.float32)
        u_all = torch.tensor(a.reshape(a.shape[0], a.shape[1], mod.I), device=dev)
        R0 = 0.01 * torch.eye(mod.dim_z, device=dev)
        with torch.no_grad():
            _f1, g1, _f2, g2 = cgkn.cgn(u_all[0, spin, mod.indices_u1].unsqueeze(-1).T)
        info.update(
            H_absmax=float((g1 * 0.05).abs().max()),
            A_absmax=float((g2 * 0.05).abs().max()),
        )
        for k in range(u_all.shape[0]):
            u = u_all[k]
            u1 = u[spin:stop, mod.indices_u1].unsqueeze(-1)
            with torch.no_grad():
                zt = mod.encode_u2(ae, u[spin:stop])
            eps = 0.1 * draw.bg(members, DZ).to(dev)
            mu0 = (zt[0].unsqueeze(0) + eps).T.contiguous()
            with torch.no_grad():
                ref = mod.CGFilter(cgkn, sigma, u1, mu0[:, :1], R0, 0.05, time_offset=spin)
                got = _kalman_l96(
                    mod, cgkn, sigma, u1, mu0[:, :1], R0, 0.05, obs_steps,
                    time_offset=spin,
                )
            _assert_batch_equiv(ref[:, :, 0], got[:, :, 0])
            with torch.no_grad():
                mus = _kalman_l96(
                    mod, cgkn, sigma, u1, mu0, R0, 0.05, obs_steps,
                    time_offset=spin,
                )
            out.append(
                (
                    eps.cpu().numpy().astype(np.float64),
                    (mus[analysis_step] - zt[analysis_step].unsqueeze(1)).T.cpu().numpy().astype(np.float64),
                )
            )
            refs.append(float(zt[analysis_step].pow(2).mean().sqrt()))
    else:
        model = mod.CGKN().to(dev)
        model.load_state_dict(torch.load(os.environ["CKPT"], map_location=dev))
        model.eval()
        with open(mod._input("NORMALIZATION", "normalization.json")) as f:
            norms = json.load(f)
        m_c, s_c = [np.asarray(v, np.float32) for v in norms["cgkn_norm"][ds]]
        mc = torch.tensor(m_c, device=dev).view(1, mod.C, 1)
        sc = torch.tensor(s_c, device=dev).view(1, mod.C, 1)
        raw = np.load(os.environ["EVAL_RUNS"]).astype(np.float32)
        if mod.LAYOUT == "nhwc":
            raw = raw.transpose(0, 1, 4, 2, 3)
        L = mod.RUN_LENGTH
        probe = torch.tensor(raw[0, spin], device=dev).reshape(1, mod.C, mod.H * mod.W)
        with torch.no_grad():
            _f1, Hm, _f2, Am = model.cgn.dense(mod.split_field((probe - mc) / sc)[0])
        info.update(H_absmax=float(Hm.abs().max()), A_absmax=float(Am.abs().max()))
        del Hm, Am
        s1sq = 0.05
        for k in range(raw.shape[0]):
            xp = torch.tensor(raw[k, : L + 1], device=dev).reshape(
                L + 1, mod.C, mod.H * mod.W
            )
            x = (xp - mc) / sc
            u1, u2 = mod.split_field(x)
            u1f = u1.reshape(L + 1, mod.DU1)
            seq = u1f[spin:stop].contiguous()
            with torch.no_grad():
                zt = model.ae.enc(u2[spin:stop]).reshape(analysis_step + 1, mod.DZ)
            eps = 0.1 * draw.bg(members, DZ).to(dev)
            mu0 = (zt[0].unsqueeze(0) + eps).T.contiguous()
            with torch.no_grad():
                ref = mod.kalman_window(model, seq, s1sq, 1.0, mu0[:, 0], time_offset=spin)
                got = _kalman_patch(
                    mod, model, seq, mu0[:, :1], obs_steps, time_offset=spin
                )
            _assert_batch_equiv(ref, got[:, :, 0])
            with torch.no_grad():
                mus = _kalman_patch(mod, model, seq, mu0, obs_steps, time_offset=spin)
            out.append(
                (
                    eps.cpu().numpy().astype(np.float64),
                    (mus[analysis_step] - zt[analysis_step].unsqueeze(1)).T.cpu().numpy().astype(np.float64),
                )
            )
            refs.append(float(zt[analysis_step].pow(2).mean().sqrt()))
            del xp, x, u1, u2, u1f
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    meta = {
        "ckpt": env["CKPT"],
        "eval_runs": env["EVAL_RUNS"],
        "latent_coupling": info,
        "reference_rms": float(np.mean(refs)),
        "geometry": geom,
        "latent_dim": int(DZ),
        "analysis_selection": "first_observation_update",
        "analysis_step": analysis_step,
        "analysis_absolute_step": spin + analysis_step,
    }
    return out, Bnp, meta
