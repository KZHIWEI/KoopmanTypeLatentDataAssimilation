from __future__ import annotations
import json, os
import numpy as np
import torch

import torch.nn as nn

DS = os.environ["DS"]
OBS = os.environ["OBS"]
GEO = {
    "cylinder": (64, 64, 2, 32, 2, 4, 32, "nchw"),
    "shallow_water": (96, 192, 3, 8, 2, 4, 40, "nchw"),
    "era5": (64, 32, 5, 32, 1, 2, 14, "nhwc"),
    "era5_high": (240, 120, 5, 12, 1, 2, 14, "nhwc"),
}[DS]
H, W, C, DZU, OBS_EVERY, SPINUP, RUN_LENGTH, LAYOUT = GEO
OBS_EVERY = int(os.environ.get("OBS_EVERY", OBS_EVERY))
RUN_LENGTH = int(os.environ.get("RUN_LENGTH", RUN_LENGTH))
SPINUP = int(os.environ.get("SPINUP", SPINUP))
OBS_R = {"5": 0.05, "15": 0.15, "30": 0.30, "50": 0.50}[OBS]
P = 8
GH, GW = H // P, W // P
NB = GH * GW
DZ = NB * DZU


def stride_mask():
    n = max(1, int(round(OBS_R * P * P)))
    tile = np.zeros(P * P, dtype=bool)
    idx = (np.arange(n) * (P * P) / n).astype(int)
    tile[idx] = True
    return np.tile(tile.reshape(P, P), (GH, GW))


_M = stride_mask()
_pix = np.arange(H * W).reshape(H, W)
patches = [
    _pix[gy * P : (gy + 1) * P, gx * P : (gx + 1) * P].reshape(-1)
    for gy in range(GH)
    for gx in range(GW)
]
obs_pix = [p[_M.reshape(-1)[p]] for p in patches]
hid_pix = [p[~_M.reshape(-1)[p]] for p in patches]
NO_P = len(obs_pix[0])
NH_P = len(hid_pix[0])
DU1_P = NO_P * C
DU2_P = NH_P * C
DU1 = NB * DU1_P
assert all(len(o) == NO_P for o in obs_pix)


def _nbrs():
    out = []
    for gy in range(GH):
        for gx in range(GW):
            ns = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ns.append(
                        min(max(gy + dy, 0), GH - 1) * GW + min(max(gx + dx, 0), GW - 1)
                    )
            out.append(ns)
    return np.array(out)


NBR = _nbrs()
_obs_idx = torch.tensor(np.stack(obs_pix))
_hid_idx = torch.tensor(np.stack(hid_pix))


def split_field(x):
    B = x.shape[0]
    o = x[:, :, _obs_idx.to(x.device)]
    h = x[:, :, _hid_idx.to(x.device)]
    return (
        o.permute(0, 2, 3, 1).reshape(B, NB, DU1_P),
        h.permute(0, 2, 3, 1).reshape(B, NB, DU2_P),
    )


def assemble_field(u1, u2b):
    B = u1.shape[0]
    out = torch.zeros(B, C, H * W, device=u1.device, dtype=u1.dtype)
    o = u1.reshape(B, NB, NO_P, C).permute(0, 3, 1, 2)
    h = u2b.reshape(B, NB, NH_P, C).permute(0, 3, 1, 2)
    out[:, :, _obs_idx.to(u1.device)] = o
    out[:, :, _hid_idx.to(u1.device)] = h
    return out


class PatchAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(DU2_P, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, DZU),
        )
        self.dec_net = nn.Sequential(
            nn.Linear(DZU + 9 * DU1_P, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, DU2_P),
        )
        self.register_buffer("nbr", torch.tensor(NBR))

    def dec(self, z, u1):
        B = z.shape[0]
        ctx = u1[:, self.nbr.reshape(-1), :].reshape(B, NB, 9 * DU1_P)
        return self.dec_net(torch.cat([z, ctx], dim=-1))


class PatchCGN(nn.Module):
    def __init__(self):
        super().__init__()
        d_in = 9 * DU1_P
        self.out_f1 = DU1_P
        self.out_g1 = DU1_P * 9 * DZU
        self.out_f2 = DZU
        self.out_g2 = DZU * 9 * DZU
        out = self.out_f1 + self.out_g1 + self.out_f2 + self.out_g2
        self.net = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out),
        )
        self.register_buffer("nbr", torch.tensor(NBR))

    def forward(self, u1):
        B = u1.shape[0]
        win = u1[:, self.nbr.reshape(-1), :].reshape(B, NB, 9 * DU1_P)
        o = self.net(win)
        i = 0
        f1 = o[..., i : i + self.out_f1]
        i += self.out_f1
        g1 = o[..., i : i + self.out_g1].reshape(B, NB, DU1_P, 9, DZU)
        i += self.out_g1
        f2 = o[..., i : i + self.out_f2]
        i += self.out_f2
        g2 = o[..., i : i + self.out_g2].reshape(B, NB, DZU, 9, DZU)
        return f1, g1, f2, g2

    def dense(self, u1):
        B = u1.shape[0]
        f1, g1, f2, g2 = self(u1)
        Hm = torch.zeros(B, DU1, DZ, device=u1.device, dtype=u1.dtype)
        Am = torch.zeros(B, DZ, DZ, device=u1.device, dtype=u1.dtype)
        for k in range(9):
            nb = self.nbr[:, k]
            idx = nb.view(1, NB, 1, 1) * DZU + torch.arange(DZU, device=u1.device).view(
                1, 1, 1, DZU
            )
            Hm.reshape(B, NB, DU1_P, DZ).scatter_add_(
                3, idx.expand(B, NB, DU1_P, DZU), g1[:, :, :, k, :]
            )
            Am.reshape(B, NB, DZU, DZ).scatter_add_(
                3, idx.expand(B, NB, DZU, DZU), g2[:, :, :, k, :]
            )
        return f1.reshape(B, DU1), Hm, f2.reshape(B, DZ), Am


class CGKN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ae = PatchAE()
        self.cgn = PatchCGN()


@torch.no_grad()
def kalman_window(model, u1_seq, s1sq, s2sq, mu0, *, spinup=0, return_observed=False, time_offset=0):
    T = u1_seq.shape[0]
    dev = u1_seq.device
    Idz = torch.eye(DZ, device=dev)
    m = mu0
    Rc = 0.01 * Idz
    mus = [m]
    observed = [u1_seq[0]]
    d1inv = 1.0 / s1sq
    for n in range(1, T):
        up = observed[-1]
        obs_step = n >= spinup and (time_offset + n) % OBS_EVERY == 0
        f1, Hm, f2, Am = model.cgn.dense(up.reshape(1, NB, DU1_P))
        f1, Hm, f2, Am = f1[0], Hm[0], f2[0], Am[0]
        if obs_step:
            un = u1_seq[n]
        else:
            un = up + Hm @ m + f1
        F = Idz + Am
        mp = F @ m + f2
        Rp = F @ Rc @ F.T + s2sq * Idz
        if obs_step:
            y = un - up
            v = y - (Hm @ m + f1)
            G = Hm.T @ (d1inv * Hm)
            Rcond = torch.linalg.inv(torch.linalg.inv(Rc) + G)
            m = mp + F @ Rcond @ (Hm.T @ (d1inv * v))
            Ra = F @ Rcond @ F.T + s2sq * Idz
            Rc = 0.5 * (Ra + Ra.T)
        else:
            m = mp
            Rc = 0.5 * (Rp + Rp.T)
        mus.append(m)
        observed.append(un)
        del Hm, Am
    mu = torch.stack(mus)
    return (mu, torch.stack(observed)) if return_observed else mu


_CGKN_IN = os.environ.get("CGKN_INPUTS", "ckpt_release/cgkn_inputs")


def _input(env, default_name):
    p = os.environ.get(env) or os.path.join(_CGKN_IN, default_name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


def main():
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    norms = json.load(
        open(
            _input(
                "NORMALIZATION",
                {
                    "era5": "normalization_era5.json",
                    "era5_high": "normalization_era5_high.json",
                }.get(DS, "normalization.json"),
            )
        )
    )
    m_c, s_c = [np.asarray(v, np.float32) for v in norms["cgkn_norm"][DS]]
    m_b, s_b = [np.asarray(v, np.float32) for v in norms["bench_norm"][DS]]

    raw = np.load(_input("EVAL_RUNS", f"eval_runs_{DS}.npy")).astype(np.float32)
    if LAYOUT == "nhwc":
        raw = raw.transpose(0, 1, 4, 2, 3)
    n_runs, T = raw.shape[0], raw.shape[1]
    assert T >= RUN_LENGTH + 1, (T, RUN_LENGTH)

    model = CGKN().to(dev)
    model.load_state_dict(torch.load(os.environ["CKPT"], map_location=dev))
    model.eval()
    print(
        f"[{DS} obs{OBS}] NB={NB} NO_P={NO_P} DU1={DU1} DZ={DZ} obs_every={OBS_EVERY} "
        f"spinup={SPINUP} L={RUN_LENGTH} runs={n_runs}",
        flush=True,
    )

    mc = torch.tensor(m_c, device=dev).view(1, C, 1)
    sc = torch.tensor(s_c, device=dev).view(1, C, 1)
    mb = torch.tensor(m_b, device=dev).view(1, C, 1)
    sb = torch.tensor(s_b, device=dev).view(1, C, 1)

    errs = np.zeros((n_runs, RUN_LENGTH), np.float64)
    n_fields = int(os.environ.get("FIELDS", 0))
    n_lat = int(os.environ.get("LATENT", 0))
    lat_tru = []
    lat_ana = []
    fields = []
    for r in range(n_runs):
        xp = torch.tensor(raw[r, : RUN_LENGTH + 1], device=dev).reshape(
            RUN_LENGTH + 1, C, H * W
        )
        x = (xp - mc) / sc
        u1, u2 = split_field(x)
        u1f = u1.reshape(RUN_LENGTH + 1, DU1)
        with torch.no_grad():
            mu0 = model.ae.enc(u2[:1]).reshape(DZ)
        mu, u1f_eff = kalman_window(
            model, u1f, s1sq=0.05, s2sq=1.0, mu0=mu0,
            spinup=SPINUP, return_observed=True,
        )
        u1_eff = u1f_eff.reshape_as(u1)
        with torch.no_grad():
            u2d = model.ae.dec(mu.reshape(-1, NB, DZU), u1_eff)
            full = assemble_field(u1_eff, u2d)
            pred_phys = full * sc + mc
            pb = ((pred_phys - mb) / sb).reshape(-1, C, H, W)
            gb = ((xp - mb) / sb).reshape(-1, C, H, W)
            num = ((pb - gb) ** 2).mean(dim=(1, 2, 3))
            den = (gb**2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
            e = torch.sqrt(num / den).cpu().numpy()
        errs[r] = e[:RUN_LENGTH]
        if r < n_lat:
            with torch.no_grad():
                lat_ana.append(
                    mu.reshape(RUN_LENGTH + 1, -1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                lat_tru.append(
                    model.ae.enc(u2)
                    .reshape(RUN_LENGTH + 1, -1)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
        if r < n_fields:
            fields.append(pb.cpu().numpy().astype(np.float32))

        print(f"  run{r} post-spinup {errs[r, SPINUP:].mean():.5f}", flush=True)
        del xp, x, u1, u2, u1f, u1f_eff, u1_eff, mu
        if dev != "cpu":
            torch.cuda.empty_cache()

    score = float(errs[:, SPINUP:].mean())
    out = os.environ["OUT"]
    meta = dict(
        dataset=DS,
        obs=OBS,
        model="CGKN",
        protocol="benchmark-strict",
        obs_every=OBS_EVERY,
        spinup=SPINUP,
        run_length=RUN_LENGTH,
        n_runs=n_runs,
        s1sq=0.05,
        s2sq=1.0,
        rrmse=score,
    )
    extra = {}
    if fields:
        extra["fields"] = np.stack(fields)
    if lat_tru:
        extra["lat_tru"] = np.stack(lat_tru)
        extra["lat_ana"] = np.stack(lat_ana)
    np.savez(out, errs=errs.astype(np.float32), meta=json.dumps(meta), **extra)
    print(f"{DS} obs{OBS} RRMSE {score:.5f}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
