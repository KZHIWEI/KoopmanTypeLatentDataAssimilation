from __future__ import annotations
import json, os
import numpy as np
import torch
import torch.nn as nn

I = 40
OBS = os.environ["OBS"]
OBS_N = {"5": 2, "15": 6, "30": 12, "50": 20}[OBS]
dim_z_unit = int(os.environ.get("DZ_UNIT", 6))

_CGKN_IN = os.environ.get("CGKN_INPUTS", "ckpt_release/cgkn_inputs")


def _input(env, default_name):
    base = os.path.dirname(os.environ["CKPT"]) if env == "NOISE" else _CGKN_IN
    p = os.environ.get(env) or os.path.join(base, default_name)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p


_FILTER = os.environ.get("FILTER", "discrete")
OBS_EVERY = int(os.environ.get("OBS_EVERY", 1))
SPINUP = int(os.environ.get("SPINUP", 2))
RUN_LENGTH = int(os.environ.get("RUN_LENGTH", 40))

indices_u1 = np.unique(np.round(np.linspace(0, I, OBS_N, endpoint=False)).astype(int))
NB = len(indices_u1)
_blocks = []
for b in range(NB):
    lo = indices_u1[b]
    hi = indices_u1[(b + 1) % NB]
    seg = list(range(lo + 1, I)) if (b + 1) % NB == 0 else list(range(lo + 1, hi))
    _blocks.append(seg)
BW = max(len(s) for s in _blocks)
block_pad = np.zeros((NB, BW), np.int64)
block_msk = np.zeros((NB, BW), np.float32)
for b, seg in enumerate(_blocks):
    for j, site in enumerate(seg):
        block_pad[b, j] = site
        block_msk[b, j] = 1.0
dim_u1 = NB
dim_z = dim_z_unit * NB
_nbr = np.stack([(np.arange(NB) - 1) % NB, np.arange(NB), (np.arange(NB) + 1) % NB], 1)
_bp = torch.tensor(block_pad)
_bm = torch.tensor(block_msk)


class AutoEncoder(nn.Module):
    def __init__(self, bw, hidden):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(bw, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, hidden),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bw),
        )


class CGN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim_u1 = NB
        self.dim_z = dim_z
        self.dzu = dim_z_unit
        out = 1 + 3 * dim_z_unit + dim_z_unit + 3 * dim_z_unit * dim_z_unit
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out),
        )
        self.register_buffer("nbr", torch.tensor(_nbr))

    def forward(self, x):
        B = x.shape[0]
        dzu = self.dzu
        win = torch.stack(
            [x[:, self.nbr[:, 0]], x[:, self.nbr[:, 1]], x[:, self.nbr[:, 2]]], -1
        )
        o = self.net(win)
        i = 0
        f1 = o[..., i : i + 1]
        i += 1
        a1 = o[..., i : i + 3 * dzu].reshape(B, NB, 3, dzu)
        i += 3 * dzu
        f2u = o[..., i : i + dzu]
        i += dzu
        a2 = o[..., i : i + 3 * dzu * dzu].reshape(B, NB, dzu, 3, dzu)
        g1 = torch.zeros(B, NB, dim_z, device=x.device)
        for k in range(3):
            cols = self.nbr[:, k].unsqueeze(-1) * dzu + torch.arange(
                dzu, device=x.device
            )
            g1.scatter_add_(2, cols.unsqueeze(0).expand(B, -1, -1), a1[:, :, k, :])
        f2 = f2u.reshape(B, dim_z, 1)
        rows = torch.arange(NB, device=x.device).unsqueeze(1).expand(NB, 3)
        G = torch.zeros(B, NB, NB, dzu, dzu, device=x.device)
        G.index_put_(
            (
                torch.arange(B, device=x.device).view(B, 1, 1),
                rows.unsqueeze(0),
                self.nbr.unsqueeze(0),
            ),
            a2.permute(0, 1, 3, 2, 4),
            accumulate=True,
        )
        g2 = G.permute(0, 1, 3, 2, 4).reshape(B, dim_z, dim_z)
        return [f1.reshape(B, NB, 1), g1, f2, g2]


class CGKN(nn.Module):
    def __init__(self, ae, cgn):
        super().__init__()
        self.autoencoder = ae
        self.cgn = cgn


def encode_u2(ae, u):
    lead = u.shape[:-1]
    blk = u[..., _bp.to(u.device)] * _bm.to(u.device)
    return ae.encoder(blk).reshape(*lead, dim_z)


def decode_blocks(ae, z):
    lead = z.shape[:-1]
    return ae.decoder(z.reshape(*lead, NB, dim_z_unit))


def true_blocks(u):
    return u[..., _bp.to(u.device)] * _bm.to(u.device)


def assemble(u1, blocks):
    lead = u1.shape[:-1]
    out = torch.zeros(*lead, I, device=u1.device, dtype=u1.dtype)
    out[..., indices_u1] = u1
    bp = _bp.to(u1.device)
    bm = _bm.to(u1.device)
    out[..., bp] = out[..., bp] * (1 - bm) + blocks * bm
    return out


def CGFilter(cgkn, sigma, u1, mu0, R0, dt, *, spinup=0, return_observed=False, time_offset=0):
    dev = u1.device
    Nt = u1.shape[0]
    d1 = u1.shape[1]
    dz = mu0.shape[0]
    observed = [u1[0]]
    if _FILTER == "discrete":
        s1sq = torch.diag(sigma[:d1] ** 2) * dt
        s2sq = torch.diag(sigma[d1:] ** 2) * dt
        Idz = torch.eye(dz, device=dev)
        mu = torch.zeros((Nt, dz, 1), device=dev)
        mu[0] = mu0
        m, Rc = mu0, R0
        for n in range(1, Nt):
            up = observed[-1]
            f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(up.T)]
            F = Idz + g2 * dt
            c = f2 * dt
            Hm = g1 * dt
            off = f1 * dt
            obs_step = n >= spinup and (time_offset + n) % OBS_EVERY == 0
            if obs_step:
                un = u1[n]
            else:
                un = up + (f1 + g1 @ m) * dt
            mp = F @ m + c
            Rp = F @ Rc @ F.T + s2sq
            if obs_step:
                y = un - up
                S = Hm @ Rc @ Hm.T + s1sq
                cross = F @ Rc @ Hm.T
                K = cross @ torch.linalg.inv(S)
                m = mp + K @ (y - (Hm @ m + off))
                Ra = Rp - K @ cross.T
                Rc = 0.5 * (Ra + Ra.T)
            else:
                m = mp
                Rc = 0.5 * (Rp + Rp.T)
            mu[n] = m
            observed.append(un)
        return (mu, torch.stack(observed)) if return_observed else mu
    s1, s2 = torch.diag(sigma[:d1]), torch.diag(sigma[d1:])
    invs1os1 = torch.linalg.inv(s1 @ s1.T)
    s2os2 = s2 @ s2.T
    mu = torch.zeros((Nt, dz, 1), device=dev)
    mu[0] = mu0
    m, Rc = mu0, R0
    for n in range(1, Nt):
        up = observed[-1]
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(up.T)]
        obs_step = n >= spinup and (time_offset + n) % OBS_EVERY == 0
        if obs_step:
            un = u1[n]
        else:
            un = up + (f1 + g1 @ m) * dt
        if obs_step:
            du1 = un - up
            m1 = (
                m
                + (f2 + g2 @ m) * dt
                + (Rc @ g1.T) @ invs1os1 @ (du1 - (f1 + g1 @ m) * dt)
            )
            R1 = (
                Rc + (g2 @ Rc + Rc @ g2.T + s2os2 - Rc @ g1.T @ invs1os1 @ g1 @ Rc) * dt
            )
        else:
            m1 = m + (f2 + g2 @ m) * dt
            R1 = Rc + (g2 @ Rc + Rc @ g2.T + s2os2) * dt
        mu[n] = m1
        m, Rc = m1, R1
        observed.append(un)
    return (mu, torch.stack(observed)) if return_observed else mu


@torch.no_grad()
def main():
    dev = torch.device(
        os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    )
    ae = AutoEncoder(BW, dim_z_unit)
    cgn = CGN()
    m = CGKN(ae, cgn).to(dev)
    m.load_state_dict(torch.load(os.environ["CKPT"], map_location=dev))
    m.eval()
    sigma = torch.load(_input("NOISE", f"sigma@{OBS}.pt"), map_location=dev)
    print(
        f"[l96 obs{OBS}] OBS_N={NB}/40 BW={BW} dz={dim_z} filter={_FILTER}",
        flush=True,
    )

    a = np.load(_input("EVAL_RUNS", "eval_runs_lorenz96.npy")).astype(np.float32)
    u_all = torch.tensor(a.reshape(a.shape[0], a.shape[1], I), device=dev)

    norms = json.load(open(_input("NORMALIZATION", "normalization.json")))
    mb, sb = [float(v[0]) for v in norms["bench_norm"]["lorenz96"]]
    SUB, DT = 1, 0.05
    n_runs = u_all.shape[0]
    errs = np.zeros((n_runs, RUN_LENGTH), np.float64)
    n_fields = int(os.environ.get("FIELDS", 0))
    n_lat = int(os.environ.get("LATENT", 0))
    lat_tru = []
    lat_ana = []
    fields = []
    for k in range(n_runs):
        u = u_all[k]
        mu0 = encode_u2(ae, u[:1]).reshape(dim_z, 1)
        u1 = u[:, indices_u1]
        mu, u1_used = CGFilter(
            m, sigma, u1.unsqueeze(-1), mu0, 0.01 * torch.eye(dim_z, device=dev), DT,
            spinup=SPINUP, return_observed=True,
        )
        sel = torch.arange(0, RUN_LENGTH + 1, device=dev) * SUB
        blk = decode_blocks(ae, mu.squeeze(-1)[sel])
        full = assemble(u1_used.squeeze(-1)[sel], blk)
        gt = u[sel]
        pb = ((full - mb) / sb).reshape(-1, 1, 8, 5)
        gb = ((gt - mb) / sb).reshape(-1, 1, 8, 5)
        num = ((pb - gb) ** 2).mean(dim=(1, 2, 3))
        den = (gb**2).mean(dim=(1, 2, 3)).clamp(min=1e-12)
        errs[k] = torch.sqrt(num / den).cpu().numpy()[:RUN_LENGTH]
        if k < n_fields:
            fields.append(pb.detach().cpu().numpy().astype(np.float32))
        if k < n_lat:
            lat_ana.append(mu.squeeze(-1)[sel].detach().cpu().numpy().astype("float32"))
            lat_tru.append(encode_u2(ae, gt).detach().cpu().numpy().astype("float32"))
        print(f"  run{k} post-spinup {errs[k, SPINUP:].mean():.5f}", flush=True)
    score = float(errs[:, SPINUP:].mean())
    meta = dict(
        dataset="lorenz96",
        obs=OBS,
        model="CGKN",
        protocol=("benchmark-strict"),
        obs_every=OBS_EVERY,
        spinup=SPINUP,
        run_length=RUN_LENGTH,
        n_runs=n_runs,
        filter=_FILTER,
        dt=DT,
        substeps=SUB,
        rrmse=score,
    )
    extra = {}
    if n_lat:
        extra["lat_tru"] = np.stack(lat_tru)
        extra["lat_ana"] = np.stack(lat_ana)
    if fields:
        extra["fields"] = np.stack(fields)
    np.savez(
        os.environ["OUT"], errs=errs.astype(np.float32), meta=json.dumps(meta), **extra
    )
    print(f"l96 obs{OBS} ({_FILTER}) RRMSE {score:.5f}", flush=True)


if __name__ == "__main__":
    main()
