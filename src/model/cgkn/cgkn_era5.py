from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch

import torch.nn as nn
import torch.utils.checkpoint as cp

H = int(os.environ.get("H", 64))
W = int(os.environ.get("W", 32))
C = 5
P = 8
GH, GW = H // P, W // P
NB = GH * GW
OBS_R = float(os.environ.get("OBS_R", 0.5))
DZU = int(os.environ.get("DZU", 32))
DZ = NB * DZU
SEED = int(os.environ.get("SEED", 1))
torch.manual_seed(SEED)
np.random.seed(SEED)
CKDIR = os.environ.get("CKDIR", "checkpoint/cgkn")
DATA = os.environ.get("DATA_DIR", "data/era5")


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

if os.environ.get("FULL_OBS", "0") == "1":
    obs_pix = list(patches)
    hid_pix = list(patches)
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
                    yy = min(max(gy + dy, 0), GH - 1)
                    xx = min(max(gx + dx, 0), GW - 1)
                    ns.append(yy * GW + xx)
            out.append(ns)
    return np.array(out)


NBR = _nbrs()

_obs_idx = torch.tensor(np.stack(obs_pix))
_hid_idx = torch.tensor(np.stack(hid_pix))


def split_field(x):
    B = x.shape[0]
    o = x[:, :, _obs_idx.to(x.device)]
    h = x[:, :, _hid_idx.to(x.device)]
    u1 = o.permute(0, 2, 3, 1).reshape(B, NB, DU1_P)
    u2 = h.permute(0, 2, 3, 1).reshape(B, NB, DU2_P)
    return u1, u2


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
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

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


def kalman_window(model, u1_seq, s1sq, s2sq, mu0):
    T = u1_seq.shape[0]
    dev = u1_seq.device
    Idz = torch.eye(DZ, device=dev)
    m = mu0
    Rc = 0.01 * Idz
    mus = [m]
    d1inv = 1.0 / s1sq
    for n in range(1, T):

        def step(m, Rc, up, un):
            f1, Hm, f2, Am = model.cgn.dense(up.reshape(1, NB, DU1_P))
            f1, Hm, f2, Am = f1[0], Hm[0], f2[0], Am[0]
            F = Idz + Am
            mp = F @ m + f2
            y = un - up
            v = y - (Hm @ m + f1)
            G = Hm.T @ (d1inv * Hm)
            Rcond = torch.linalg.inv(torch.linalg.inv(Rc) + G)
            m_new = mp + F @ Rcond @ (Hm.T @ (d1inv * v))
            Ra = F @ Rcond @ F.T + s2sq * Idz
            return m_new, 0.5 * (Ra + Ra.T)

        m, Rc = cp.checkpoint(
            step, m, Rc, u1_seq[n - 1], u1_seq[n], use_reentrant=False
        )
        mus.append(m)
    return torch.stack(mus)


SKIP = int(os.environ.get("SKIP", "1"))
_frame_cache = {}


def load_frames(split):
    if split not in _frame_cache:
        f = h5py.File(f"{DATA}/{split}.h5", "r")
        _frame_cache[split] = np.asarray(f["data"][::SKIP])
    return _frame_cache[split]


class Norm:
    def __init__(self, arr):
        sub = np.asarray(arr[:2000]).astype(np.float32)
        self.mean = sub.mean(axis=(0, 1, 2))
        self.std = sub.std(axis=(0, 1, 2)) + 1e-6

    def __call__(self, a):
        x = (np.asarray(a, dtype=np.float32) - self.mean) / self.std
        t = torch.tensor(x)
        return t.permute(
            *range(t.dim() - 3), t.dim() - 1, t.dim() - 3, t.dim() - 2
        ).reshape(*t.shape[:-3], C, H * W)


def stage1(dev):
    os.makedirs(CKDIR, exist_ok=True)
    tr = load_frames("train")
    nrm = Norm(tr)
    model = CGKN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    T = tr.shape[0]
    for ep in range(1, int(os.environ.get("S1_EPOCHS", 10)) + 1):
        perm = np.random.permutation(T - 1)[:4000]
        tot = n = 0
        for i in range(0, len(perm), 64):
            idx = np.sort(perm[i : i + 64])
            x0 = nrm(tr[idx]).to(dev)
            x1 = nrm(tr[idx + 1]).to(dev)
            u1a, u2a = split_field(x0)
            u1b, u2b = split_field(x1)
            z0 = model.ae.enc(u2a)
            z1t = model.ae.enc(u2b)
            rec = model.ae.dec(z0, u1a)
            f1, g1, f2, g2 = model.cgn(u1a)
            znb = z0[:, model.cgn.nbr.reshape(-1), :].reshape(-1, NB, 9, DZU)
            du1 = f1 + torch.einsum(
                "bnok,bnk->bno",
                g1.reshape(-1, NB, DU1_P, 9 * DZU),
                znb.reshape(-1, NB, 9 * DZU),
            )
            dz = f2 + torch.einsum(
                "bnok,bnk->bno",
                g2.reshape(-1, NB, DZU, 9 * DZU),
                znb.reshape(-1, NB, 9 * DZU),
            )
            loss = (
                nn.functional.mse_loss(rec, u2a)
                + nn.functional.mse_loss(u1a + du1, u1b)
                + nn.functional.mse_loss(model.ae.dec(z0 + dz, u1a + du1), u2b)
                + nn.functional.mse_loss(z0 + dz, z1t.detach())
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            n += 1
        print(f"[s1 ep{ep}] loss {tot/max(n,1):.4f}", flush=True)
        torch.save(model.state_dict(), f"{CKDIR}/stage1.pt")


def rrmse_full(pred, tgt):
    return float(torch.sqrt(((pred - tgt) ** 2).mean()) / torch.sqrt((tgt**2).mean()))


@torch.no_grad()
def evaluate(model, dev, n_windows=8, length=40, spin=5):
    te = load_frames("val")
    nrm = Norm(load_frames("train"))
    Tt = te.shape[0]
    starts = np.linspace(0, Tt - length - 2, n_windows).astype(int)
    errs = []
    for s in starts:
        x = nrm(te[s : s + length + 1]).to(dev)
        u1, u2 = split_field(x)
        u1f = u1.reshape(length + 1, DU1)
        mu0 = model.ae.enc(u2[:1]).reshape(DZ)
        mu = kalman_window(model, u1f, s1sq=0.05, s2sq=1.0, mu0=mu0)
        u2d = model.ae.dec(mu.reshape(-1, NB, DZU), u1)
        full = assemble_field(u1, u2d)
        for t in range(spin, length + 1):
            errs.append(rrmse_full(full[t : t + 1], x[t : t + 1]))
    return float(np.mean(errs))


def stage2(dev):
    os.makedirs(CKDIR, exist_ok=True)
    tr = load_frames("train")
    nrm = Norm(tr)
    model = CGKN().to(dev)
    if os.path.exists(f"{CKDIR}/stage1.pt"):
        sd = torch.load(f"{CKDIR}/stage1.pt")
        own = model.state_dict()
        keep = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        print(f"[s2] warm: kept {len(keep)}/{len(own)} tensors", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    T = tr.shape[0]
    LONG = int(os.environ.get("S2_LONG", 100))
    best = 9e9
    for ep in range(1, int(os.environ.get("S2_EPOCHS", 400)) + 1):
        s = np.random.randint(0, T - LONG - 1)
        x = nrm(tr[s : s + LONG + 1]).to(dev)
        u1, u2 = split_field(x)
        u1f = u1.reshape(LONG + 1, DU1)
        mu0 = model.ae.enc(u2[:1]).reshape(DZ).detach()
        mu = kalman_window(model, u1f, s1sq=0.05, s2sq=1.0, mu0=mu0)
        u2d = model.ae.dec(mu.reshape(-1, NB, DZU), u1)
        loss_da = nn.functional.mse_loss(u2d, u2)
        rec = model.ae.dec(model.ae.enc(u2), u1)
        loss = loss_da + 0.2 * nn.functional.mse_loss(rec, u2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % 10 == 0:
            print(f"[s2 ep{ep}] da {float(loss_da):.4f}", flush=True)
        if ep % 50 == 0:
            v = evaluate(model, dev)
            print(f"[s2 ep{ep}] val {v:.4f}", flush=True)
            if v < best:
                best = v
                torch.save(model.state_dict(), f"{CKDIR}/stage2.pt")
                print(f"    new best {best:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["1", "2", "eval"], default="1")
    a = ap.parse_args()
    dev = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] obs_r={OBS_R} NO_P={NO_P} DU1={DU1} DZ={DZ} dev={dev}", flush=True)
    if a.stage == "1":
        stage1(dev)
    elif a.stage == "2":
        stage2(dev)
    else:
        m = CGKN().to(dev)
        m.load_state_dict(torch.load(f"{CKDIR}/stage2.pt", map_location=dev))
        print("EVAL", evaluate(m, dev))


if __name__ == "__main__":
    main()
