from __future__ import annotations
import argparse, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as nnF, torchdiffeq

I = 40
DT = float(os.environ.get("DT", 0.05))
DATA_DIR = os.environ.get("DATA_DIR", "data/lorenz96")
OBS_N = int(os.environ.get("OBS_N", 20))
dim_z_unit = int(os.environ.get("DZ_UNIT", 6))

indices_u1 = np.unique(np.round(np.linspace(0, I, OBS_N, endpoint=False)).astype(int))
NB = len(indices_u1)
_blocks = []
for b in range(NB):
    lo = indices_u1[b]
    hi = indices_u1[(b + 1) % NB]
    if (b + 1) % NB == 0:
        seg = list(range(lo + 1, I))
    else:
        seg = list(range(lo + 1, hi))
    _blocks.append(seg)
BW = max(len(s) for s in _blocks)
block_pad = np.zeros((NB, BW), np.int64)
block_msk = np.zeros((NB, BW), np.float32)
for b, seg in enumerate(_blocks):
    for j, site in enumerate(seg):
        block_pad[b, j] = site
        block_msk[b, j] = 1.0
if os.environ.get("FULL_OBS", "0") == "1":
    indices_u1 = np.arange(I)
    NB = I
    _blocks = [[b] for b in range(I)]
    BW = 1
    block_pad = np.arange(I, dtype=np.int64).reshape(I, 1)
    block_msk = np.ones((I, 1), np.float32)

indices_u2 = np.array([s for seg in _blocks for s in seg])
n_u2 = len(indices_u2)
dim_u1 = NB
dim_z = dim_z_unit * NB
CKDIR = os.environ.get("CKDIR", "checkpoint/cgkn")
os.makedirs(CKDIR, exist_ok=True)

_nbr = np.stack([(np.arange(NB) - 1) % NB, np.arange(NB), (np.arange(NB) + 1) % NB], 1)


def load_flat(split):
    a = np.load(f"{DATA_DIR}/{split}.npy")
    return torch.tensor(a.reshape(a.shape[0], a.shape[1], I), dtype=torch.float32)


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
        g2 = torch.zeros(B, dim_z, dim_z, device=x.device)
        for k in range(3):
            for b in range(NB):
                r0 = b * dzu
                c0 = int(self.nbr[b, k]) * dzu
                g2[:, r0 : r0 + dzu, c0 : c0 + dzu] += a2[:, b, :, k, :]
        return [f1.reshape(B, NB, 1), g1, f2, g2]


class CGKN(nn.Module):
    def __init__(self, ae, cgn):
        super().__init__()
        self.autoencoder = ae
        self.cgn = cgn

    def forward(self, t, u_ext):
        u1 = u_ext[:, :NB]
        z = u_ext[:, NB:].unsqueeze(-1)
        f1, g1, f2, g2 = self.cgn(u1)
        return torch.cat([(f1 + g1 @ z).squeeze(-1), (f2 + g2 @ z).squeeze(-1)], dim=-1)


_bp = torch.tensor(block_pad)
_bm = torch.tensor(block_msk)


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


_FILTER = os.environ.get("FILTER", "discrete")


def CGFilter(cgkn, sigma, u1, mu0, R0, dt):
    dev = u1.device
    Nt = u1.shape[0]
    d1 = u1.shape[1]
    dz = mu0.shape[0]
    if _FILTER == "discrete":
        s1sq = torch.diag(sigma[:d1] ** 2) * dt
        s2sq = torch.diag(sigma[d1:] ** 2) * dt
        Idz = torch.eye(dz, device=dev)
        mu = torch.zeros((Nt, dz, 1), device=dev)
        R = torch.zeros((Nt, dz, dz), device=dev)
        mu[0], R[0] = mu0, R0
        m, Rc = mu0, R0
        for n in range(1, Nt):
            f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(u1[n - 1].T)]
            F = Idz + g2 * dt
            c = f2 * dt
            H = g1 * dt
            off = f1 * dt
            y = u1[n] - u1[n - 1]
            mp = F @ m + c
            Rp = F @ Rc @ F.T + s2sq
            S = H @ Rc @ H.T + s1sq
            cross = F @ Rc @ H.T
            K = cross @ torch.linalg.inv(S)
            m = mp + K @ (y - (H @ m + off))
            Ra = Rp - K @ cross.T
            Rc = 0.5 * (Ra + Ra.T)
            mu[n], R[n] = m, Rc
        return mu, R
    s1, s2 = torch.diag(sigma[:d1]), torch.diag(sigma[d1:])
    invs1os1 = torch.linalg.inv(s1 @ s1.T)
    s2os2 = s2 @ s2.T
    mu = torch.zeros((Nt, dz, 1), device=dev)
    R = torch.zeros((Nt, dz, dz), device=dev)
    mu[0], R[0] = mu0, R0
    for n in range(1, Nt):
        f1, g1, f2, g2 = [e.squeeze(0) for e in cgkn.cgn(u1[n - 1].T)]
        du1 = u1[n] - u1[n - 1]
        mu1 = (
            mu0
            + (f2 + g2 @ mu0) * dt
            + (R0 @ g1.T) @ invs1os1 @ (du1 - (f1 + g1 @ mu0) * dt)
        )
        R1 = R0 + (g2 @ R0 + R0 @ g2.T + s2os2 - R0 @ g1.T @ invs1os1 @ g1 @ R0) * dt
        mu[n], R[n] = mu1, R1
        mu0, R0 = mu1, R1
    return mu, R


def da_nrmse(u2_pred_blocks, u_true, dev):
    bm = _bm.to(dev).bool()
    pred = u2_pred_blocks[..., bm]
    tgt = true_blocks(u_true)[..., bm]
    return torch.mean(
        torch.sqrt(torch.mean((pred - tgt) ** 2, 0) / torch.var(tgt, 0))
    ).item()


@torch.no_grad()
def estimate_sigma(cgkn, train_u, dev):
    U = train_u[:8].reshape(-1, I).to(dev)
    z = encode_u2(cgkn.autoencoder, U)
    ext = torch.cat([U[:, indices_u1], z], dim=-1)
    ed = torch.diff(ext, dim=0) / DT
    pred = cgkn(None, ext)
    sig = torch.sqrt(DT * torch.mean((ed - pred[:-1]) ** 2, dim=0))
    sig[dim_u1:] = 1.0
    return sig


@torch.no_grad()
def test_da(cgkn, sigma, test_u, dev, cut):
    errs = []
    for k in range(test_u.shape[0]):
        u = test_u[k].to(dev)
        mu, _ = CGFilter(
            cgkn,
            sigma,
            u[:, indices_u1].unsqueeze(-1),
            torch.zeros(dim_z, 1, device=dev),
            0.01 * torch.eye(dim_z, device=dev),
            DT,
        )
        blk = decode_blocks(cgkn.autoencoder, mu.squeeze(-1))
        errs.append(da_nrmse(blk[cut:], u[cut:], dev))
    return float(np.mean(errs)), float(np.std(errs))


def stage1(dev):
    tr = load_flat("train")
    us, ud = [], []
    for k in range(tr.shape[0]):
        u = tr[k]
        us.append(u[:-1])
        ud.append(torch.diff(u, dim=0) / DT)
    U, Ud = torch.cat(us), torch.cat(ud)
    print(
        f"[s1 obs={NB}/40={NB/40*100:.0f}%] BW={BW} n_u2={n_u2} dz={dim_z}", flush=True
    )
    ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(U, Ud),
        shuffle=True,
        batch_size=200,
        drop_last=True,
    )
    ae = AutoEncoder(BW, dim_z_unit)
    cgn = CGN()
    m = CGKN(ae, cgn).to(dev)
    bm = _bm.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(1, int(os.environ.get("S1_EPOCHS", 500)) + 1):
        t0 = time.time()
        lf = la = lz = 0.0
        for u, d in ld:
            u, d = u.to(dev), d.to(dev)
            opt.zero_grad()
            blk = true_blocks(u)
            z = ae.encoder(blk)
            la_ = (((ae.decoder(z) - blk) * bm) ** 2).sum() / bm.sum() / u.shape[0]
            ext = torch.cat([u[:, indices_u1], z.reshape(u.shape[0], -1)], dim=-1)
            extp = ext + m(None, ext) * DT
            up = assemble(
                extp[:, :NB],
                ae.decoder(extp[:, NB:].reshape(u.shape[0], NB, dim_z_unit)),
            )
            lf_ = nnF.mse_loss(u + d * DT, up)
            zn = ae.encoder(true_blocks(u + d * DT)).reshape(u.shape[0], -1)
            lz_ = nnF.mse_loss(zn, extp[:, NB:])
            (lf_ + la_ + lz_).backward()
            opt.step()
            lf += lf_.item()
            la += float(la_)
            lz += lz_.item()
        n = len(ld)
        if ep % 20 == 0 or ep == 1:
            print(
                f"{ep:4d} t{time.time()-t0:5.1f} fore {lf/n:.4f} ae {la/n:.4f} fz {lz/n:.4f}",
                flush=True,
            )
    torch.save(m.state_dict(), f"{CKDIR}/stage1.pt")
    sig = estimate_sigma(m, tr, dev)
    torch.save(sig.cpu(), f"{CKDIR}/sigma.pt")
    mn, sd = test_da(m, sig, load_flat("val"), dev, int(2.0 / DT))
    print(f"[s1] val {mn:.4f} {sd:.4f}", flush=True)


def stage2(dev):
    tr = load_flat("train").to(dev)
    ae = AutoEncoder(BW, dim_z_unit)
    cgn = CGN()
    m = CGKN(ae, cgn).to(dev)
    sigma = torch.load(f"{CKDIR}/sigma.pt", map_location=dev)
    print("[s2] from-scratch", flush=True)
    bm = _bm.to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    short = max(2, int(0.2 / DT))
    long_s = min(int(os.environ.get("S2_LONG", 1000)), tr.shape[1] - 1)
    cut = int(2.0 / DT)
    Ntr = tr.shape[0]
    ts = torch.arange(short, device=dev) * DT
    best = 1e9
    for ep in range(1, int(os.environ.get("S2_EPOCHS", 2500)) + 1):
        t0 = time.time()
        opt.zero_grad()
        ks = torch.randint(0, Ntr, (200,))
        ss = torch.randint(0, tr.shape[1] - short, (200,))
        us = torch.stack([tr[k, s : s + short] for k, s in zip(ks, ss)], dim=1)
        z0 = encode_u2(ae, us[0])
        e0 = torch.cat([us[0, :, indices_u1], z0], dim=-1)
        ep_ = torchdiffeq.odeint(m, e0, ts, method="rk4", options={"step_size": 0.01})
        lz = nnF.mse_loss(encode_u2(ae, us), ep_[:, :, NB:])
        up = assemble(ep_[:, :, :NB], decode_blocks(ae, ep_[:, :, NB:]))
        lf = nnF.mse_loss(us, up)
        kl = int(torch.randint(0, Ntr, (1,)).item())
        ul = tr[kl, :long_s]
        blk = true_blocks(ul)
        la = (((ae.decoder(ae.encoder(blk)) - blk) * bm) ** 2).sum() / bm.sum() / long_s
        mu = CGFilter(
            m,
            sigma,
            ul[:, indices_u1].unsqueeze(-1),
            torch.zeros(dim_z, 1, device=dev),
            0.01 * torch.eye(dim_z, device=dev),
            DT,
        )[0].squeeze(-1)[cut:]
        blkda = decode_blocks(ae, mu)
        lda = (
            (((blkda - true_blocks(ul)[cut:]) * bm) ** 2).sum()
            / bm.sum()
            / (long_s - cut)
        )
        loss = lf + lda + la + lz
        if torch.isnan(loss):
            print(ep, "nan; skip", flush=True)
            continue
        loss.backward()
        opt.step()
        if ep % 50 == 0 or ep == 1:
            print(
                f"{ep:5d} t{time.time()-t0:5.1f} fore {lf.item():.4f} da {float(lda):.4f} ae {float(la):.4f}",
                flush=True,
            )
        if ep % 200 == 0:
            with torch.no_grad():
                mn, sd = test_da(m, sigma, load_flat("val"), dev, cut)
            print(f"    [ep{ep}] val {mn:.4f} {sd:.4f}", flush=True)
            if mn < best:
                best = mn
                torch.save(m.state_dict(), f"{CKDIR}/stage2.pt")
                print(f"    [ep{ep}] new best {best:.4f}", flush=True)
    print(f"[s2 obs={NB}/40] val {best:.4f}", flush=True)


def evaluate_only(dev):
    ae = AutoEncoder(BW, dim_z_unit)
    cgn = CGN()
    m = CGKN(ae, cgn).to(dev)
    sigma = torch.load(f"{CKDIR}/sigma.pt", map_location=dev)
    m.load_state_dict(torch.load(f"{CKDIR}/stage2.pt", map_location=dev))
    m.eval()
    cut = 100
    with torch.no_grad():
        mn, sd = test_da(m, sigma, load_flat("test"), dev, cut)
    print(f"[eval obs={NB}/40] test {mn:.4f} {sd:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["1", "2", "eval"], default="1")
    _seed = int(os.environ.get("CGKN_SEED", 0))
    a = ap.parse_args()
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[cgkn-band] obs={NB}/40 ({NB/40*100:.0f}%) BW={BW} n_u2={n_u2} dz={dim_z} DT={DT} dev={dev}",
        flush=True,
    )
    {"1": stage1, "2": stage2, "eval": evaluate_only}[a.stage](dev)


if __name__ == "__main__":
    main()
