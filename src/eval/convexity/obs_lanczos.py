from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.eval.convexity.hessian import hvp_fn
from src.eval.convexity.obs_fast import obs_term_at_step


def _symmetry_batched(h, B, D, dev, dtype, gen, pairs=2):
    worst = 0.0
    for _ in range(pairs):
        v = torch.randn((B, D), generator=gen).to(dev).to(dtype)
        w = torch.randn((B, D), generator=gen).to(dev).to(dtype)
        Hv, Hw = h(v), h(w)
        num = ((Hv * w).sum(-1) - (Hw * v).sum(-1)).abs()
        den = (Hv.norm(dim=-1) * w.norm(dim=-1)).clamp_min(1e-30)
        worst = max(worst, float((num / den).max().item()))
    return worst


def lanczos_ends(h, B, D, m, dev, dtype, gen, reorth=True):
    m = min(m, D)
    Q = torch.zeros((B, m + 1, D), device=dev, dtype=dtype)
    al = torch.zeros((B, m), device=dev, dtype=torch.float64)
    be = torch.zeros((B, m), device=dev, dtype=torch.float64)
    q = torch.randn((B, D), generator=gen).to(dev).to(dtype)
    q /= q.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    Q[:, 0] = q
    used = 0
    for j in range(m):
        w = h(Q[:, j])
        al[:, j] = (w * Q[:, j]).sum(-1).double()
        w = w - al[:, j, None].to(dtype) * Q[:, j]
        if j > 0:
            w = w - be[:, j - 1, None].to(dtype) * Q[:, j - 1]
        if reorth:
            for _ in range(2):
                c = torch.einsum("bkd,bd->bk", Q[:, : j + 1], w)
                w = w - torch.einsum("bk,bkd->bd", c, Q[:, : j + 1])
        nb = w.norm(dim=-1).double()
        be[:, j] = nb
        used = j + 1
        if float(nb.max()) <= 1e-12:
            break
        Q[:, j + 1] = w / nb[:, None].to(dtype).clamp_min(1e-30)
    k = used
    T = torch.zeros((B, k, k), device=dev, dtype=torch.float64)
    idx = torch.arange(k, device=dev)
    T[:, idx, idx] = al[:, :k]
    if k > 1:
        T[:, idx[:-1], idx[1:]] = be[:, : k - 1]
        T[:, idx[1:], idx[:-1]] = be[:, : k - 1]
    ev = torch.linalg.eigvalsh(T)
    return ev[:, 0], ev[:, -1], k


def ends(m_model, centre, masks, obs, sr, obs_idx, dev, iters=200, seed=1234):
    B, n, D = centre.shape
    gen = torch.Generator(device="cpu").manual_seed(seed)
    los, his, asyms, total = [], [], [], 0
    if len(obs_idx) < n:
        los.append(torch.zeros(B, device=dev, dtype=torch.float64))
        his.append(torch.zeros(B, device=dev, dtype=torch.float64))
    for k, t in enumerate(obs_idx):
        h = hvp_fn(
            obs_term_at_step(m_model, masks, obs, sr, k), centre[:, t].contiguous()
        )
        lo, hi, used = lanczos_ends(h, B, D, iters, dev, centre.dtype, gen)
        los.append(lo)
        his.append(hi)
        total += used
        asyms.append(_symmetry_batched(h, B, D, dev, centre.dtype, gen))
    lo = torch.stack(los).min(0).values
    hi = torch.stack(his).max(0).values
    asym = torch.full(
        (B,), max(asyms) if asyms else 0.0, device=dev, dtype=torch.float64
    )
    return lo, hi, asym, total
