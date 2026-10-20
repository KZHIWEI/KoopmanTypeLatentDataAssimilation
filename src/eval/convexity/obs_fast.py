from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.eval.convexity.hessian import hvp_fn


def obs_term_at_step(m, masks, obs, sr, k):
    def f(z):
        dec = m.decode(z).reshape(z.shape[0], *obs.shape[2:])
        return ((masks[:, k] * dec - obs[:, k]) ** 2).sum() / sr**2

    return f


def _symmetry(h, D, dev, dtype, gen, pairs=3):
    worst = 0.0
    for _ in range(pairs):
        v = torch.randn((1, D), generator=gen).to(dev).to(dtype)
        w = torch.randn((1, D), generator=gen).to(dev).to(dtype)
        Hv, Hw = h(v), h(w)
        num = (Hv * w).sum() - (Hw * v).sum()
        den = (Hv.norm() * w.norm()).clamp_min(1e-30)
        worst = max(worst, float((num.abs() / den).item()))
    return worst


def ends(m, centre, masks, obs, sr, obs_idx, dev, seed=1234, tol=0.0):
    B, n, D = centre.shape
    lo = torch.zeros(B, dtype=torch.float64)
    hi = torch.zeros(B, dtype=torch.float64)
    asym = torch.zeros(B, dtype=torch.float64)
    calls = [0]
    gen = torch.Generator(device="cpu").manual_seed(seed)

    for b in range(B):
        vlo, vhi, va = [], [], []
        if len(obs_idx) < n:
            vlo.append(0.0)
            vhi.append(0.0)
        for k, t in enumerate(obs_idx):
            z = centre[b : b + 1, t].contiguous()
            h = hvp_fn(obs_term_at_step(m, masks[b : b + 1], obs[b : b + 1], sr, k), z)

            def mv(x, _h=h):
                calls[0] += 1
                y = _h(torch.as_tensor(x, device=dev, dtype=centre.dtype).reshape(1, D))
                return y.reshape(-1).double().cpu().numpy()

            op = LinearOperator((D, D), matvec=mv, dtype=np.float64)
            vlo.append(
                float(eigsh(op, k=1, which="SA", tol=tol, return_eigenvectors=False)[0])
            )
            vhi.append(
                float(eigsh(op, k=1, which="LA", tol=tol, return_eigenvectors=False)[0])
            )
            va.append(_symmetry(h, D, dev, centre.dtype, gen))
        lo[b], hi[b] = min(vlo), max(vhi)
        asym[b] = max(va) if va else 0.0
    return lo.to(dev), hi.to(dev), asym.to(dev), calls[0]
