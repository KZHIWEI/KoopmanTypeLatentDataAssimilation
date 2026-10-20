from __future__ import annotations

import os

import numpy as np
import torch

from src.assimilation.variational import _rrmse_batch
from src.assimilation.observation import MaskSampler


@torch.no_grad()
def evaluate(
    dbf,
    runs,
    sampler: MaskSampler,
    spinup: int,
    run_length: int,
    device,
    return_fields: bool = False,
):
    B = len(runs)
    gt = [
        torch.stack([runs[b][t] for b in range(B)]).to(device)
        for t in range(run_length + 1)
    ]
    T = run_length + 1
    obs = []
    for t in range(T):
        if t == 0 and os.environ.get("DBF_TRUTH_INIT", "1") == "1":
            obs.append(gt[t].reshape(B, -1))
            continue
        masks = torch.stack([sampler.mask(b, t) for b in range(B)]).to(device)
        obs.append((gt[t] * masks).reshape(B, -1))
    obs = torch.stack(obs, dim=1).float()
    dbf.bench_obs_every = int(getattr(dbf, "bench_obs_every", 1))
    _, _, mu_t, _ = dbf._compute_mu_t_sigma_t(obs, T, B, 1, spinup=spinup)
    z = mu_t.reshape(B * T, -1)
    dec = dbf.h_network(z).reshape(B, T, -1)
    errs = []
    for t in range(run_length):
        errs.append(
            _rrmse_batch(dec[:, t].reshape(B, *gt[t].shape[1:]), gt[t]).cpu().numpy()
        )
    if return_fields:
        fld = dec[:, :run_length].reshape(B, run_length, *gt[0].shape[1:])
        return np.stack(errs, axis=1), fld.float().cpu().numpy()
    return np.stack(errs, axis=1)
