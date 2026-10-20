from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.ae import AEConfig, ViTEncoder, ViTDecoder


def batch_pinv(z_seq: torch.Tensor, i_factor: float = 1e-2) -> torch.Tensor:
    B, T, D = z_seq.size()
    trans = T < D
    if trans:
        z_seq = z_seq.transpose(1, 2)
        T, D = D, T
    z_t = z_seq.transpose(1, 2)
    I = torch.eye(D, device=z_seq.device, dtype=z_seq.dtype).expand(B, D, D)
    pinv = torch.linalg.solve(torch.bmm(z_t, z_seq) + i_factor * I, z_t)
    if trans:
        pinv = pinv.transpose(1, 2)
    return pinv


class TensorVarForward(nn.Module):

    def __init__(self, cfg: AEConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.hidden_dim = cfg.latent_dim
        self.phi_S = ViTEncoder(cfg)
        self.phi_inv_S = ViTDecoder(cfg)
        self.register_buffer("C_forward", torch.eye(self.hidden_dim), persistent=True)

    def encode(self, x):
        return self.phi_S(x)

    def decode(self, z):
        return self.phi_inv_S(z)

    def step(self, z):
        return z @ self.C_forward

    def spectral_radius(self) -> float:
        return float(
            torch.linalg.eigvals(self.C_forward.detach().float().cpu()).abs().max()
        )

    def _encode_seq(self, seq):
        B, T = seq.shape[:2]
        z = self.phi_S(seq.reshape(B * T, *seq.shape[2:]))
        return z.reshape(B, T, -1)

    def compute_loss(self, state_seq, state_next_seq):
        B, T = state_seq.shape[:2]
        z_seq = self._encode_seq(state_seq)
        z_next_seq = self._encode_seq(state_next_seq)
        with torch.autocast(device_type=z_seq.device.type, enabled=False):
            z_seq, z_next_seq = z_seq.float(), z_next_seq.float()
            C_fwd = torch.bmm(batch_pinv(z_seq), z_next_seq).mean(0)
            self.C_forward = C_fwd.detach()
            z_pred = z_seq @ C_fwd

        flat = lambda t: t.reshape(B * T, -1)
        s = flat(state_seq)
        recon = self.phi_inv_S(z_seq.reshape(B * T, -1)).reshape(B * T, -1)
        losses = {"recon": F.mse_loss(recon, s), "pred": F.mse_loss(z_pred, z_next_seq)}
        return losses, C_fwd

    def load_pretrained_ae(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = {
            k.replace("_orig_mod.", ""): v for k, v in ck.get("state_dict", ck).items()
        }
        self.phi_S.load_state_dict(
            {k[8:]: v for k, v in sd.items() if k.startswith("encoder.")}
        )
        self.phi_inv_S.load_state_dict(
            {k[8:]: v for k, v in sd.items() if k.startswith("decoder.")}
        )

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class TensorVarInverse(nn.Module):

    def __init__(
        self,
        cfg: AEConfig,
        forward_model: TensorVarForward,
        history_len: int = 10,
        obs_ratio: float = 0.15,
        nu: float = 1.0,
        hist_obs_every: int = 1,
        hist_include_current: bool = False,
        hist_mask_kind: str = "random",
        mask_aware: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.history_len = history_len
        self.obs_ratio = obs_ratio
        self.nu = nu
        self.hist_obs_every = hist_obs_every
        self.hist_include_current = hist_include_current
        self.hist_mask_kind = hist_mask_kind
        self.mask_aware = bool(mask_aware)
        _in_ch = cfg.in_channels * history_len + (history_len if mask_aware else 0)
        obs_cfg = dataclasses.replace(cfg, in_channels=_in_ch)
        self.K_O = ViTEncoder(obs_cfg)
        self.phi_S = forward_model.phi_S
        self.phi_inv_S = forward_model.phi_inv_S
        for p in self.phi_S.parameters():
            p.requires_grad_(False)
        for p in self.phi_inv_S.parameters():
            p.requires_grad_(False)

    def make_obs_history(self, states, seed: int | None = None):
        B, T, C, H, W = states.shape
        g = (
            torch.Generator(device=states.device).manual_seed(int(seed))
            if seed is not None
            else None
        )
        if self.hist_mask_kind == "stride":
            k = max(1, int(round(1.0 / self.obs_ratio)))
            m0 = torch.zeros(H * W, device=states.device)
            m0[::k] = 1.0
            mask = m0.view(1, 1, 1, H, W).expand(B, T, 1, H, W)
        elif self.hist_mask_kind == "fixed":
            from src.assimilation.observation import build_mask

            m0 = build_mask([H, W], self.obs_ratio, "random", 0).to(states.device)
            mask = m0.view(1, 1, 1, H, W).expand(B, T, 1, H, W)
        else:
            mask = (
                torch.rand(B, T, 1, H, W, device=states.device, generator=g)
                < self.obs_ratio
            ).float()
        obs = states * mask
        oe = self.hist_obs_every
        if oe > 1:
            phase = torch.randint(0, oe, (B,), device=states.device, generator=g)
            keep = (
                torch.arange(T, device=states.device)[None, :] + phase[:, None]
            ) % oe == 0
            obs = obs * keep[:, :, None, None, None].float()
        m = self.history_len
        zero = torch.zeros(B, C, H, W, device=states.device)
        ks = range(m - 1, -1, -1) if self.hist_include_current else range(m, 0, -1)
        hist = [
            torch.cat([obs[:, i - k] if i - k >= 0 else zero for k in ks], dim=1)
            for i in range(T)
        ]
        out = torch.stack(hist, dim=1)
        if self.mask_aware:
            eff = mask.expand(B, T, 1, H, W).clone()
            if oe > 1:
                eff = eff * keep[:, :, None, None, None].float()
            zero1 = torch.zeros(B, 1, H, W, device=states.device)
            mhist = [
                torch.cat([eff[:, i - k] if i - k >= 0 else zero1 for k in ks], dim=1)
                for i in range(T)
            ]
            out = torch.cat([out, torch.stack(mhist, dim=1)], dim=2)
        return out

    def observe(self, obs_hist):
        return self.K_O(obs_hist)

    def compute_loss(self, state_seq):
        B, T = state_seq.shape[:2]
        hist = self.make_obs_history(state_seq)
        z_pred = self.K_O(hist.reshape(B * T, *hist.shape[2:]))
        with torch.no_grad():
            z_true = self.phi_S(state_seq.reshape(B * T, *state_seq.shape[2:]))
        loss_feat = F.mse_loss(z_pred, z_true)
        dec = self.phi_inv_S(z_pred).reshape(B * T, -1)
        loss_state = F.mse_loss(dec, state_seq.reshape(B * T, -1))
        return {
            "feat": loss_feat,
            "state": loss_state,
            "total": loss_feat + self.nu * loss_state,
        }

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
