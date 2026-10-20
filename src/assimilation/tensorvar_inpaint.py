from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.tensorvar import TensorVarInverse

from src.config import build_ae_config
from src.model.tensorvar import TensorVarForward
from src.assimilation import tensorvar as RT


def _pad(x, p):
    x = F.pad(x, (0, 0, p, p), mode="circular")
    return F.pad(x, (p, p, 0, 0), mode="replicate")


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, groups=8):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3)
        self.c2 = nn.Conv2d(cout, cout, 3)
        self.n1 = nn.GroupNorm(min(groups, cout), cout)
        self.n2 = nn.GroupNorm(min(groups, cout), cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(_pad(x, 1))))
        h = self.n2(self.c2(_pad(h, 1)))
        return F.silu(h + self.skip(x))


class Attn(nn.Module):

    def __init__(self, ch, n_tokens, heads=8, depth=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            ch,
            heads,
            ch * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, depth)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, ch))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.tr(x.flatten(2).transpose(1, 2) + self.pos)
        return t.transpose(1, 2).reshape(B, C, H, W)


class Inpainter(nn.Module):

    def __init__(self, in_ch, out_ch, image_size=(64, 32), width=96, n_attn=4):
        super().__init__()
        w = width
        n_tok = (image_size[0] // 4) * (image_size[1] // 4)
        self.stem = ConvBlock(in_ch, w)
        self.d1 = ConvBlock(w, w * 2)
        self.d2 = ConvBlock(w * 2, w * 4)
        self.mid = ConvBlock(w * 4, w * 4)
        self.attn = Attn(w * 4, n_tok, heads=8, depth=n_attn)
        self.u2 = ConvBlock(w * 4 + w * 2, w * 2)
        self.u1 = ConvBlock(w * 2 + w, w)
        self.head = nn.Conv2d(w, out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.d1(F.avg_pool2d(s0, 2))
        s2 = self.d2(F.avg_pool2d(s1, 2))
        m = self.attn(self.mid(s2))
        u = self.u2(
            torch.cat([F.interpolate(m, size=s1.shape[-2:], mode="nearest"), s1], 1)
        )
        u = self.u1(
            torch.cat([F.interpolate(u, size=s0.shape[-2:], mode="nearest"), s0], 1)
        )
        return self.head(u)


class KOInpaint(nn.Module):

    def __init__(self, in_ch, out_ch, phi, image_size=(64, 32), width=96, n_attn=4):
        super().__init__()
        self.net = Inpainter(in_ch, out_ch, image_size, width, n_attn)
        object.__setattr__(self, "_phi", phi)

    def field(self, hist):
        return self.net(hist)

    def forward(self, hist):
        return self._phi(self.net(hist))


class InverseInpaint(TensorVarInverse):

    def __init__(self, cfg, forward_model, width=96, n_attn=4, w_pix=1.0, **kw):
        super().__init__(cfg, forward_model, **kw)
        in_ch = cfg.in_channels * self.history_len + (
            self.history_len if self.mask_aware else 0
        )
        self.K_O = KOInpaint(
            in_ch, cfg.in_channels, self.phi_S, cfg.image_size, width, n_attn
        )
        self.w_pix = float(w_pix)

    def compute_loss(self, state_seq):
        B, T = state_seq.shape[:2]
        hist = self.make_obs_history(state_seq)
        flat = hist.reshape(B * T, *hist.shape[2:])
        s_hat = self.K_O.field(flat)
        x = state_seq.reshape(B * T, *state_seq.shape[2:])
        loss_pix = F.mse_loss(s_hat, x)
        z_pred = self.phi_S(s_hat)
        with torch.no_grad():
            z_true = self.phi_S(x)
        loss_feat = F.mse_loss(z_pred, z_true)
        dec = self.phi_inv_S(z_pred).reshape(B * T, -1)
        loss_state = F.mse_loss(dec, x.reshape(B * T, -1))
        return {
            "feat": loss_feat,
            "state": loss_state,
            "pix": loss_pix,
            "total": loss_feat + self.nu * loss_state + self.w_pix * loss_pix,
        }


def _tail_loss(self, state_seq, n_sup, body):
    hist = self.make_obs_history(state_seq)
    B, T = state_seq.shape[:2]
    hist = hist[:, T - n_sup :]
    tgt = state_seq[:, T - n_sup :]
    n = B * n_sup
    return body(hist.reshape(n, *hist.shape[2:]), tgt.reshape(n, *tgt.shape[2:]))




class InverseInpaintFullHist(InverseInpaint):

    def __init__(self, cfg, forward_model, n_sup: int, **kw):
        super().__init__(cfg, forward_model, **kw)
        self.n_sup = int(n_sup)

    def compute_loss(self, state_seq):
        def body(flat_hist, x):
            s_hat = self.K_O.field(flat_hist)
            lp = F.mse_loss(s_hat, x)
            z_pred = self.phi_S(s_hat)
            with torch.no_grad():
                z_true = self.phi_S(x)
            lf = F.mse_loss(z_pred, z_true)
            ls = F.mse_loss(
                self.phi_inv_S(z_pred).reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
            )
            return {
                "feat": lf,
                "state": ls,
                "pix": lp,
                "total": lf + self.nu * ls + self.w_pix * lp,
            }

        return _tail_loss(self, state_seq, self.n_sup, body)


class TVDAInpaint(RT.TVDA):
    def __init__(self, fwd_ckpt, inv_ckpt, device, width=96, n_attn=4):
        ck_f = torch.load(fwd_ckpt, map_location="cpu", weights_only=False)
        fwd = TensorVarForward(build_ae_config(ck_f["cfg"]))
        fwd.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in ck_f["state_dict"].items()}
        )
        ck_i = torch.load(inv_ckpt, map_location="cpu", weights_only=False)
        icfg = ck_i["cfg"]
        inv = InverseInpaint(
            build_ae_config(icfg),
            fwd,
            width=width,
            n_attn=n_attn,
            history_len=int(icfg.get("history_len", 10)),
            obs_ratio=float(icfg.get("obs_ratio", 0.15)),
            hist_include_current=bool(icfg.get("hist_include_current", False)),
            hist_obs_every=int(icfg.get("hist_obs_every", 1)),
            mask_aware=bool(icfg.get("mask_aware", False)),
        )
        before = RT.snapshot_shared_ae(fwd)
        missing, unexpected = inv.load_state_dict(
            {k.replace("_orig_mod.", ""): v for k, v in ck_i["state_dict"].items()},
            strict=False,
        )
        assert not [k for k in unexpected], unexpected[:5]
        assert not [k for k in missing if k.startswith("K_O.")], missing[:5]
        RT.assert_shared_ae_unmoved(fwd, before, inv_ckpt, fwd_ckpt)
        for p in fwd.parameters():
            p.requires_grad_(False)
        for p in inv.parameters():
            p.requires_grad_(False)
        self.fwd, self.inv = fwd.to(device).eval(), inv.to(device).eval()
        self.device, self.train_cfg = device, ck_f["cfg"]
        self.history_len = inv.history_len
        self.C = fwd.C_forward.detach().t().to(device)
        self.D = self.C.shape[0]
