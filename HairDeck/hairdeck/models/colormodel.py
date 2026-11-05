"""Per-card hair colour: the image model's network fine-tuned to read the colours of the generated cards off the image.

Input: the clean card tokens of a generated hairstyle (hairdeck.cards.tokens.clean_tokens, then sign_token_normals) at
t = 0, the DINOv3 tokens of the model's 448 px crop, and per 16 px patch of the crop its mean RGB, the fraction of the
character (the person matte of the preprocessing) and the mean RGB under the matte, added to the patch tokens of the
context (DINOv3 features keep little exact colour). Output: per card 16 CIELAB colours (D65), one per sixteenth of the
card's length from root to tip (the v of hairdeck.cards.mesh.card_uv), in the order of the card tokens.
Trained on renders of the training hairstyles with the colours of their VRM materials, starting from the image model
(scripts/train/train_color_dit.py): route 'reg1' (the released model) predicts one colour per bin, the colour the card
shows in the image's view (hairdeck.color.targets.view_targets); route 'reg' the bin's lit and shade colour
(card_targets), the first stage of the released model. The new layers (patch statistics, colour head) start at zero.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ckpt import load as load_ckpt
from .imgmodel import build_img_model
from .setdit import modulate, timestep_embedding

BINS = 16  # colours per card along v


class ColorDiT(nn.Module):
    def __init__(self, tok_dim, a, mu=(0.0, 0.0, 0.0), sd=(1.0, 1.0, 1.0), route='reg1'):
        """a: run args with dim, depth, heads, dino, res (those of the image model it starts from); mu, sd: the Lab
        standardisation of the output; route: 'reg1' (one colour per bin) or 'reg' (lit and shade per bin)"""
        super().__init__()
        self.route = route
        self.dit = build_img_model(a, tok_dim)
        d = a.dim
        self.patch = nn.Linear(7, d)  # per patch: mean RGB, matte fraction, mean RGB under the matte
        self.col_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.col_ada = nn.Linear(d, 2 * d)
        self.col_out = nn.Linear(d, BINS * (3 if route == 'reg1' else 6))
        for m in (self.patch, self.col_ada, self.col_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.register_buffer('mu', torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.as_tensor(sd, dtype=torch.float32))

    def encode_ctx(self, tokens, img, matte):
        """DINOv3 tokens (B, 789, 1024), img float 0..255 (B, 3, res, res), matte bool (B, res, res) -> context"""
        ctx = self.dit.encode_cond(tokens)
        x = img / 255
        m = matte[:, None].float()
        frac = F.avg_pool2d(m, 16)
        st = torch.cat([F.avg_pool2d(x, 16), frac, F.avg_pool2d(x * m, 16) / frac.clamp(min=1e-3)], 1)
        return torch.cat([ctx[:, :5], ctx[:, 5:] + self.patch(st.flatten(2).transpose(1, 2))], 1)

    def raw(self, x, ctx):
        """x: card tokens (B, N, token dim) -> the standardised colours (B, N, BINS * 3 or 6), fp32"""
        dit = self.dit
        c = dit.t_mlp(timestep_embedding(torch.zeros(len(x), device=x.device), 256)) + dit.pool_mlp(ctx.mean(1))
        h = dit.inp(x)
        for blk in dit.blocks:
            h = blk(h, c, ctx)
        with torch.autocast('cuda', enabled=False):
            sh, sc = self.col_ada(F.silu(c.float())).chunk(2, -1)
            return self.col_out(modulate(self.col_norm(h.float()), sh, sc))

    def unstd(self, y):
        """standardised colours -> (lit, shade) Lab (B, N, BINS, 3); reg1: the one colour twice"""
        y = y.view(*y.shape[:-1], BINS, -1)
        if y.shape[-1] == 3:
            c = y * self.sd + self.mu
            return c, c
        return y[..., :3] * self.sd + self.mu, y[..., 3:] * self.sd + self.mu

    @torch.no_grad()
    def forward(self, x, ctx):
        """x: clean card tokens (B, N, token dim) -> Lab (B, N, BINS, 3) (reg: the lit colour)"""
        return self.unstd(self.raw(x, ctx))[0]


def load_color_dit(path, tok_dim, device='cuda'):
    """(model, run args) of a colour model checkpoint (EMA weights; ema_XXXXXX.pt or model.safetensors)"""
    ck = load_ckpt(path)
    a = argparse.Namespace(**ck['args'])
    sd = ck['ema']
    model = ColorDiT(tok_dim, a, sd['mu'], sd['sd'], getattr(a, 'route', 'reg1')).to(device).eval()
    model.load_state_dict(sd)
    return model, a


def lab_to_srgb(lab):
    """CIELAB (D65, (..., 3)) -> sRGB 0..1, clipped (skimage.color.lab2rgb)"""
    from skimage.color import lab2rgb
    lab = np.asarray(lab, np.float64)
    return lab2rgb(lab.reshape(-1, 1, 3)).reshape(lab.shape)
