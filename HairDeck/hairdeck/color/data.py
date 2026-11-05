"""Training data of the colour model (scripts/train/train_color_dit.py): the render views as the image condition,
the card tokens of the hairstyle (head frame, as the image DiT's training targets) and the colour targets of
hairdeck.color.targets (data/color/: targets_{split}.npz, views/{split}_*.npz, scripts/data.py --steps colors).
Augmentations: the training crop of hairdeck.data.crop with its jitter, brightness / contrast (on the image only, or on
the image and the targets together on the GPU: photo_aug), blur, JPEG; on the GPU a hue rotation of the image and the
targets together; card dropout and token noise (the gap between dataset and generated cards).
"""
import glob
import io
import json
import os

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .. import paths
from ..data.crop import LAB_FACE, LAB_HAIR, TRAIN_CROP, apply_crop, composite, crop_box
from .targets import BINS, VIEWS

RES, NMAX = 448, 192
RENDERS = {'train': 'renders', 'test': 'renders_test'}

# sRGB (D65) <-> CIELAB on the GPU, as skimage.color
M_XYZ = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
WHITE = torch.tensor([0.95047, 1.0, 1.08883])


def rgb2lab(x):
    """x (..., 3) sRGB in [0, 1] -> Lab"""
    lin = torch.where(x > 0.04045, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4, x / 12.92)
    xyz = lin @ M_XYZ.to(x).T / WHITE.to(x)
    f = torch.where(xyz > 0.008856, xyz.clamp(min=1e-12) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab2rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    f = torch.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = torch.where(f > 0.2069, f ** 3, (f - 16 / 116) / 7.787) * WHITE.to(lab)
    lin = (xyz @ torch.linalg.inv(M_XYZ).to(lab).T).clamp(min=0)
    return torch.where(lin > 0.0031308, 1.055 * lin ** (1 / 2.4) - 0.055, 12.92 * lin).clamp(0, 1)


def photo_aug(img, labs, g):
    """brightness then contrast (each with p 0.5, factor in [0.9, 1.1], as PIL's ImageEnhance) on img (B, 3, H, W)
    0..255 and the same on the colours labs (Lab (B, ...)), so the targets stay the colours the image shows"""
    B, dev = len(img), img.device
    fac = lambda: torch.where(torch.rand(B, device=dev, generator=g) < 0.5,  # noqa: E731
                              0.9 + 0.2 * torch.rand(B, device=dev, generator=g), torch.ones(B, device=dev))
    fb, fc = fac(), fac()
    img = (img * fb.view(-1, 1, 1, 1)).clamp(0, 255)
    m = (img * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1).mean((1, 2))  # PIL's L mean
    img = (m.view(-1, 1, 1, 1) + fc.view(-1, 1, 1, 1) * (img - m.view(-1, 1, 1, 1))).clamp(0, 255)
    out = []
    for lab in labs:
        sh = (-1,) + (1,) * (lab.dim() - 1)
        s = (lab2rgb(lab) * fb.view(sh)).clamp(0, 1)
        s = (m.view(sh) / 255 + fc.view(sh) * (s - m.view(sh) / 255)).clamp(0, 1)
        out.append(torch.where(((fb != 1) | (fc != 1)).view(sh), rgb2lab(s), lab))  # untouched rows: no gamut clip
    return img, out


def rot_ab(lab, th):
    """rotate the ab plane of lab (B, ..., 3) by th (B,)"""
    c, s = th.cos().view(-1, *[1] * (lab.dim() - 2)), th.sin().view(-1, *[1] * (lab.dim() - 2))
    a, b = lab[..., 1], lab[..., 2]
    return torch.stack([lab[..., 0], c * a - s * b, s * a + c * b], -1)


def head_frame(split, n):
    """(valid, k, b) of the n hairstyles of cards_{split}.npz"""
    hz = np.load(paths.data(f'head_frame_{split}.npz'))
    return hz['valid'][:n], hz['k'][:n], hz['b'][:n]


def targets(split):
    return np.load(paths.data('color', f'targets_{split}.npz'))


def items_of(split):
    """hairstyles with colour targets from the VRM materials, a head fit, <= NMAX cards and renders"""
    g = targets(split)
    hp = np.load(paths.data(f'cards_{split}.npz'))['hair_ptr']
    valid = head_frame(split, len(hp) - 1)[0]
    meta = json.load(open(paths.data(f'{split}_cards_meta.json')))
    rd = np.array([os.path.isfile(f'{paths.data(RENDERS[split])}/{m["key"].replace("/", "_")}/meta.json') for m in meta])
    return np.flatnonzero(g['done'] & g['factors'] & valid & (np.diff(hp) <= NMAX) & rd)


def hard_items(split, items, gap=5.0):
    """the hairstyles of items whose lit and shade colours are apart: mean Delta E over the bins with a target > gap"""
    g = targets(split)
    hp = np.load(paths.data(f'cards_{split}.npz'))['hair_ptr']
    d = np.linalg.norm(g['lab'] - g['lab_shade'], axis=-1) * (g['n'] > 0)
    cs = np.concatenate([[0], np.cumsum(d.sum(1))])
    cc = np.concatenate([[0], np.cumsum((g['n'] > 0).sum(1))])
    per = (cs[hp[1:]] - cs[hp[:-1]]) / np.maximum(cc[hp[1:]] - cc[hp[:-1]], 1)
    return items[per[items] > gap]


def color_stats(split='train'):
    """mean and std of the target Lab colours (lit and shade together) over the bins with samples"""
    g = targets(split)
    w = g['n'] > 0
    x = np.concatenate([g['lab'][w], g['lab_shade'][w]]).astype(np.float64)
    return x.mean(0), x.std(0)


def view_targets(split, hp, fill_w=0.5):
    """the per-view targets for the cards of cards_<split>.npz (hair_ptr hp): Lab (C, VIEWS, BINS, 3) float16, weight
    (C, VIEWS, BINS) float16 (1: measured in that view, fill_w: from the shading model, 0: no target), hairstyles with
    targets"""
    C, Hn = int(hp[-1]), len(hp) - 1
    gt = np.zeros((C, VIEWS, BINS, 3), np.float16)
    w = np.zeros((C, VIEWS, BINS), np.float16)
    have = np.zeros(Hn, bool)
    for f in sorted(glob.glob(paths.data('color', 'views', f'{split}_*.npz'))):
        z = np.load(f)
        for h in z['hair']:
            a, b = int(hp[h]), int(hp[h + 1])
            g = z[f'gt_{h}']
            if len(g) != b - a:  # not the same cards
                continue
            gt[a:b] = g
            w[a:b] = np.where(z[f'nvis_{h}'] >= 5, 1.0, fill_w) * z[f'views_{h}'][None, :, None]
            have[h] = True
    return gt, w, have


class Views(Dataset):
    """one item = (hairstyle, view) -> the condition crop (uint8, RES), its hair label and the character (the render's
    alpha, the person matte at inference): a random view with hair, the training crop, brightness / contrast (photo:
    here on the image only), blur, JPEG"""

    def __init__(self, split, items, seed=0, photo=True):
        meta = json.load(open(paths.data(f'{split}_cards_meta.json')))
        self.rdir = [f'{paths.data(RENDERS[split])}/{m["key"].replace("/", "_")}' for m in meta]
        self.items, self.seed, self.photo, self.epoch = items, seed, photo, 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, j):
        i = int(self.items[j])
        d = self.rdir[i]
        rng = np.random.default_rng([self.seed, self.epoch, j])
        v0 = int(rng.integers(VIEWS))
        for v in [(v0 + u) % VIEWS for u in range(VIEWS)]:  # a view with hair in it
            lab = np.asarray(Image.open(f'{d}/v{v}_lab.png'))
            if (lab == LAB_HAIR).any():
                break
        else:
            v = v0
            lab = np.asarray(Image.open(f'{d}/v{v}_lab.png'))
        try:
            box = crop_box(lab == LAB_HAIR, lab == LAB_FACE, rng=rng, **TRAIN_CROP)
        except ValueError:  # neither hair nor face in this view: the whole image
            box = (0.0, 0.0, float(lab.shape[1]), float(lab.shape[0]))
        rgba = apply_crop(Image.open(f'{d}/v{v}.png'), box, RES)
        img = composite(rgba)
        hm = np.asarray(apply_crop(Image.fromarray(((lab == LAB_HAIR) * 255).astype(np.uint8)), box, RES,
                                   resample=Image.Resampling.NEAREST)) > 127
        pm = np.asarray(rgba.convert('RGBA'))[..., 3] > 127
        for E in (ImageEnhance.Brightness, ImageEnhance.Contrast):
            if rng.random() < 0.5 and self.photo:
                img = E(img).enhance(float(rng.uniform(0.9, 1.1)))
        if rng.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.2))))
        if rng.random() < 0.3:
            buf = io.BytesIO()
            img.save(buf, 'JPEG', quality=int(rng.integers(50, 95)))
            img = Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
        return dict(img=torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1), hair=torch.from_numpy(hm),
                    person=torch.from_numpy(pm), hid=i, v=v)


class Cards:
    """card tokens (head frame) and colour targets of given hairstyles, gathered on the GPU in a random slot order"""

    def __init__(self, split, tok, vae, dev, seed=0, view=False, fill_w=0.5):
        z = np.load(paths.data(f'cards_{split}.npz'))
        _, k, b = head_frame(split, len(z['hair_ptr']) - 1)
        g = targets(split)
        T = lambda x, dt=None: torch.from_numpy(np.ascontiguousarray(x)).to(dev, dt)  # noqa: E731
        self.tok, self.dev = tok, dev
        self.hp, self.root, self.nrm = T(z['hair_ptr']), T(z['res'][:, 0, :3], torch.float32), T(z['nrm'], torch.float32)
        self.n, self.shape = T(z['n'].astype(np.int64)), T(z['shape'].astype(np.int64))
        self.lat = T(np.load(paths.ckpt(vae, f'latents_{split}_head.npy')), torch.float32)
        self.k, self.b = T(k, torch.float32), T(b, torch.float32)
        self.lab, self.lab_s = T(g['lab'], torch.float32), T(g['lab_shade'], torch.float32)
        self.w = T(g['n'] > 0, torch.float32)
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.gtv = self.wv = self.have = None
        if view:
            gv, wv, self.have = view_targets(split, z['hair_ptr'], fill_w)
            self.gtv, self.wv = T(gv), T(wv)

    def __call__(self, hid, view=None):
        B, N, dev = len(hid), NMAX, self.dev
        ar = torch.arange(N, device=dev)
        a = self.hp[hid]
        nc = torch.minimum(self.hp[hid + 1] - a, torch.tensor(N, device=dev))
        perm = torch.argsort(torch.rand(B, N, device=dev, generator=self.g), 1)
        val = torch.where(ar[None] < nc[:, None], a[:, None] + ar[None], torch.full((B, N), -1, device=dev))
        cid = torch.full((B, N), -1, dtype=torch.long, device=dev).scatter_(1, perm.contiguous(), val)
        mask = cid >= 0
        # card dropout: half of the hairstyles keep 70 to 100 % of their cards
        keep = 0.7 + 0.3 * torch.rand(B, 1, device=dev, generator=self.g)
        on = torch.rand(B, 1, device=dev, generator=self.g) < 0.5
        drop = on & (torch.rand(B, N, device=dev, generator=self.g) > keep)
        drop &= mask.sum(1, keepdim=True) > 1
        mask = mask & ~drop
        cid = torch.where(mask, cid, torch.full_like(cid, -1))
        safe = cid.clamp(min=0)
        root = self.k[hid][:, None, None] * self.root[safe] + self.b[hid][:, None, :]
        res = torch.zeros(B, N, 1, 5, device=dev)
        res[..., 0, :3] = root
        x = self.tok.encode(res, self.nrm[safe] * mask[..., None], torch.where(mask, self.n[safe], torch.full_like(safe, 3)),
                            self.shape[safe] * mask, mask, self.lat[safe] * mask[..., None])
        # token noise on the root and the latent of the existing cards
        nz = torch.randn(B, N, x.shape[-1], device=dev, generator=self.g) * mask[..., None]
        x[..., :3] += 0.05 * nz[..., :3]
        x[..., 3:3 + self.tok.zdim] += 0.1 * nz[..., 3:3 + self.tok.zdim]
        m = mask[..., None].float()
        out = dict(x=x, mask=mask, cid=cid, lab=self.lab[safe] * m[..., None], lab_s=self.lab_s[safe] * m[..., None],
                   w=self.w[safe] * m)
        if view is not None and self.gtv is not None:  # view (B,): the render view of each hairstyle's image
            vv = view.to(dev).long()[:, None].expand(B, N)
            out['lab_v'] = self.gtv[safe, vv].float() * m[..., None]
            out['w_v'] = self.wv[safe, vv].float() * m
        return out
