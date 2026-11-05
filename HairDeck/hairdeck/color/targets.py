"""The colour model's targets, per hairstyle with a hair mesh, VRM materials and renders (scripts/data.py --steps colors).

Samples: NS area-uniform surface samples of the hair mesh (materials.sample_parts, generator seeded [37, row]), coloured
by its texture times the VRM materials' lit and shade colours; samples with alpha < 0.5 are dropped. Each sample goes
to the nearest point of the nearest card centreline (the card resampled to 64 points by arc length) and so to one of
BINS bins along the card's arc fraction v.

card_targets: per card and bin the mean lit and shade colour of its samples (CIELAB; n = the number of samples, 0: no
  target), whether every part matched a VRM material, and a check against the renders: the render's mean hair colour in
  views 0 and 4 against the nearest mix of the mean lit and shade colours (Delta E).
view_targets: the colour of every bin under the light of each render view v (the target of an image of view v):
  visible bins (>= MIN_N samples visible in v): the geometric median (Lab) of the render colours of the bin's visible
  samples, unless that colour is more than Delta E 25 from the bin's own [shade, lit] segment (a part moved there by
  the spring motion); then, as the other bins: shade + a (lit - shade) (linear RGB), a from a shading model fitted on
  v's visible samples within Delta E 10 of their own segment: a = X(N.L, N.V, shadow depth) . c, least squares on a
  grid of (s0, w, tau), X = [1, smoothstep(s0, s0 + w, N.L), [behind > tau], N.V, (1 - N.V)^2, min(behind, 0.1),
  N.L^2] (N the sample's vertex normal posed with the head and turned towards the camera, behind its depth behind the
  first surface seen from the sun over the hair height); the bin's a is the mean over its samples visible in any view
  (all its samples if none). nvis: the visible samples per bin and view (0: from the model).
Visibility: the rest-pose hair mesh moves rigidly with the posed head of the view (the head fit's similarity, the
view's head matrix times the inverse rest head matrix) and is projected with the view's camera; a sample is visible
where its depth is within 1 % of the hair height of the mesh's depth buffer at its pixel, the pixel is labelled hair,
lies at least 2 px inside the hair label (the MToon outline) and is opaque. Spring-bone motion is not modelled.
Depth buffers: pyrender (EGL or OSMesa).
"""
import itertools
import json
import math
import os

import numpy as np

from ..cards.codec import resample
from ..data.crop import LAB_HAIR
from .materials import lin_to_srgb, match_materials, sample_parts, srgb_to_lin, textured_parts, vrm_materials

NS, BINS, VIEWS, MIN_N = 120000, 16, 8, 5
GRID = list(itertools.product(np.linspace(-0.8, 0.4, 7), (0.6, 1.0, 1.5, 2.5), (0.005, 0.03)))  # (s0, w, tau)
# the light presets of the renders (blender/vrm_scene.py LIGHTS): euler rotation, sun energy, world strength
LIGHTS = {'key_left': ((0.6, 0.0, 0.4), 3.0, 0.5), 'key_right': ((0.6, 0.0, -0.7), 3.0, 0.5),
          'back_top': ((-0.6, 0.0, 0.2), 4.0, 0.4), 'flat': ((0.0, 0.0, 0.0), 0.3, 1.2)}


def samples(row, folder, vrm):
    """The surface samples of a hairstyle's hair mesh: (parts, factors, points, lit, shade (sRGB), vertex normals), the
    samples with alpha >= 0.5; factors None when the VRM is missing."""
    parts = textured_parts(os.path.join(folder, 'hair.glb'))
    fac = match_materials(parts, vrm_materials(vrm)) if vrm and os.path.isfile(vrm) else None
    p, lit, sh, alpha, _, _, nrm_v = sample_parts(parts, NS, np.random.default_rng([37, row]), fac)
    keep = alpha >= 0.5
    return parts, fac, p[keep], lit[keep], sh[keep], nrm_v[keep]


def bin_keys(p, cards):
    """card * BINS + bin of every sample: the nearest point of the 64-point centrelines"""
    from scipy.spatial import cKDTree
    R = np.stack([resample(s[:, :5], 64)[:, :3] for s, _ in cards])
    _, j = cKDTree(R.reshape(-1, 3)).query(p)
    return (j // 64) * BINS + np.minimum((j % 64) * BINS // 64, BINS - 1)


def card_targets(row, folder, cards, render_dir):
    """One hairstyle (row of the cards file, its folder and cards, its render folder) -> dict(lab, lab_shade (C, BINS,
    3) float32, n (C, BINS) int32, factors bool, render_dE (2,))"""
    from PIL import Image
    from skimage.color import rgb2lab
    vrm = json.load(open(f'{render_dir}/meta.json')).get('vrm') if os.path.isfile(f'{render_dir}/meta.json') else None
    _, fac, p, rgb, rgb_s, _ = samples(row, folder, vrm)
    lab, lab_s = rgb2lab(rgb[None])[0], rgb2lab(rgb_s[None])[0]
    key = bin_keys(p, cards)
    n = np.bincount(key, minlength=len(cards) * BINS)
    mus = []
    for x in (lab, lab_s):
        s = np.zeros((len(cards) * BINS, 3))
        np.add.at(s, key, x)
        mus.append(s / np.maximum(n, 1)[:, None])
    gl, gs = lab.mean(0), lab_s.mean(0)
    rd = [np.nan, np.nan]
    for k, v in enumerate((0, 4)):
        f = f'{render_dir}/v{v}.png'
        if os.path.isfile(f):
            im = np.asarray(Image.open(f).convert('RGBA'), np.float64) / 255.0
            lb = np.asarray(Image.open(f'{render_dir}/v{v}_lab.png'))
            m = (lb == LAB_HAIR) & (im[..., 3] > 0.5)
            if m.sum() > 50:
                r = rgb2lab(im[m][None, :, :3])[0].mean(0)
                t = np.clip(np.dot(r - gs, gl - gs) / max(np.dot(gl - gs, gl - gs), 1e-9), 0, 1)
                rd[k] = float(np.linalg.norm(r - (gs + t * (gl - gs))))
    return dict(lab=mus[0].reshape(len(cards), BINS, 3).astype(np.float32),
                lab_shade=mus[1].reshape(len(cards), BINS, 3).astype(np.float32),
                n=n.reshape(len(cards), BINS).astype(np.int32),
                factors=fac is not None and all(f is not None for f in fac), render_dE=rd)


# ----------------------------------------------------------------------------- the render views
def dataset_to_world(p, fit, m, view):
    """dataset-frame rest points -> posed Blender world points of the view (the head fit's similarity, then the view's
    head motion)"""
    a = (np.asarray(p, np.float64) - np.array(fit['t'])) / fit['s']
    B = np.stack([fit['sx'] * a[:, 0], fit['sz'] * a[:, 2], a[:, 1]], 1)
    T = np.array(view['head_matrix']) @ np.linalg.inv(np.array(m['head_rest_matrix']))
    return B @ T[:3, :3].T + T[:3, 3]


def dataset_to_world_dir(n, fit, m, view):
    """unit directions (normals) through the same map"""
    B = np.stack([fit['sx'] * n[:, 0], fit['sz'] * n[:, 2], n[:, 1]], 1)
    T = np.array(view['head_matrix']) @ np.linalg.inv(np.array(m['head_rest_matrix']))
    d = B @ T[:3, :3].T
    return d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)


def project(pw, view, res):
    """world points -> (pixel u, v), depth along the view (the view's pinhole camera)"""
    C = np.array(view['cam_matrix'])
    q = (pw - C[:3, 3]) @ C[:3, :3]
    t = q[:, :2] / -q[:, 2:3]
    k = view['lens'] / view['sensor']
    u = (0.5 + t[:, 0] * k - view['shift'][0]) * res
    v = (0.5 - (t[:, 1] * k - view['shift'][1])) * res
    return np.stack([u, v], 1), -q[:, 2]


def euler_xyz(r):
    """Blender's XYZ euler -> rotation matrix"""
    x, y, z = r
    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def light_of(look):
    """(direction towards the sun, world space), sun energy, world strength of a view's look"""
    li = look['light']
    rot, energy, world = (LIGHTS[li['name']] if li['kind'] == 'preset' else (li['rot'], li['energy'], li['world']))
    return euler_xyz(rot) @ np.array([0.0, 0.0, 1.0]), float(energy), float(world)


def depth_buffer(V, F, view, res):
    """the mesh's depth buffer in the view's camera (0: no surface)"""
    import pyrender
    import trimesh
    k = view['lens'] / view['sensor'] * res
    cx = (0.5 - view['shift'][0]) * res
    cy = (0.5 + view['shift'][1]) * res
    s = pyrender.Scene()
    s.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(V, F, process=False), smooth=False))
    s.add(pyrender.IntrinsicsCamera(k, k, cx, cy, znear=0.05, zfar=100.0), pose=np.array(view['cam_matrix']))
    r = pyrender.OffscreenRenderer(res, res)
    d = r.render(s, flags=pyrender.RenderFlags.DEPTH_ONLY)
    r.delete()
    return d


def light_depth(V, F, L, pts, res=1024):
    """per point its depth behind the first surface of the mesh along the sun direction L (towards the sun; world
    units, 0 = on the lit side): an orthographic depth buffer of the mesh seen from the sun"""
    import pyrender
    import trimesh
    c = (V.min(0) + V.max(0)) / 2
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    z = L / np.linalg.norm(L)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, c + z * 2 * ext
    s = pyrender.Scene()
    s.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(V, F, process=False), smooth=False))
    s.add(pyrender.OrthographicCamera(xmag=ext / 2, ymag=ext / 2, znear=0.01, zfar=4 * ext), pose=pose)
    r = pyrender.OffscreenRenderer(res, res)
    D = r.render(s, flags=pyrender.RenderFlags.DEPTH_ONLY)
    r.delete()
    q = (pts - pose[:3, 3]) @ pose[:3, :3]
    u = np.clip(np.round((q[:, 0] / (ext / 2) * 0.5 + 0.5) * res).astype(int), 0, res - 1)
    v = np.clip(np.round((0.5 - q[:, 1] / (ext / 2) * 0.5) * res).astype(int), 0, res - 1)
    d = D[v, u]
    return np.where(d > 0, -q[:, 2] - d, 0.0)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


def design(ndl, ndv, behind, prm):
    """the shading model's regressors of a"""
    s0, w, tau = prm
    return np.stack([np.ones_like(ndl), smoothstep(s0, s0 + w, ndl), (behind > tau).astype(np.float64), ndv,
                     (1 - ndv) ** 2, np.minimum(behind, 0.1), ndl * ndl], 1)


def geo_median(keys, X, nk, it=20):
    """per key the geometric median of the rows of X (Weiszfeld, from the per-channel median)"""
    o = np.argsort(keys, kind='stable')
    k = keys[o]
    start, end = np.searchsorted(k, np.arange(nk)), np.searchsorted(k, np.arange(nk), side='right')
    c = np.zeros((nk, 3))
    for j in range(3):
        oo = np.lexsort((X[:, j], keys))
        v = X[oo, j]
        cnt = end - start
        lo = np.minimum(start + np.maximum(cnt - 1, 0) // 2, len(v) - 1)
        hi = np.minimum(start + cnt // 2, len(v) - 1)
        c[:, j] = np.where(cnt > 0, 0.5 * (v[lo] + v[hi]), 0.0)
    for _ in range(it):
        w = 1 / np.maximum(np.linalg.norm(X - c[keys], axis=1), 1e-3)
        s, sw = np.zeros((nk, 3)), np.zeros(nk)
        np.add.at(s, keys, X * w[:, None])
        np.add.at(sw, keys, w)
        c = np.where((sw > 0)[:, None], s / np.maximum(sw, 1e-12)[:, None], c)
    return c


def seg_a(c, S, L):
    """the position a of colours c on the segments [S, L] (clipped to [0, 1]) and whether the segment has a length"""
    d = L - S
    dd = (d * d).sum(-1)
    return np.clip(((c - S) * d).sum(-1) / np.maximum(dd, 1e-9), 0, 1), dd > 1e-6


def view_targets(row, folder, cards, render_dir, fit):
    """One hairstyle (row of the cards file, its folder and cards, its render folder, its head fit) -> (gt (C, VIEWS,
    BINS, 3) float16 Lab, nvis (C, VIEWS, BINS) uint16, views (VIEWS,) bool: the views with a render)"""
    from PIL import Image
    from scipy import ndimage
    from skimage.color import lab2rgb, rgb2lab
    m = json.load(open(f'{render_dir}/meta.json'))
    parts, _, p, lit, sh, nrm_v = samples(row, folder, m['vrm'])
    lit, sh = srgb_to_lin(lit), srgb_to_lin(sh)
    C = len(cards)
    key = bin_keys(p, cards)
    nk = C * BINS
    cnt = np.maximum(np.bincount(key, minlength=nk), 1)[:, None]
    binL, binS = np.zeros((nk, 3)), np.zeros((nk, 3))
    np.add.at(binL, key, lit)
    np.add.at(binS, key, sh)
    binL, binS = binL / cnt, binS / cnt
    Vs, Fs, off = [], [], 0
    for mm, *_ in parts:
        Vs.append(np.asarray(mm.vertices))
        Fs.append(np.asarray(mm.faces) + off)
        off += len(mm.vertices)
    V0, F0 = np.concatenate(Vs), np.concatenate(Fs)
    gt = np.zeros((C, VIEWS, BINS, 3), np.float16)
    nvis = np.zeros((C, VIEWS, BINS), np.uint16)
    have = np.zeros(VIEWS, bool)
    per_view = []
    vis_any = np.zeros(len(p), bool)
    for v, view in enumerate(m['views'][:VIEWS]):
        f = f'{render_dir}/v{v}.png'
        if not os.path.isfile(f):
            continue
        im = np.asarray(Image.open(f).convert('RGBA'))
        lb = np.asarray(Image.open(f'{render_dir}/v{v}_lab.png'))
        res = lb.shape[0]
        Vw = dataset_to_world(V0, fit, m, view)
        D = depth_buffer(Vw, F0, view, res)
        pw = dataset_to_world(p, fit, m, view)
        uv, dep = project(pw, view, res)
        x, y = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inside = (x >= 0) & (x < res) & (y >= 0) & (y < res)
        x, y = np.clip(x, 0, res - 1), np.clip(y, 0, res - 1)
        core = ndimage.binary_erosion(lb == LAB_HAIR, iterations=2)
        span = float(np.ptp(pw[:, 1]))
        vis = inside & (D[y, x] > 0) & (dep <= D[y, x] + 0.01 * span) & core[y, x] & (im[y, x, 3] > 127)
        Cm = np.array(view['cam_matrix'])
        tc = Cm[:3, 3][None] - pw
        tc /= np.linalg.norm(tc, axis=1, keepdims=True)
        nw = dataset_to_world_dir(nrm_v, fit, m, view)
        nw = np.where(((nw * tc).sum(1) < 0)[:, None], -nw, nw)  # double-sided: the side towards the camera
        ldir, _, _ = light_of(view['look'])
        behind = light_depth(Vw, F0, ldir, pw) / span
        idx = np.flatnonzero(vis)
        obs = srgb_to_lin(im[y[idx], x[idx], :3].astype(np.float64) / 255)
        vis_any[idx] = True
        per_view.append((v, idx, obs, nw @ ldir, (nw * tc).sum(1), behind))
        have[v] = True
    for v, idx, obs, ndl, ndv, behind in per_view:
        k = key[idx]
        n = np.bincount(k, minlength=nk)
        obs_lab = rgb2lab(np.clip(lin_to_srgb(obs), 0, 1)[None])[0]
        cm_lab = geo_median(k, obs_lab, nk) if len(k) else np.zeros((nk, 3))
        a, ok = seg_a(obs, sh[idx], lit[idx])
        near = sh[idx] + a[:, None] * (lit[idx] - sh[idx])
        ok &= np.linalg.norm(obs_lab - rgb2lab(np.clip(lin_to_srgb(near), 0, 1)[None])[0], axis=1) <= 10
        best = None
        if ok.sum() > 50:
            for prm in GRID:
                X = design(ndl[idx][ok], ndv[idx][ok], behind[idx][ok], prm)
                coef, *_ = np.linalg.lstsq(X, a[ok], rcond=None)
                e = float(np.mean((X @ coef - a[ok]) ** 2))
                if best is None or e < best[0]:
                    best = (e, prm, coef)
        if best is not None:
            am = np.clip(design(ndl, ndv, behind, best[1]) @ best[2], 0, 1)
        else:
            am = np.full(len(p), float(a[ok].mean()) if ok.any() else 0.5)
        use = vis_any if vis_any.any() else np.ones(len(p), bool)
        sa, ca = np.zeros(nk), np.zeros(nk)
        np.add.at(sa, key[use], am[use])
        np.add.at(ca, key[use], 1)
        sa2, ca2 = np.zeros(nk), np.zeros(nk)
        np.add.at(sa2, key, am)
        np.add.at(ca2, key, 1)
        ab = np.where(ca > 0, sa / np.maximum(ca, 1), sa2 / np.maximum(ca2, 1))
        X = binS + ab[:, None] * (binL - binS)
        # plausibility of the measured colour: its distance to the bin's own [shade, lit] segment
        cm_lin = srgb_to_lin(np.clip(lab2rgb(cm_lab[None])[0], 0, 1))
        t, _ = seg_a(cm_lin, binS, binL)
        far = np.linalg.norm(cm_lab - rgb2lab(np.clip(lin_to_srgb(binS + t[:, None] * (binL - binS)), 0, 1)[None])[0],
                             axis=1) > 25
        meas = (n >= MIN_N) & ~far
        lab = np.where(meas[:, None], cm_lab, rgb2lab(np.clip(lin_to_srgb(X), 0, 1)[None])[0])
        n = np.where(meas, n, 0)
        gt[:, v] = lab.reshape(C, BINS, 3).astype(np.float16)
        nvis[:, v] = np.minimum(n, 65535).reshape(C, BINS).astype(np.uint16)
    return gt, nvis, have
