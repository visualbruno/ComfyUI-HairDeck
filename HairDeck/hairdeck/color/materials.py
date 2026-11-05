"""Hair colours from the source VRM: the materials' MToon colours, the hair mesh's textured parts, and colour samples
of its surface.

The hair mesh (hair.glb of the dataset) lost the VRM materials' colour factors in Blender's export (about a third of
the hair textures are grey, tinted only by them), so the factors come from the character's VRM file (its glTF JSON
chunk): the lit colour is the texture times MToon's _Color (VRM 0.x) or baseColorFactor (VRM 1.0), the shade colour the
texture times _ShadeColor / shadeColorFactor. The renders and the source illustrations show the lit colour in the
light and the shade colour in the shade.
"""
import json
import re
import struct

import numpy as np


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def vrm_materials(vrm):
    """[(name, lit linear RGBA, shade linear RGB)] of a VRM file's materials in order. VRM 0.x: MToon _Color /
    _ShadeColor (sRGB in materialProperties, index-aligned with the materials, else by name); VRM 1.0: baseColorFactor /
    VRMC_materials_mtoon shadeColorFactor (linear); other materials: baseColorFactor for both. (The
    pbrMetallicRoughness.baseColorFactor of a VRM 0.x file is not the MToon colour.)"""
    with open(vrm, 'rb') as f:
        f.read(12)
        n, _ = struct.unpack('<II', f.read(8))
        g = json.loads(f.read(n))
    mats = g.get('materials', [])
    mp = (g.get('extensions', {}).get('VRM') or {}).get('materialProperties') or []
    by_name = {q.get('name'): q for q in mp}
    out = []
    for i, m in enumerate(mats):
        name = m.get('name', '')
        pbr = np.asarray(m.get('pbrMetallicRoughness', {}).get('baseColorFactor', [1, 1, 1, 1]), np.float64)
        q = mp[i] if len(mp) == len(mats) else by_name.get(name)
        mt = (m.get('extensions') or {}).get('VRMC_materials_mtoon')
        if q is not None and str(q.get('shader', '')).startswith('VRM/MToon'):
            vp = q.get('vectorProperties', {})
            c = np.asarray(vp.get('_Color', [1, 1, 1, 1]), np.float64)
            sh = np.asarray(vp.get('_ShadeColor', c[:3]), np.float64)[:3]
            out.append((name, np.r_[srgb_to_lin(c[:3]), c[3]], srgb_to_lin(sh)))
        elif mt is not None:
            out.append((name, pbr, np.asarray(mt.get('shadeColorFactor', pbr[:3]), np.float64)[:3]))
        else:
            out.append((name, pbr, pbr[:3]))
    return out


def match_materials(parts, mats):
    """Per textured part (hair.glb material name; Blender's suffix .NNN marks the NNN-th material of that name) the
    (lit, shade) of the matching VRM material, or None."""
    by = {}
    for name, lit, shade in mats:
        by.setdefault(name, []).append((lit, shade))
    out = []
    for pt in parts:
        m = re.match(r'^(.*?)(?:\.(\d{3}))?$', pt[4] or '')
        lst = by.get(m.group(1)) or by.get(pt[4] or '')
        out.append(lst[min(int(m.group(2) or 0), len(lst) - 1)] if lst else None)
    return out


def textured_parts(path):
    """[(trimesh, per-vertex UV or None, texture image or None, base colour factor, material name)] of a hair mesh, in
    its own (the dataset's) frame."""
    import trimesh
    s = trimesh.load(path)
    parts = []
    for m in (s.dump() if hasattr(s, 'dump') else [s]):
        mat = getattr(m.visual, 'material', None)
        tex = getattr(mat, 'baseColorTexture', None)
        fac = np.asarray(getattr(mat, 'baseColorFactor', None) if getattr(mat, 'baseColorFactor', None) is not None
                         else [255, 255, 255, 255], np.float64)
        fac = fac / 255.0 if fac.max() > 1.0 else fac
        uv = getattr(m.visual, 'uv', None)
        parts.append((m, uv, tex, fac, getattr(mat, 'name', None)))
    return parts


def sample_parts(parts, n, rng, factors):
    """Area-uniform samples over all parts -> points (n, 3), lit sRGB (n, 3), shade sRGB (n, 3), alpha (n,), part (n,),
    face normal (n, 3), interpolated vertex normal (n, 3); factors: per part (lit linear RGBA, shade linear RGB) or None
    (match_materials), multiplied onto the texture in linear space; a part without one keeps the mesh's own factor for
    both colours."""
    import trimesh
    areas = np.array([p[0].area for p in parts])
    cnt = rng.multinomial(n, areas / areas.sum())
    P, LIT, SH, A, PI, N, NV = [], [], [], [], [], [], []
    for pi, ((m, uv, tex, fac, name), k) in enumerate(zip(parts, cnt)):
        if k == 0:
            continue
        fa = m.area_faces / m.area_faces.sum()
        fi = rng.choice(len(m.faces), k, p=fa)
        r1, r2 = rng.random(k), rng.random(k)
        s1 = np.sqrt(r1)
        bc = np.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
        P.append((m.vertices[m.faces[fi]] * bc[..., None]).sum(1))
        N.append(m.face_normals[fi])
        vn = (m.vertex_normals[m.faces[fi]] * bc[..., None]).sum(1)
        NV.append(vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12))
        if tex is not None and uv is not None:
            u = (uv[m.faces[fi]] * bc[..., None]).sum(1)
            rgba = trimesh.visual.color.uv_to_color(u % 1.0, tex).astype(np.float64) / 255.0
        else:
            rgba = np.ones((k, 4))
        f = factors[pi] if factors is not None else None
        if f is not None:
            t = srgb_to_lin(rgba[:, :3])
            LIT.append(lin_to_srgb(t * f[0][None, :3]))
            SH.append(lin_to_srgb(t * f[1][None, :3]))
            A.append(rgba[:, 3] * f[0][3])
        else:
            LIT.append(rgba[:, :3] * fac[None, :3])
            SH.append(LIT[-1])
            A.append(rgba[:, 3] * fac[3])
        PI.append(np.full(k, pi))
    cat = np.concatenate
    return cat(P), np.clip(cat(LIT), 0, 1), np.clip(cat(SH), 0, 1), cat(A), cat(PI), cat(N), cat(NV)
