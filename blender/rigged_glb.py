"""Rigged GLB of the generated hair alone (any Blender with the bundled glTF exporter; no VRM Add-on, no body import).
Inputs come from hairdeck.vrm.prep (the prep npz), in the Blender world of the default body (Z up, meters): the hair
mesh with per-card UVs and the chain joints. Builds the armature HairRig: a root bone Head at the body's head joint, and
per chain c the bones HairGen_cCC_KK from joint k to joint k+1 (the first parented to Head) plus a leaf HairGen_cCC_end,
as blender/vrm_hair.py does on the body; the mesh Hair_gen skinned with the weights of hairdeck.spring.sim.bind (head
1 - w, segment k w * alpha, neighbour segment w * (1 - alpha)); a double-sided material in the hair colour, or, when the
cards carry the colour model's colours, textured with their colour atlas (P['atlas'], hairdeck.vrm.prep.color_atlas;
the npz's UVs then already point into it). The predicted chain stiffness goes into the armature's extras
(hairdeck_stiffness, one per chain, root bone names in hairdeck_chain_roots) for a spring-bone setup outside Blender.

blender -b --python rigged_glb.py -- PREP_NPZ OUT_GLB      (or bpy_runner.py in place of blender, with the bpy module)
"""
import json
import sys

import bpy
import numpy as np
from mathutils import Vector

HEAD, HEAD_LEN = 'Head', 0.12  # root bone; it points up from the head joint


def build(P):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    arm = bpy.data.objects.new('HairRig', bpy.data.armatures.new('HairRig'))
    scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    ptr = P['chain_ptr']
    chains = [P['chain_J'][ptr[c]:ptr[c + 1]] for c in range(len(ptr) - 1)]
    names = []
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm.data.edit_bones
    hb = eb.new(HEAD)
    hb.head = Vector(P['head_joint'])
    hb.tail = hb.head + Vector((0, 0, HEAD_LEN))
    for c, J in enumerate(chains):
        prev, nm = hb, []
        for k in range(len(J) - 1):
            b = eb.new(f'HairGen_c{c:02d}_{k:02d}')
            b.head, b.tail = Vector(J[k]), Vector(J[k + 1])
            b.parent, b.use_connect, b.use_deform = prev, k > 0, True
            prev = b
            nm.append(b.name)
        d = Vector(J[-1] - J[-2]).normalized()
        e = eb.new(f'HairGen_c{c:02d}_end')
        e.head, e.tail = Vector(J[-1]), Vector(J[-1]) + d * 0.02
        e.parent, e.use_connect, e.use_deform = prev, True, True
        names.append(nm)
    bpy.ops.object.mode_set(mode='OBJECT')
    arm['hairdeck_stiffness'] = [float(s) for s in P['stiffness']]
    arm['hairdeck_chain_roots'] = [nm[0] for nm in names]
    # mesh
    me = bpy.data.meshes.new('Hair_gen')
    me.from_pydata(P['V'].tolist(), [], P['F'].tolist())
    me.update()
    uvl = me.uv_layers.new(name='UVMap')
    vi = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get('vertex_index', vi)
    uvl.data.foreach_set('uv', np.asarray(P['UV'], np.float32)[vi].reshape(-1))
    me.shade_smooth()
    obj = bpy.data.objects.new('Hair_gen', me)
    scene.collection.objects.link(obj)
    obj.parent = arm
    obj.modifiers.new('Armature', 'ARMATURE').object = arm
    vg_head, vgs = obj.vertex_groups.new(name=HEAD), {}

    def add(bn, i, wt):
        if bn not in vgs:
            vgs[bn] = obj.vertex_groups.new(name=bn)
        vgs[bn].add([i], wt, 'REPLACE')
    ch, sg, w, sg2, al = P['vchain'], P['vseg'], P['vw'], P['vseg2'], P['valpha']
    for i in range(len(P['V'])):  # as blender/vrm_hair.py add_generated_hair
        if ch[i] < 0 or w[i] <= 0:
            vg_head.add([i], 1.0, 'REPLACE')
            continue
        two = sg2[i] != sg[i] and al[i] < 1
        add(names[ch[i]][sg[i]], i, float(w[i] * (al[i] if two else 1.0)))
        if two:
            add(names[ch[i]][sg2[i]], i, float(w[i] * (1 - al[i])))
        if w[i] < 1:
            vg_head.add([i], float(1 - w[i]), 'REPLACE')
    # material: double sided; the hair colour (linear RGB), or the per-card colour atlas as the base colour texture
    mat = bpy.data.materials.new('Hair_gen')
    if bpy.app.version < (5, 0, 0):  # always on from Blender 5.0
        mat.use_nodes = True
    mat.use_backface_culling = False
    bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Roughness'].default_value = 0.6
    if 'atlas' in P:  # as blender/vrm_hair.py builds the image: rows flipped (Blender stores them bottom up), sRGB
        A = np.asarray(P['atlas'], np.float32) / 255
        img = bpy.data.images.new('Hair_gen_colors', A.shape[1], A.shape[0], alpha=False)
        img.colorspace_settings.name = 'sRGB'
        img.pixels.foreach_set(np.concatenate([A[::-1], np.ones(A.shape[:2] + (1,), np.float32)], -1).reshape(-1))
        img.file_format = 'PNG'
        img.pack()
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = img
        tex.interpolation = 'Linear'  # the atlas UVs rely on bilinear lookup between the colours along a card
        tex.extension = 'EXTEND'
        mat.node_tree.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    else:
        bsdf.inputs['Base Color'].default_value = (*[float(x) for x in P['colour']], 1.0)
    me.materials.append(mat)
    return dict(n_chains=len(chains), n_bones=1 + sum(len(nm) + 1 for nm in names), n_verts=len(P['V']),
                n_faces=len(P['F']), n_groups=len(obj.vertex_groups),
                colors=f"atlas {P['atlas'].shape[1]}x{P['atlas'].shape[0]}" if 'atlas' in P else 'one hair colour')


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    P = dict(np.load(argv[0]))
    info = build(P)
    r = bpy.ops.export_scene.gltf(filepath=argv[1], export_format='GLB', export_extras=True)
    info['export'] = sorted(r)
    print('RIGGED', json.dumps(info))


if __name__ == '__main__':
    main()
