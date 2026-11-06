"""A user's VRM 0.x avatar prepared for HairDeck's VRM export (avatar.py), in Blender with the VRM Add-on for Blender
(through bpy_runner.py, which enables and patches it):
  1. import with HairDeck's blender/vrm_scene.import_vrm and its humanoid head bone;
  2. the meshes HairDeck replaces (vrm_scene.hair_objects: 'hair' in the object name) and the meshes that measure the
     head: vrm_scene.head_objects ('face' in the name, VRoid's Face), else meshes with 'head' in the name;
  3. the rest-pose fields blender/body_boxes.py writes for hairdeck.data.head_frame.fit_one: hair_rest_bbox,
     face_rest_bbox (evaluated vertices) and head_rest_matrix (the head bone's rest matrix in the Blender world);
  4. with --collider, when the avatar has no VRM 0.x collider group: one sphere collider on the head bone, fitted to
     the cranium (the head meshes' vertices above the head joint: centre of their box, radius half its smaller
     horizontal side), so the new hair's spring bones collide with the head and their scalp joints stay rigid
     (blender/vrm_hair.py uses every collider group when the removed hair had none); the avatar is then re-exported
     to OUT_DIR/avatar.vrm (its meta unchanged). Otherwise the source VRM is used as it is.

blender -b --python avatar_prep.py -- VRM OUT_DIR [--collider]
Writes OUT_DIR/boxes.json and prints AVATAR + json.
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'HairDeck', 'blender'))
import vrm_hair as E  # noqa: E402
import vrm_scene as S  # noqa: E402


def head_collider(arm, head_bone, pts, head_joint):
    """One VRM 0.x collider group on the head bone with a sphere fitted to the cranium; returns (centre, radius)."""
    top = pts[pts[:, 2] > head_joint[2]]  # Z up: the part of the head above the head joint
    lo, hi = top.min(0), top.max(0)
    centre, radius = (lo + hi) / 2, float(min(hi[0] - lo[0], hi[1] - lo[1]) / 2)
    bpy.ops.vrm.add_vrm0_secondary_animation_collider_group(armature_object_name=arm.name)
    sa = arm.data.vrm_addon_extension.vrm0.secondary_animation
    gi = len(sa.collider_groups) - 1
    sa.collider_groups[gi].node.bone_name = head_bone
    bpy.ops.vrm.add_vrm0_secondary_animation_collider_group_collider(armature_object_name=arm.name,
                                                                      collider_group_index=gi, bone_name=head_bone)
    obj = sa.collider_groups[gi].colliders[-1].bpy_object
    bpy.context.view_layer.update()
    obj.matrix_world = Matrix.Translation(centre.tolist())  # keeps the bone parent; local offset from the world spot
    obj.empty_display_size = radius
    bpy.context.view_layer.update()
    return centre.tolist(), radius


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    vrm, out = argv[0], argv[1]
    os.makedirs(out, exist_ok=True)
    arm = S.import_vrm(vrm)
    hb = S.human_bones(arm)
    if 'head' not in hb:
        raise RuntimeError('the avatar has no humanoid head bone (VRM 0.x humanoid)')
    meshes = [o.name for o in bpy.data.objects if o.type == 'MESH']
    hair = S.hair_objects()
    face = S.head_objects() or [o for o in bpy.data.objects
                                if o.type == 'MESH' and 'head' in o.name.lower() and o not in hair]
    if not hair or not face:
        raise RuntimeError(f'need hair meshes ("hair" in the name) and a face or head mesh ("face" or "head" in the '
                           f'name); found hair {[o.name for o in hair]}, face {[o.name for o in face]} among {meshes}')
    hair_pts, face_pts = S.eval_points(hair), S.eval_points(face)
    M = arm.matrix_world @ arm.data.bones[hb['head']].matrix_local
    head_joint = np.array(M.translation)
    info = dict(vrm=vrm, hair_objects=[o.name for o in hair], face_objects=[o.name for o in face], meshes=meshes,
                hair_rest_bbox=[hair_pts.min(0).tolist(), hair_pts.max(0).tolist()],
                face_rest_bbox=[face_pts.min(0).tolist(), face_pts.max(0).tolist()],
                head_rest_matrix=[list(map(float, r)) for r in M], head_bone=hb['head'],
                collider_groups=len(arm.data.vrm_addon_extension.vrm0.secondary_animation.collider_groups),
                spring_groups=len(arm.data.vrm_addon_extension.vrm0.secondary_animation.bone_groups),
                prepared_vrm=None, head_collider=None)
    if '--collider' in argv and info['collider_groups'] == 0:
        centre, radius = head_collider(arm, hb['head'], face_pts, head_joint)
        info['head_collider'] = dict(centre=centre, radius=radius)
        path = os.path.join(out, 'avatar.vrm')
        if not E.export(arm, path):
            raise RuntimeError(f'the VRM export of the prepared avatar failed ({path})')
        info['prepared_vrm'] = path
    json.dump(info, open(os.path.join(out, 'boxes.json'), 'w'), indent=1)
    print('AVATAR', json.dumps(info))


main()
