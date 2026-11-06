"""A user's VRM 0.x avatar as a HairDeck body (hairdeck.vrm.body), in place of HairDeck's bundled avatar.

HairDeck's own body build (python -m hairdeck.vrm.body build) needs the avatar's hair as hair.glb in the frame of its
training data, which only the bundled avatar has. That file serves one purpose, hairdeck.data.head_frame.fit_one reads
its bounding box to tie Blender's frame to that frame; any frame works as long as it is consistent. Here hair.glb is a
box of the avatar's own rest-pose hair box in the dataset axes (axis_map), so the fit comes out with scale 1 and no
shift, and the rest is HairDeck's: fit_one on the rest-pose boxes, then k = W / w and b = O - k * anchor with the
canonical head (CANONICAL_HEAD), as hairdeck.vrm.body.build computes them.

build_avatar(vrm, backend, blender_path, collider) -> {'body': body file, 'name', 'info'}; one folder per file content
under <workspace>/avatars/<name>_<sha>/ (avatar.vrm or source.vrm, boxes.json, hair.glb, body.json, blender.log),
reused while the file is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np

from . import hairdeck_env as H

SCRIPT = H.HERE / "blender" / "avatar_prep.py"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def vrm_version(path) -> str:
    """'0.x', '1.0' or 'none' from the glTF JSON chunk of a .vrm file."""
    with open(path, "rb") as f:
        head = f.read(20)
        if len(head) < 20 or head[:4] != b"glTF":
            return "none"
        n = struct.unpack_from("<I", head, 12)[0]
        ext = json.loads(f.read(n)).get("extensions", {})
    return "0.x" if "VRM" in ext else "1.0" if "VRMC_vrm" in ext else "none"


def build_avatar(vrm, backend="auto", blender_path="", collider=True) -> dict:
    vrm = os.path.abspath(vrm)
    version = vrm_version(vrm)
    if version != "0.x":
        raise ValueError(f"[HairDeck] {os.path.basename(vrm)} is not a VRM 0.x file ({version}). HairDeck reads the "
                         "VRM 0.x humanoid and writes VRM 0.x spring bones; export the avatar as VRM 0.x (VRoid "
                         "Studio: Export > VRM 0.0).")
    name = re.sub(r"[^A-Za-z0-9_-]", "_", Path(vrm).stem) or "avatar"
    d = H.workspace() / "avatars" / f"{name}_{file_sha256(vrm)[:12]}{'' if collider else '_nocollider'}"
    body_json = d / "body.json"
    if body_json.is_file():
        return dict(body=str(body_json), name=name, info=json.loads((d / "summary.json").read_text()))
    d.mkdir(parents=True, exist_ok=True)
    # Blender: boxes, and with collider the prepared VRM (bpy_runner.py enables and patches the VRM Add-on first)
    cmd, uses_bpy = H.blender_cmd(backend, blender_path)
    runner = [] if uses_bpy else ["-b", "--python", str(H.RUNNER)]
    argv = cmd + runner + ["-b", "--python", str(SCRIPT), "--", vrm, str(d)] + (["--collider"] if collider else [])
    H.install_vrm_addon()
    env = dict(H.blender_env(), HAIRDECK_VRM_ADDON=H.VRM_ADDON_MODULE)
    log = d / "blender.log"
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(H.VENDOR), timeout=900)
    if r.returncode != 0 or not (d / "boxes.json").is_file():
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"[HairDeck] preparing the avatar failed (exit {r.returncode}), see {log}:\n{tail}")
    boxes = json.loads((d / "boxes.json").read_text())
    # the avatar's own VRM in the folder, so the body does not depend on the input file staying in place
    src = boxes["prepared_vrm"] or str(d / "source.vrm")
    if not boxes["prepared_vrm"]:
        shutil.copyfile(vrm, src)

    import trimesh
    from hairdeck import paths
    from hairdeck.data.head_frame import anchor, axis_map, fit_one
    from hairdeck.vrm.body import CANONICAL_HEAD

    lo, hi = (axis_map(np.array(x)) for x in boxes["hair_rest_bbox"])
    trimesh.creation.box(bounds=[np.minimum(lo, hi), np.maximum(lo, hi)]).export(str(d / "hair.glb"))
    _, fit = fit_one((name, str(d), str(d / "boxes.json")))
    if fit.get("status") != "ok":
        raise RuntimeError(f"[HairDeck] the head fit of the avatar failed: {fit}")
    hz = np.load(paths.data(CANONICAL_HEAD))
    W, O = float(hz["W"]), hz["O"].astype(np.float64)
    k = W / fit["w"]
    body = dict(h=901, key=name, vrm=src, dir=str(d), fit=fit, k=float(k), b=(O - k * anchor(fit)).tolist(),
                file_tag=name, made_by="ComfyUI-HairDeck avatar.py")  # no vrm_meta: the avatar keeps its own meta
    summary = dict(source=vrm, hair_meshes=boxes["hair_objects"], head_meshes=boxes["face_objects"],
                   head_width_m=round(fit["w"] / fit["s"], 3), spring_groups=boxes["spring_groups"],
                   collider_groups=boxes["collider_groups"], head_collider_added=boxes["head_collider"])
    (d / "summary.json").write_text(json.dumps(summary, indent=1))
    body_json.write_text(json.dumps(body, indent=1))  # last: its presence marks a complete folder
    return dict(body=str(body_json), name=name, info=summary)
