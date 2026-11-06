"""ComfyUI nodes for HairDeck: stylized hair cards and the spring-bone chains that animate them, from one character
image, saved as GLB, as a rigged GLB (armature + skinned hair) or as a VRM avatar with spring-bone physics.

The pipeline (SAM 3 + BiRefNet preprocessing, DINOv3, the image set-DiT, the card VAE and the chain DiT) is HairDeck's
own hairdeck.infer.image.Pipeline, vendored unmodified in HairDeck/. Each generation writes HairDeck's usual result
folder (cards, chains, tokens, GLBs, info.json) under output/; the rig and VRM nodes work from that folder.
"""

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import comfy.model_management as mm
import folder_paths

try:  # ComfyUI's 3D file type (Preview 3D input); absent in older ComfyUI versions
    from comfy_api.latest import Types

    File3D = Types.File3D
except (ImportError, AttributeError):
    File3D = None

from . import hairdeck_env as H

log = logging.getLogger("HairDeck")

CATEGORY = "3D/HairDeck"
BLENDER_BACKENDS = ["auto", "bpy", "executable"]

# One pipeline at a time: it holds SAM 3, BiRefNet, DINOv3 and the DiTs (~6 GB of weights).
_PIPELINE = {"key": None, "pipe": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(img):
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(a)[None]


def _image_to_array(image, mask=None, invert_mask=False):
    """First IMAGE of a batch -> (H, W, 3|4) uint8; the alpha from the mask, else from a 4th channel."""
    frame = image[0].detach().cpu().float().numpy()
    rgb = (np.clip(frame[..., :3], 0, 1) * 255).round().astype(np.uint8)
    alpha = None
    if mask is not None:
        m = mask[0] if mask.ndim == 3 else mask
        m = m.detach().cpu().float().numpy()
        if invert_mask:
            m = 1.0 - m
        alpha = (np.clip(m, 0, 1) * 255).round().astype(np.uint8)
        if alpha.shape != rgb.shape[:2]:
            alpha = np.asarray(Image.fromarray(alpha).resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
    elif frame.shape[-1] == 4:
        alpha = (np.clip(frame[..., 3], 0, 1) * 255).round().astype(np.uint8)
    if alpha is not None and alpha.max() < 8:
        # LoadImage on a JPG gives an all-zero mask: that means "no alpha", not "nothing to see"
        log.warning("[HairDeck] the mask is empty, ignoring it (try toggling invert_mask)")
        alpha = None
    return np.concatenate([rgb, alpha[..., None]], -1) if alpha is not None else rgb


def _new_output_dir(filename_prefix):
    """A fresh folder under output/ named like ComfyUI's numbered files: <prefix>_00001_/."""
    full, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory())
    name = f"{filename}_{counter:05}_"
    path = os.path.join(full, name)
    os.makedirs(path, exist_ok=True)
    return path


def _output_file(filename_prefix, ext):
    """(absolute path, path relative to output/, ui preview entry) of a new numbered output file."""
    full, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory())
    os.makedirs(full, exist_ok=True)
    name = f"{filename}_{counter:05}_.{ext}"
    return (os.path.join(full, name), os.path.join(subfolder, name).replace("\\", "/"),
            {"filename": name, "subfolder": subfolder, "type": "output"})


def _file3d(path):
    return File3D(path, file_format="glb") if File3D is not None else path


def _run(cmd, log_path, env, what, timeout=900):
    """One child process with its output in log_path; raises with the log's tail."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(H.VENDOR), timeout=timeout)
    if r.returncode != 0:
        tail = open(log_path, encoding="utf-8", errors="replace").read()[-2000:]
        raise RuntimeError(f"[HairDeck] {what} failed (exit {r.returncode}), see {log_path}:\n{tail}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class HairDeckModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "offload": (["on", "auto", "off"], {
                    "default": "on",
                    "tooltip": "on: the models wait in system RAM and visit the GPU only for their stage (leaves the "
                               "GPU to the rest of the workflow). auto: HairDeck decides from the free VRAM. off: "
                               "everything stays on the GPU (fastest, ~10 GB)."}),
                "load_chains": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load the chain DiT, which generates the spring bones. Needed for rigging and VRM."}),
                "auto_download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fetch missing weights (~8 GB) into models/hairdeck. SAM 3 and DINOv3 are gated on "
                               "Hugging Face: accept their licences and log in first."}),
            },
            "optional": {
                "copy_from_workspace": ("STRING", {
                    "default": "",
                    "tooltip": "An existing HairDeck workspace (the folder holding ckpt/, data/, third_party/) to copy "
                               "missing files from instead of downloading them."}),
                "gated_source": (["huggingface", "modelscope"], {
                    "default": "huggingface",
                    "tooltip": "Where SAM 3 and DINOv3 come from. Both are checked against the same sha256."}),
                # Widgets added after the first release go last: saved workflows store widget values by position.
                "load_colors": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load the colour model, which predicts 16 colours along every card. Off: one hair "
                               "colour, measured from the image's hair mask."}),
            },
        }

    RETURN_TYPES = ("HAIRDECK_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Loads the HairDeck pipeline: SAM 3 + BiRefNet preprocessing, DINOv3, the image DiT, the colour model "
                   "and the chain DiT.")

    def load(self, offload, load_chains, auto_download, copy_from_workspace="", gated_source="huggingface",
             load_colors=True):
        H.ensure_weights(auto_download, copy_from_workspace.strip().strip('"'), gated_source)
        device = mm.get_torch_device()
        key = (offload, bool(load_chains), bool(load_colors), str(device), str(H.workspace()))
        if _PIPELINE["key"] != key:
            _PIPELINE.update(key=None, pipe=None)
            mm.unload_all_models()
            mm.soft_empty_cache()
            from hairdeck.infer.image import Pipeline

            with torch.inference_mode(False), H.own_modules():
                pipe = Pipeline(device=device, chains=bool(load_chains), colors=bool(load_colors), offload=offload)
            log.info(f"[HairDeck] pipeline loaded in {pipe.load_sec} s from {H.workspace()}")
            _PIPELINE.update(key=key, pipe=pipe)
        pipe = _PIPELINE["pipe"]
        if load_chains and pipe.chain is None:
            log.warning(f"[HairDeck] the chain DiT did not load, chains are off: {pipe.chain_error}")
        if load_colors and pipe.color is None:
            log.warning(f"[HairDeck] the colour model did not load, one hair colour: {pipe.color_error}")
        return (pipe,)


class HairDeckGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("HAIRDECK_MODEL",),
                "image": ("IMAGE", {"tooltip": "A character image with the head clearly visible. Only the first image "
                                               "of a batch is used."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1,
                                 "tooltip": "Seed of the cards' noise; the chains use the same seed."}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200, "tooltip": "Euler steps of the image DiT."}),
                "cfg": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0, "step": 0.1,
                                  "tooltip": "Classifier-free guidance of the image DiT."}),
                "generate_chains": ("BOOLEAN", {"default": True,
                                                "tooltip": "Generate the spring-bone chains (needs load_chains)."}),
                "filename_prefix": ("STRING", {"default": "HairDeck/hair",
                                               "tooltip": "Result folder under output/: <prefix>_00001_/."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional foreground mask (1 = character). HairDeck finds the character "
                                             "itself; a mask only helps on busy backgrounds."}),
                "invert_mask": ("BOOLEAN", {"default": False,
                                            "tooltip": "Turn on for the MASK of the core Load Image node, which is 1 "
                                                       "where the image is transparent."}),
                # Widgets added after the first release go last: saved workflows store widget values by position.
                "colors": ("BOOLEAN", {"default": True,
                                       "tooltip": "Colour every card with 16 colours from root to tip (needs "
                                                  "load_colors). Off: one hair colour."}),
            },
        }

    RETURN_TYPES = ("HAIRDECK_HAIR", "TRIMESH", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("hair", "mesh", "preprocess", "model_input", "info")
    OUTPUT_TOOLTIPS = (
        "The generated hairstyle (cards, chains, colours and its result folder) for the save, rig and VRM nodes.",
        "The hair cards as one trimesh with UVs, in HairDeck's head frame (Y up, metres): the colour model's colours "
        "as vertex colours on a white material, or one hair colour.",
        "HairDeck's preprocessing panel: masks and the head crop box.",
        "The head crop the model saw.",
        "Counts, timings and the result folder.",
    )
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Generates hair cards, their colours and spring-bone chains from one character image."

    def generate(self, model, image, seed, steps, cfg, generate_chains, filename_prefix, mask=None,
                 invert_mask=False, colors=True):
        from hairdeck.infer.glb import hair_trimesh

        if generate_chains and model.chain is None:
            log.warning("[HairDeck] generate_chains is on but the model was loaded without chains")
        if colors and model.color is None:
            log.warning("[HairDeck] colors is on but the model was loaded without the colour model")
        array = _image_to_array(image, mask, invert_mask)
        out_dir = _new_output_dir(filename_prefix)
        with torch.inference_mode(False), H.own_modules():
            res = model.run(array, seed=int(seed), steps=int(steps), cfg=float(cfg), n_samples=1, out_dir=out_dir,
                            chains=bool(generate_chains), colors=bool(colors))
        if not res["ok"]:
            raise RuntimeError(f"[HairDeck] {res['message']}")
        smp = res["samples"][0]
        card_srgb = None
        if smp.get("colors"):  # the run keeps only the flag; the colours are in the cards file it wrote
            with open(smp["json"], encoding="utf-8") as f:
                card_srgb = [c["colors"] for c in json.load(f)]
        hair = dict(out_dir=out_dir, seed=int(seed), hair_color=[float(c) for c in res["hair_color"]],
                    cards=smp["cards"], card_srgb=card_srgb, chains=smp["chains"] or [], n_cards=smp["n_cards"],
                    n_chains=smp["n_chains"] or 0)
        mesh = hair_trimesh(smp["cards"], res["hair_color"], card_srgb=card_srgb)
        info = (f"{smp['n_cards']} cards, {hair['n_chains']} chains (seed {seed})\n"
                f"colours: {'16 per card (colour model)' if card_srgb else 'one hair colour'}; hair colour (sRGB): "
                f"{', '.join('%.3f' % c for c in res['hair_color'])}\n"
                f"timings: {json.dumps(res['timings'])}\n"
                f"folder: {out_dir}")
        if res["message"]:
            info += "\n" + res["message"]
        for note in res["notes"]:
            if note.startswith(("colours failed", "one hair colour", "chains failed")):
                info += "\n" + note
        return (hair, mesh, _pil_to_tensor(res["prep_vis"]), _pil_to_tensor(res["model_input"]), info)


class HairDeckSaveGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hair": ("HAIRDECK_HAIR",),
                "include_head_proxy": ("BOOLEAN", {"default": False,
                                                   "tooltip": "Add HairDeck's grey head ellipsoid (a fitting aid)."}),
                "show_chains": ("BOOLEAN", {"default": False,
                                            "tooltip": "Draw the spring-bone chains as orange tubes (not bones)."}),
                "filename_prefix": ("STRING", {"default": "3D/HairDeck"}),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB", "STRING")
    RETURN_NAMES = ("model_3d", "glb_path")
    OUTPUT_TOOLTIPS = ("The saved GLB: connect to Preview 3D.", "GLB path relative to the output folder.")
    FUNCTION = "save"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Saves the hair cards as a static GLB (no bones), optionally with the head proxy and the chains drawn."

    def save(self, hair, include_head_proxy, show_chains, filename_prefix):
        from hairdeck.infer.glb import chain_trimesh, hair_trimesh, head_trimesh, write_glb

        path, rel, preview = _output_file(filename_prefix, "glb")
        chains = chain_trimesh(hair["chains"]) if show_chains and hair["chains"] else None
        write_glb(path, hair_trimesh(hair["cards"], hair["hair_color"], card_srgb=hair.get("card_srgb")),
                  head_trimesh() if include_head_proxy else None, chains)
        return {"ui": {"3d": [preview]}, "result": (_file3d(path), rel)}


class HairDeckRiggedGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hair": ("HAIRDECK_HAIR",),
                "filename_prefix": ("STRING", {"default": "3D/HairDeck_rigged"}),
                "blender": (BLENDER_BACKENDS, {
                    "default": "auto",
                    "tooltip": "bpy: the bpy module in ComfyUI's Python (pip install bpy). executable: a Blender "
                               "install (blender_path, HAIRDECK_BLENDER or the newest under Program Files). auto: "
                               "bpy when installed, else the executable."}),
            },
            "optional": {
                "blender_path": ("STRING", {"default": "", "tooltip": "blender.exe for the executable backend."}),
                "avatar": ("HAIRDECK_AVATAR", {"tooltip": "Place and size the hair for this avatar's head (HairDeck "
                                                          "Load Avatar). Without it: HairDeck's bundled avatar."}),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB", "STRING", "STRING")
    RETURN_NAMES = ("model_3d", "glb_path", "info")
    OUTPUT_TOOLTIPS = ("The rigged GLB: connect to Preview 3D.", "GLB path relative to the output folder.",
                       "Bones, chains and vertex groups written.")
    FUNCTION = "rig"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("Rigged GLB of the hair alone: a Head bone plus one bone chain per spring-bone chain, the hair "
                   "skinned with HairDeck's weights, coloured by a per-card colour texture (or one hair colour). No "
                   "physics: the chain stiffness is stored in the armature's extras. Placed and sized for the avatar's "
                   "head: a loaded avatar, or HairDeck's bundled one (about 1.4 m up).")

    def rig(self, hair, filename_prefix, blender, blender_path="", avatar=None):
        from hairdeck.infer.glb import srgb_to_linear
        from hairdeck.vrm import body as B
        from hairdeck.vrm.prep import prepare

        out_dir, seed = hair["out_dir"], hair["seed"]
        bd = B.body(avatar["body"] if avatar else None)
        d = os.path.join(out_dir, f"rig_seed{seed}")
        prepare(out_dir, seed, bd, srgb_to_linear(hair["hair_color"]), os.path.join(d, "run"),
                os.path.join(d, "prep"), os.path.join(d, "prep.log"))
        npz = os.path.join(d, "prep", f"{B.file_name('prep', bd)}.npz")
        path, rel, preview = _output_file(filename_prefix, "glb")
        cmd, uses_bpy = H.blender_cmd(blender, blender_path)
        script = str(H.HERE / "blender" / "rigged_glb.py")
        extra = [] if uses_bpy else ["--factory-startup"]
        log_path = os.path.join(d, "blender.log")
        _run(cmd + ["-b", *extra, "--python", script, "--", npz, path], log_path, H.blender_env(), "rigged GLB")
        res = next((json.loads(x[len("RIGGED "):]) for x in open(log_path, encoding="utf-8", errors="replace")
                    if x.startswith("RIGGED ")), None)
        if res is None or not os.path.isfile(path):
            raise RuntimeError(f"[HairDeck] Blender did not write {path}; see {log_path}")
        info = (f"{res['n_chains']} chains, {res['n_bones']} bones, {res['n_groups']} vertex groups, "
                f"{res['n_verts']} vertices\ncolours: {res['colors']}\navatar: {bd['key']}\nBlender: {' '.join(cmd)}")
        return {"ui": {"3d": [preview]}, "result": (_file3d(path), rel, info)}


class HairDeckExportVRM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hair": ("HAIRDECK_HAIR",),
                "filename_prefix": ("STRING", {"default": "3D/HairDeck_avatar"}),
                "blender": (BLENDER_BACKENDS, {
                    "default": "auto",
                    "tooltip": "bpy: the bpy module in ComfyUI's Python. executable: a Blender install (blender_path, "
                               "HAIRDECK_BLENDER or the newest under Program Files). Both get the VRM Add-on "
                               "automatically, in models/hairdeck/blender_user, never in your own Blender profile. "
                               "auto: bpy when installed."}),
            },
            "optional": {
                "blender_path": ("STRING", {"default": "", "tooltip": "blender.exe for the executable backend."}),
                "avatar": ("HAIRDECK_AVATAR", {"tooltip": "Put the hair on this avatar (HairDeck Load Avatar). "
                                                          "Without it: HairDeck's bundled avatar."}),
            },
        }

    RETURN_TYPES = ("FILE_3D_GLB", "STRING", "STRING")
    RETURN_NAMES = ("model_3d", "vrm_path", "info")
    OUTPUT_TOOLTIPS = ("A GLB copy of the VRM for Preview 3D (VRM is glTF binary).",
                       "VRM path relative to the output folder.",
                       "HairDeck's export summary and validation.")
    FUNCTION = "export"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = ("HairDeck's VRM 0.x export: the hair on an avatar (HairDeck's bundled one, or a loaded one; its own "
                   "hair removed), skinned to new chain bones, with one VRM spring-bone group per chain (the predicted "
                   "stiffness, the avatar's colliders). The physics runs in VRM players (UniVRM, three-vrm, VSeeFace, "
                   "VRoid Hub).")

    def export(self, hair, filename_prefix, blender, blender_path="", avatar=None):
        from hairdeck.vrm import export as X

        cmd, uses_bpy = H.blender_cmd(blender, blender_path)
        # HairDeck's export_vrm runs `HAIRDECK_BLENDER -b --python vrm_export.py -- ...` with this process's
        # environment. bpy_runner.py enables the pinned VRM Add-on (unpacked into the workspace's Blender user folders)
        # and patches it, before vrm_export.py: with bpy it is the command itself, run by ComfyUI's Python; with an
        # executable it is a first --python script, and Blender runs vrm_export.py after it in the same session.
        blender_argv = cmd if uses_bpy else cmd + ["-b", "--python", str(H.RUNNER)]
        H.install_vrm_addon()
        override = dict(H.blender_user_env(), HAIRDECK_BLENDER=json.dumps(blender_argv),
                        HAIRDECK_VRM_ADDON=H.VRM_ADDON_MODULE,
                        HAIRDECK_VRM_WORK=os.path.join(folder_paths.get_temp_directory(), "hairdeck_vrm"))
        body = avatar["body"] if avatar else None
        with H.environ(**override):
            info = X.export_vrm(hair["out_dir"], hair["seed"], hair["hair_color"], body=body, render=False)
        path, rel, _ = _output_file(filename_prefix, "vrm")
        shutil.copyfile(info["vrm"], path)
        # Preview 3D reads .glb; a VRM is a glTF binary with VRM extensions, so a copy under that name previews.
        preview_name = f"hairdeck_vrm_{int(time.time() * 1000)}.glb"
        preview_path = os.path.join(folder_paths.get_temp_directory(), preview_name)
        shutil.copyfile(info["vrm"], preview_path)
        text = (f"{'valid' if info['ok'] else 'PROBLEMS: ' + '; '.join(map(str, info['problems']))}\n"
                f"{info['n_chains']} chains, {info['n_spring_groups']} spring-bone groups, {info['hair_verts']} hair "
                f"vertices, {info['size_mb']} MB\navatar: {avatar['name'] if avatar else 'HairDeck avatar'}, "
                f"its hair removed: {', '.join(n for n, _, _ in info['removed_hair'] or []) or 'none'}\n"
                f"Blender: {' '.join(cmd)}")
        preview = {"filename": preview_name, "subfolder": "", "type": "temp"}
        return {"ui": {"3d": [preview]}, "result": (_file3d(preview_path), rel, text)}


def _input_vrms():
    root = folder_paths.get_input_directory()
    found = []
    for d, _, files in os.walk(root):
        found += [os.path.relpath(os.path.join(d, f), root).replace("\\", "/") for f in files if f.lower().endswith(".vrm")]
    return sorted(found)


class HairDeckLoadAvatar:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vrm": (_input_vrms() or [""], {"tooltip": "A VRM 0.x avatar in ComfyUI's input folder."}),
                "add_head_collider": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When the avatar has no colliders, add a sphere collider on its head bone, so the new "
                               "hair's spring bones don't swing through the head."}),
                "blender": (BLENDER_BACKENDS, {"default": "auto",
                                               "tooltip": "Blender for the one-time preparation, as on Export VRM."}),
            },
            "optional": {
                "blender_path": ("STRING", {"default": "", "tooltip": "blender.exe for the executable backend."}),
                "vrm_path": ("STRING", {"default": "",
                                        "tooltip": "Full path of a .vrm file anywhere on disk (quotes allowed). When "
                                                   "set, it is used instead of the vrm list."}),
            },
        }

    RETURN_TYPES = ("HAIRDECK_AVATAR", "STRING")
    RETURN_NAMES = ("avatar", "info")
    OUTPUT_TOOLTIPS = ("The avatar for Export VRM and Rigged GLB.", "What was found in the avatar.")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("Prepares your own VRM 0.x avatar for Export VRM and Rigged GLB, in place of HairDeck's bundled "
                   "avatar: a file from ComfyUI's input folder, or any file through vrm_path. Its hair meshes ('hair' "
                   "in the name) are replaced, and its head is measured from a mesh with 'face' or 'head' in the "
                   "name. Done once per file, cached in models/hairdeck/avatars.")

    @staticmethod
    def _path(vrm, vrm_path=""):
        """vrm_path when set (surrounding quotes, ~ and environment variables allowed), else the input-folder file."""
        p = (vrm_path or "").strip().strip('"').strip("'")
        if p:
            return os.path.abspath(os.path.expandvars(os.path.expanduser(p)))
        return folder_paths.get_annotated_filepath(vrm) if vrm else ""

    @classmethod
    def IS_CHANGED(cls, vrm, vrm_path="", **kwargs):
        path = cls._path(vrm, vrm_path)
        return f"{path}|{os.path.getmtime(path)}" if path and os.path.isfile(path) else path

    def load(self, vrm, add_head_collider, blender, blender_path="", vrm_path=""):
        from .avatar import build_avatar

        path = self._path(vrm, vrm_path)
        if vrm_path and vrm_path.strip() and not os.path.isfile(path):
            raise FileNotFoundError(f"[HairDeck] vrm_path: no such file: {path}")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError("[HairDeck] select a .vrm file from ComfyUI's input folder, or set vrm_path")
        a = build_avatar(path, blender, blender_path, bool(add_head_collider))
        s = a["info"]
        collider = s["head_collider_added"]
        info = (f"{a['name']}: hair meshes {', '.join(s['hair_meshes'])} (replaced), head measured on "
                f"{', '.join(s['head_meshes'])} ({s['head_width_m']} m wide)\n"
                f"spring groups {s['spring_groups']}, collider groups {s['collider_groups']}"
                + (f"; head collider added (radius {collider['radius']:.3f} m)" if collider else ""))
        return (a, info)


NODE_CLASS_MAPPINGS = {
    "HairDeckModelLoader": HairDeckModelLoader,
    "HairDeckGenerate": HairDeckGenerate,
    "HairDeckSaveGLB": HairDeckSaveGLB,
    "HairDeckRiggedGLB": HairDeckRiggedGLB,
    "HairDeckExportVRM": HairDeckExportVRM,
    "HairDeckLoadAvatar": HairDeckLoadAvatar,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HairDeckModelLoader": "HairDeck Model Loader",
    "HairDeckGenerate": "HairDeck Generate Hair",
    "HairDeckSaveGLB": "HairDeck Save GLB",
    "HairDeckRiggedGLB": "HairDeck Rigged GLB",
    "HairDeckExportVRM": "HairDeck Export VRM",
    "HairDeckLoadAvatar": "HairDeck Load Avatar",
}
