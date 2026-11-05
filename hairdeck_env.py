"""Where HairDeck's code, weights and Blender come from inside ComfyUI.

* **Code**: the HairDeck release is vendored, unmodified, in ``HairDeck/`` (see
  ``HairDeck/UPSTREAM``) and put first on ``sys.path``.
* **Weights**: a HairDeck workspace (``ckpt/``, ``data/``, ``third_party/``)
  at ``HAIRDECK_ROOT`` when that is set, otherwise ``models/hairdeck``. Missing
  items are fetched by HairDeck's own downloader (sha256-checked), either from
  Hugging Face or copied from another workspace.
* **Blender**: the ``bpy`` module in ComfyUI's own Python, run in a child
  process through ``blender/bpy_runner.py`` so Blender's global state and its
  DLLs never enter the ComfyUI process. A Blender executable is the fallback.
  Blender's user folders (the VRM Add-on) live in the workspace, so the user's
  own Blender profile is never touched.
"""

from __future__ import annotations

import contextlib
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import folder_paths

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "HairDeck"
RUNNER = HERE / "blender" / "bpy_runner.py"

if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

MODELS_DIR = Path(folder_paths.models_dir) / "hairdeck"
# Workspace items needed for inference, in HairDeck's downloader names.
ITEMS = ("hairdeck", "birefnet", "sam3", "sam3_src", "dinov3_src", "dinov3")


def workspace() -> Path:
    """The workspace root; also exported as HAIRDECK_ROOT for HairDeck's paths module."""
    root = os.environ.get("HAIRDECK_ROOT")
    if not root:
        root = str(MODELS_DIR)
        os.environ["HAIRDECK_ROOT"] = root
    return Path(root)


def _downloader():
    """HairDeck's scripts/setup/download_weights.py as a module (for its pinned constants and url_get)."""
    spec = importlib.util.spec_from_file_location(
        "hairdeck_download_weights", VENDOR / "scripts" / "setup" / "download_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def missing_items() -> list[str]:
    """Inference items whose files are not all present (a presence check, not the downloader's sha256 pass)."""
    workspace()
    from hairdeck import paths
    from hairdeck.infer.image import CHAIN_DIT, COLOR_DIT, IMAGE_DIT

    root = Path(paths.get("root"))
    out = []
    try:
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
        files = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
        test = cfg.get("data", {}).get("test_data", {}).get("files", [])
        rels = [r for r in files if not any(r == e or (e.endswith("/") and r.startswith(e)) for e in test)]
        dl = _downloader()
        # A MANIFEST from an older release does not list newer checkpoints (the colour model came after the first
        # release), so the checkpoints the vendored pipeline loads are required by name as well; the downloader then
        # fetches the current MANIFEST and what is missing from it.
        ckpts = [paths.ckpt(*c) for c in (IMAGE_DIT, CHAIN_DIT, COLOR_DIT)]
        if not all(os.path.isfile(dl.dst_of(r)) for r in rels) or not all(os.path.isfile(c) for c in ckpts):
            out.append("hairdeck")
    except (OSError, ValueError, KeyError):
        out.append("hairdeck")
    birefnet = ("model.safetensors", "birefnet.py", "BiRefNet_config.py", "config.json")
    if not all(os.path.isfile(paths.weights("birefnet_matting", f)) for f in birefnet):
        out.append("birefnet")
    if not os.path.isfile(paths.weights("sam3", "sam3.pt")):
        out.append("sam3")
    if not os.path.isdir(os.path.join(paths.sam3_repo(), "sam3")):
        out.append("sam3_src")
    if not os.path.isfile(os.path.join(paths.dinov3_repo(), "hubconf.py")):
        out.append("dinov3_src")
    if not os.path.isfile(paths.weights("dinov3", "model.safetensors")):
        out.append("dinov3")
    return out


def download(items, source: str = "", gated_source: str = "huggingface") -> None:
    """Run HairDeck's downloader for the items: from Hugging Face, or copied from the workspace `source`."""
    cmd = [sys.executable, str(VENDOR / "scripts" / "setup" / "download_weights.py"),
           "--root", str(workspace()), "--only", *items, "--gated-source", gated_source]
    if source:
        cmd += ["--from", source]
    print(f"[HairDeck] fetching {', '.join(items)} into {workspace()}"
          + (f" from {source}" if source else ""), flush=True)
    env = dict(os.environ, PYTHONUTF8="1")
    r = subprocess.run(cmd, env=env, cwd=str(VENDOR))
    if r.returncode != 0:
        raise RuntimeError(
            f"[HairDeck] downloading {', '.join(items)} failed (see the console). SAM 3 and DINOv3 are gated: accept "
            "their licences on huggingface.co/facebook/sam3 and huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m "
            "and log in with `hf auth login`, or set gated_source to modelscope.")


def ensure_weights(auto_download: bool, source: str = "", gated_source: str = "huggingface") -> Path:
    todo = missing_items()
    if todo:
        if not auto_download and not source:
            raise FileNotFoundError(
                f"[HairDeck] missing in {workspace()}: {', '.join(todo)}. Enable auto_download, or set "
                "copy_from_workspace to an existing HairDeck workspace.")
        download(todo, source, gated_source)
        left = missing_items()
        if left:
            raise FileNotFoundError(f"[HairDeck] still missing after the download: {', '.join(left)}")
    return workspace()


# ---------------------------------------------------------------------------
# Third-party code that HairDeck imports by its top-level name
# ---------------------------------------------------------------------------

# SAM 3 and DINOv3 are imported as `sam3` and `dinov3`. Other node packs bundle their own copies under the same names
# (ComfyUI-RMBG imports its `sam3` at startup, whose build_sam3_image_model looks for its vocabulary file elsewhere),
# and Python keeps one module per name. While a HairDeck step runs, its own copies take those names; afterwards the
# other pack's modules are put back, so each pack only ever sees its own. ComfyUI runs one node at a time.
SHARED_MODULES = ("sam3", "dinov3")
_own_modules: dict = {}


def _take_shared() -> dict:
    return {n: sys.modules.pop(n) for n in list(sys.modules) if n.split(".", 1)[0] in SHARED_MODULES}


@contextlib.contextmanager
def own_modules():
    """HairDeck's sam3 / dinov3 (models/hairdeck/third_party) under their names for the duration."""
    workspace()
    from hairdeck import paths

    others = _take_shared()
    sys.modules.update(_own_modules)
    repo = paths.sam3_repo()  # dinov3 needs no path entry: torch.hub.load(source='local') adds its own
    sys.path.insert(0, repo)
    try:
        yield
    finally:
        sys.path.remove(repo)  # the first occurrence: the one inserted above
        _own_modules.clear()
        _own_modules.update(_take_shared())
        sys.modules.update(others)


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------

def has_bpy() -> bool:
    return importlib.util.find_spec("bpy") is not None


def find_blender_exe(explicit: str = "") -> str | None:
    """An explicit path, HAIRDECK_BLENDER, the newest Blender under Program Files, or `blender` on PATH."""
    for cand in (explicit, os.environ.get("HAIRDECK_BLENDER", "")):
        cand = cand.strip().strip('"')
        if cand and os.path.isfile(cand):
            return cand
    found = glob.glob(os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                   "Blender Foundation", "Blender *", "blender.exe"))
    if found:
        def version(p):
            v = Path(p).parent.name.split()[-1]
            return tuple(int(x) for x in v.split(".") if x.isdigit())
        return max(found, key=version)
    return shutil.which("blender")


def blender_user_dir() -> Path:
    """Blender's user folders (config, extensions) for HairDeck, inside the workspace."""
    return workspace() / "blender_user"


def blender_user_env() -> dict:
    """The variables that point a Blender (bpy module or executable) at blender_user_dir()."""
    user = blender_user_dir()
    return dict(BLENDER_USER_RESOURCES=str(user), BLENDER_USER_CONFIG=str(user / "config"),
                BLENDER_USER_SCRIPTS=str(user / "scripts"), BLENDER_USER_DATAFILES=str(user / "datafiles"),
                BLENDER_USER_EXTENSIONS=str(user / "extensions"),
                BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION="true", PYTHONUTF8="1")


def blender_env() -> dict:
    """A child environment: this one plus blender_user_env()."""
    return dict(os.environ, **blender_user_env())


def blender_cmd(backend: str, blender_path: str = "") -> tuple[list[str], bool]:
    """(argv prefix that takes Blender's command line (-b --python SCRIPT -- ARGS), whether it is the bpy module)."""
    if backend in ("auto", "bpy") and has_bpy():
        return [sys.executable, str(RUNNER)], True
    if backend == "bpy":
        raise RuntimeError("[HairDeck] the bpy module is not installed in ComfyUI's Python "
                           "(pip install bpy==5.2.3 for Python 3.13), or choose the executable backend")
    exe = find_blender_exe(blender_path)
    if not exe:
        raise RuntimeError("[HairDeck] no Blender found: install the bpy module, or set blender_path / HAIRDECK_BLENDER")
    return [exe], False


@contextlib.contextmanager
def environ(**values):
    """os.environ with the values set (None removes one) for the duration; every key restored afterwards."""
    saved = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


VRM_ADDON_MODULE = "bl_ext.user_default.vrm"


def install_vrm_addon() -> Path:
    """The VRM Add-on for Blender that HairDeck pins (downloaded once, sha256-checked), unpacked into the
    user_default extensions repository of blender_user_dir(). Unpacked in place, with the sha256 stamp written last as
    the completion mark: Blender's own installer (and any extract-then-rename) unpacks to a temporary folder and
    renames it, which Windows refuses while a virus scanner still holds the new files."""
    import time
    import zipfile

    dl = _downloader()
    url, sha = dl.VRM_ADDON
    archive = workspace() / "third_party" / "blender" / url.rsplit("/", 1)[1]
    dl.url_get(url, str(archive), sha)
    dst = blender_user_dir() / "extensions" / "user_default" / "vrm"
    stamp = dst / ".hairdeck_sha256"
    if stamp.is_file() and stamp.read_text().strip() == sha:
        return dst
    for attempt in range(10):  # an interrupted or older unpack: files may still be held by a scanner for a moment
        try:
            if dst.exists():
                shutil.rmtree(dst)
            break
        except OSError:
            if attempt == 9:
                raise
            time.sleep(0.5)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dst)
    stamp.write_text(sha + "\n")
    return dst
