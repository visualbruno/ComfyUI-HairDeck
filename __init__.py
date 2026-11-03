"""ComfyUI-HairDeck: animation-ready stylized hair cards from a single image.

Registers ``models/hairdeck`` (HairDeck's workspace: checkpoints, data, third-party weights) and exposes the nodes.
"""

import os

import folder_paths

HAIRDECK_MODEL_DIR = os.path.join(folder_paths.models_dir, "hairdeck")

try:
    os.makedirs(HAIRDECK_MODEL_DIR, exist_ok=True)
    folder_paths.add_model_folder_path("hairdeck", HAIRDECK_MODEL_DIR, is_default=True)
except Exception:
    # A read-only models directory is not a reason to fail the import; HAIRDECK_ROOT can point elsewhere.
    pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
