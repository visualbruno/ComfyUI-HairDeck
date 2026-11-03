"""SAM 3 text-prompt grounding of the preprocessing: the image model with its processor (confidence 0.5), the text
features of all prompts computed once, and one batched grounding pass of several prompts on an image.

SAM 3 (facebookresearch/sam3, SAM License) is imported on first use: either the installed sam3 package or a clone at
hairdeck.paths.sam3_repo(); the checkpoint is weights/sam3/sam3.pt (the gated facebook/sam3 download).
"""
import os
import sys

import torch

from .. import dev, paths


def _triton_stub():
    """macOS has no Triton wheels; SAM 3 imports Triton at import time even with compilation off (hairdeck/vendor)."""
    if sys.platform != 'darwin':  # other platforms keep their own Triton (or their own import error)
        return
    try:
        import triton  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vendor'))


def _sam3():
    _triton_stub()
    try:
        try:
            __import__('sam3')
        except ModuleNotFoundError as exc:
            # A missing dependency of an installed SAM 3 is not a missing SAM 3
            # package. Preserve that failure instead of trying another checkout.
            if exc.name != 'sam3':
                raise
            repo = paths.sam3_repo()
            if not os.path.isdir(repo):
                raise ImportError('SAM 3 source is missing: run python scripts/setup/download_weights.py --only sam3_src '
                                  'or set HAIRDECK_SAM3_REPO to a SAM 3 checkout') from None
            sys.path.insert(0, repo)
        from sam3.model.data_misc import FindStage
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ModuleNotFoundError as exc:
        if sys.platform == 'win32' and (exc.name or '').split('.')[0] == 'triton':
            raise ImportError('SAM 3 requires Triton at import time, even with compilation disabled. '
                              'For native Windows, run python -m pip install -r requirements-windows.txt '
                              'from the HairDeck repository in this Python environment. '
                              'The Windows requirements pair PyTorch 2.8 with triton-windows 3.4.') from exc
        raise
    return FindStage, Sam3Processor, build_sam3_image_model


def load_model(compile=False, device=None, inference_device=None):
    """Build weights on device; keep processor tensors on the inference device during CPU offload."""
    device = dev.select(device)
    _, Sam3Processor, build_sam3_image_model = _sam3()
    model = build_sam3_image_model(checkpoint_path=paths.weights('sam3', 'sam3.pt'), load_from_HF=False, compile=compile,
                                  device=str(device))
    compute_device = device if inference_device is None else inference_device
    return model, Sam3Processor(model, device=str(compute_device), confidence_threshold=0.5)


@torch.inference_mode()
def text_feats(model, prompts, device=None):
    device = dev.select(device)
    with dev.amp(device):
        return model.backbone.forward_text(prompts, device=str(device))


@torch.inference_mode()
def ground(model, state, tf, idx, device=None):
    """Grounding for the prompts idx (indices into tf) on the image in state; returns probs (P, Q), boxes (P, Q, 4)
    cxcywh normalised, mask logits (P, Q, h, w)."""
    FindStage = _sam3()[0]
    device = dev.select(device)
    n = len(idx)
    bo = dict(state['backbone_out'])
    bo['language_features'] = tf['language_features'][:, idx]
    bo['language_mask'] = tf['language_mask'][idx]
    if 'language_embeds' in tf:
        bo['language_embeds'] = tf['language_embeds'][:, idx] if tf['language_embeds'].dim() == 3 else tf['language_embeds']
    fs = FindStage(img_ids=torch.zeros(n, device=device, dtype=torch.long),
                   text_ids=torch.arange(n, device=device, dtype=torch.long),
                   input_boxes=None, input_boxes_mask=None, input_boxes_label=None, input_points=None,
                   input_points_mask=None)
    out = model.forward_grounding(backbone_out=bo, find_input=fs, geometric_prompt=model._get_dummy_prompt(n),
                                  find_target=None)
    probs = (out['pred_logits'].sigmoid() * out['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
    return probs.float(), out['pred_boxes'].float(), out['pred_masks']
