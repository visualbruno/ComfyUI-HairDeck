"""Whole-character soft alpha of the preprocessing: BiRefNet-matting (ZhengPeng7/BiRefNet-matting, MIT).

The model code is the Hugging Face repository's own remote code (birefnet.py and BiRefNet_config.py: a plain PyTorch
model in a transformers PreTrainedModel wrapper, no network access), loaded from weights/birefnet_matting/ together
with model.safetensors. Preprocessing as in the model card: resize to size x size, ImageNet normalisation, sigmoid of
the last output, in half precision.
"""
import importlib.util
import sys
import types

import torch
import torch.nn.functional as F

from .. import dev, paths


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class BiRefNet:
    def __init__(self, name='birefnet_matting', size=1024, device=None, inference_device=None):
        # Model storage may start on CPU; small buffers and inputs use the selected GPU.
        device = dev.select(device)
        self.device = torch.device(device if inference_device is None else inference_device)
        code_dir = paths.weights(name)
        pkg = f'birefnet_pkg_{name}'  # birefnet.py does `from .BiRefNet_config import ...`
        sys.modules[pkg] = types.ModuleType(pkg)
        sys.modules[pkg].__path__ = [code_dir]
        _load_module(f'{code_dir}/BiRefNet_config.py', f'{pkg}.BiRefNet_config')
        mod = _load_module(f'{code_dir}/birefnet.py', f'{pkg}.birefnet')
        self.net = mod.BiRefNet(bb_pretrained=False)
        from safetensors.torch import load_file
        sd = load_file(f'{code_dir}/model.safetensors')
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
        self.net.eval().to(device).half()
        self.size = size
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, rgb):
        h0, w0 = rgb.shape[:2]
        t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float() / 255
        t = F.interpolate(t, (self.size, self.size), mode='bilinear', align_corners=False)
        t = ((t - self.mean) / self.std).half()
        a = self.net(t)[-1].sigmoid().float()
        return F.interpolate(a, (h0, w0), mode='bilinear', align_corners=False)[0, 0]


def build(name, size, device=None, inference_device=None):
    if name != 'birefnet_matting':
        raise ValueError(f'matting model {name!r}: the release has birefnet_matting only')
    return BiRefNet(name, size or 1024, device=device, inference_device=inference_device)
