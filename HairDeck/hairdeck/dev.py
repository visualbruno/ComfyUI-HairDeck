"""Device selection and the device-agnostic autocast / synchronise / noise helpers.

The release targets one CUDA GPU; the same code runs on Apple MPS through these helpers. HAIRDECK_DEVICE picks the
device: ``cuda`` (the default off macOS, as before), ``mps``, ``cpu``, or ``favour`` (CUDA, else Apple MPS, else CPU;
the default on macOS, which has no CUDA).
"""
import os
import sys

import torch

DEFAULT = 'favour' if sys.platform == 'darwin' else 'cuda'


def select(explicit=None):
    """torch.device of `explicit` or HAIRDECK_DEVICE, ``favour`` resolved to the best available device."""
    if explicit is not None and not isinstance(explicit, str):
        return torch.device(explicit)
    want = (explicit or os.environ.get('HAIRDECK_DEVICE') or DEFAULT).lower()
    if want in ('favour', 'favor', 'auto'):
        want = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(want)


def amp(device, enabled=True):
    """bf16 autocast as the release uses it on CUDA, on Apple MPS too; a no-op on the CPU."""
    kind = torch.device(device).type
    enabled = bool(enabled) and kind in ('cuda', 'mps')
    return torch.autocast(kind if kind in ('cuda', 'mps') else 'cpu', dtype=torch.bfloat16, enabled=enabled)


def sync(device):
    """Wait for the device's queued work."""
    kind = torch.device(device).type
    if kind == 'cuda':
        torch.cuda.synchronize(device)
    elif kind == 'mps':
        torch.mps.synchronize()


def generator(device, seed=None):
    """Seeded generator of `device`'s RNG; MPS has no seeded RNG, so its noise comes from a CPU generator."""
    return torch.Generator(device='cpu' if torch.device(device).type == 'mps' else device).manual_seed(int(seed or 0))


def randn(shape, device, gen=None):
    """torch.randn of `shape` on `device` (see generator())."""
    if torch.device(device).type == 'mps':
        return torch.randn(shape, generator=gen).to(device)
    return torch.randn(shape, device=device, generator=gen)


def peak_gb(device):
    """Peak memory allocated by this process on the device, in GiB (0 where the device keeps no such counter)."""
    if torch.device(device).type != 'cuda':
        return 0.0
    return round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
