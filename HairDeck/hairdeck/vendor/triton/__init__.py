"""Import-only stand-in for Triton, which has no macOS wheels.

SAM 3 imports Triton and decorates its CUDA kernels with @triton.jit at import time; those kernels only ever run on CUDA
(edt_triton asserts is_cuda, the perflib kernels run under nvcc), so here the decorators return the plain Python function
and every other attribute is an inert placeholder. Installed by hairdeck.prep.sam3_ground._sam3 on macOS only.
shortcut: a Triton kernel cannot run on this platform; use real Triton if it ever must.
"""


class _Any:
    """Any Triton attribute: callable (a decorator returns its function), subscriptable, and inert."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]  # @triton.jit, or @triton.autotune(...) applied to a kernel
        return self

    def __getattr__(self, name):
        return _Any(f'{self._name}.{name}')

    def __getitem__(self, key):
        return self

    def __repr__(self):
        return f'<triton stub {self._name}>'


def __getattr__(name):
    return _Any(f'triton.{name}')


jit = _Any('triton.jit')
autotune = _Any('triton.autotune')
cdiv = _Any('triton.cdiv')

from . import language  # noqa: E402,F401  (after jit: `import triton.language as tl` must resolve)
