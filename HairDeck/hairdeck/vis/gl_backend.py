"""Rendering backend selection, before importing pyrender or OpenGL."""
import os
import sys


def render_env(env=None):
    """Copy an environment with a desktop Windows/macOS or headless Linux rendering backend.

    HAIRDECK_GL overrides PYOPENGL_PLATFORM. ``pyglet`` selects pyrender's default
    desktop backend by leaving PYOPENGL_PLATFORM unset; it is not a PyOpenGL plugin.
    """
    e = dict(os.environ if env is None else env)
    desktop = sys.platform in ('win32', 'darwin')  # EGL is headless Linux only; macOS has no EGL
    backend = e.get('HAIRDECK_GL') or e.get('PYOPENGL_PLATFORM') or ('pyglet' if desktop else 'egl')
    if backend not in ('pyglet', 'egl', 'osmesa'):
        raise ValueError('HAIRDECK_GL must be pyglet (desktop OpenGL), egl, or osmesa; got ' + repr(backend))
    if backend == 'pyglet':
        # Retain the choice when configure_gl() is called more than once.
        e['HAIRDECK_GL'] = 'pyglet'
        e.pop('PYOPENGL_PLATFORM', None)
        e.pop('EGL_DEVICE_ID', None)
    else:
        e['PYOPENGL_PLATFORM'] = backend
        if backend == 'egl':
            e.setdefault('EGL_DEVICE_ID', e.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0] or '0')
        else:
            e.pop('EGL_DEVICE_ID', None)
    return e


def configure_gl():
    """Apply the backend to this process; call before importing any OpenGL library."""
    e = render_env()
    for name in ('HAIRDECK_GL', 'PYOPENGL_PLATFORM', 'EGL_DEVICE_ID'):
        if name in e:
            os.environ[name] = e[name]
        else:
            os.environ.pop(name, None)
