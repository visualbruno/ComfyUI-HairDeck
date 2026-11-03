"""Blender executable and child-process environment for the managed Linux/Windows install.

HAIRDECK_BLENDER may be an executable path or a JSON argv array. Legacy shell-style
commands remain supported; no command is executed through a shell. An explicit
override uses its own Blender profile and must already have the VRM Add-on enabled.
"""
import json
import os
import shlex
import sys

from . import paths

VERSION = '4.2.23'
# https://download.blender.org/release/Blender4.2/blender-4.2.23.sha256
BUILDS = {
    'linux': ('https://download.blender.org/release/Blender4.2/blender-4.2.23-linux-x64.tar.xz',
              'bea0eb3146be13eae6225409a117b215184f41b7f79e799f97cb3abb8f6dc404',
              'blender-4.2.23-linux-x64', 'blender'),
    'win32': ('https://download.blender.org/release/Blender4.2/blender-4.2.23-windows-x64.zip',
              '82e791475779a7342424a480bdde9a20b43710da9264c60346125aa16cd910cb',
              'blender-4.2.23-windows-x64', 'blender.exe'),
}


def blender_build():
    """Official archive, checksum, directory and executable for the host platform."""
    key = 'linux' if sys.platform.startswith('linux') else sys.platform
    if key not in BUILDS:
        raise RuntimeError('The managed Blender installer supports Linux x64 and Windows x64. '
                           'For an existing Blender 4.2 LTS with the VRM Add-on, set HAIRDECK_BLENDER.')
    return BUILDS[key]


def managed_dir():
    return os.path.abspath(paths.root('third_party', 'blender'))


def blender_cmd():
    """An argv prefix, preserving Windows path separators and spaces."""
    command = os.environ.get('HAIRDECK_BLENDER', '').strip()
    if not command:
        _, _, directory, executable = blender_build()
        return [os.path.join(managed_dir(), directory, executable)]
    if command.startswith('['):
        argv = json.loads(command)
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            raise ValueError('HAIRDECK_BLENDER JSON must be a nonempty array of nonempty strings.')
        return argv
    # An unquoted executable path with spaces is a common Windows environment value.
    path = os.path.expanduser(command)
    if os.path.isfile(path) or (sys.platform == 'win32' and path.lower().endswith('.exe')):
        return [path]
    argv = shlex.split(command, posix=sys.platform != 'win32')
    if sys.platform == 'win32':
        argv = [x[1:-1] if len(x) >= 2 and x[0] == x[-1] and x[0] in ('"', "'") else x for x in argv]
        if len(argv) > 1 and any(x.lower().endswith('.exe') for x in argv[1:]):
            raise ValueError('Quote the executable path in HAIRDECK_BLENDER, or use a JSON argv array for arguments.')
    if not argv or not argv[0]:
        raise ValueError('HAIRDECK_BLENDER must name an executable.')
    argv[0] = os.path.expanduser(argv[0])
    return argv


def blender_env(env=None):
    """Copy the environment; isolate managed Blender resources without changing the parent process."""
    child = dict(os.environ if env is None else env)
    if os.environ.get('HAIRDECK_BLENDER', '').strip():
        return child
    base = managed_dir()
    home, temp, user = (os.path.join(base, name) for name in ('home', 'tmp', 'user'))
    for directory in (home, temp, user):
        os.makedirs(directory, exist_ok=True)
    child.update(HOME=home, XDG_CONFIG_HOME=os.path.join(home, '.config'),
                 XDG_CACHE_HOME=os.path.join(home, '.cache'), XDG_DATA_HOME=os.path.join(home, '.local', 'share'),
                 TMPDIR=temp, TEMP=temp, TMP=temp,
                 BLENDER_USER_RESOURCES=user, BLENDER_USER_CONFIG=os.path.join(user, 'config'),
                 BLENDER_USER_SCRIPTS=os.path.join(user, 'scripts'),
                 BLENDER_USER_DATAFILES=os.path.join(user, 'datafiles'),
                 BLENDER_USER_EXTENSIONS=os.path.join(user, 'extensions'),
                 __GL_SHADER_DISK_CACHE_PATH=os.path.join(home, 'nvcache'),
                 MESA_SHADER_CACHE_DIR=os.path.join(home, 'mesacache'),
                 BLENDER_VRM_AUTOMATIC_LICENSE_CONFIRMATION='true')
    if sys.platform == 'win32':
        child.update(USERPROFILE=home, APPDATA=os.path.join(home, 'AppData', 'Roaming'),
                     LOCALAPPDATA=os.path.join(home, 'AppData', 'Local'))
    child.pop('DISPLAY', None)
    child.pop('WAYLAND_DISPLAY', None)
    return child
