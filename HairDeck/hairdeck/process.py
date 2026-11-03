"""Small process helpers shared with Blender's bundled Python (standard library only)."""
import ctypes
from ctypes import wintypes
import errno
import os
import subprocess
import sys


def _windows_process_alive(pid):
    # A zero-time wait tests a process handle without sending a signal. In contrast,
    # os.kill(pid, 0) terminates the target process on Windows.
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: this PID no longer exists
            return False
        if error == 5:  # ERROR_ACCESS_DENIED: exists, but belongs to another security context
            return True
        raise ctypes.WinError(error)
    try:
        result = kernel.WaitForSingleObject(handle, 0)
        if result == 0x00000102:  # WAIT_TIMEOUT: process has not exited
            return True
        if result == 0:  # WAIT_OBJECT_0: process has exited
            return False
        raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def process_alive(pid):
    """Whether a positive PID exists, without changing its state on either platform."""
    pid = int(pid)
    if pid <= 0:
        return False
    if sys.platform == 'win32':
        return _windows_process_alive(pid)
    try:
        os.kill(pid, 0)
    except OSError as error:
        if error.errno == errno.ESRCH:
            return False
        if error.errno == errno.EPERM:
            return True
        raise
    return True


def background_process_kwargs():
    """Detach a background worker from console signals without POSIX-only flags on Windows."""
    if sys.platform == 'win32':
        return {'creationflags': (getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x00000200)
                                  | getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000))}
    return {'start_new_session': True}
