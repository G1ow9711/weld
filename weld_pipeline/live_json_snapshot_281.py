"""Read atomic Windows JSON snapshots while permitting replacement by writers."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path


def read_json_snapshot(path, *, maximum_bytes=8 * 1024 * 1024):
    """Share deletion so a sealed old snapshot can be read during atomic rename."""
    path = Path(path)
    if os.name != "nt":
        data = path.read_bytes()
        if len(data) > maximum_bytes:
            raise ValueError("Live JSON snapshot exceeds its explicit size budget")
        return json.loads(data.decode("utf-8-sig"))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    kernel.ReadFile.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path.resolve()), 0x80000000, 1 | 2 | 4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    blocks, total = [], 0
    try:
        buffer = ctypes.create_string_buffer(65536)
        while True:
            count = wintypes.DWORD()
            if not kernel.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if not count.value:
                break
            total += count.value
            if total > maximum_bytes:
                raise ValueError("Live JSON snapshot exceeds its explicit size budget")
            blocks.append(buffer.raw[:count.value])
    finally:
        kernel.CloseHandle(handle)
    return json.loads(b"".join(blocks).decode("utf-8-sig"))


def read_json_when_available(path):
    try:
        return read_json_snapshot(path)
    except (PermissionError, FileNotFoundError, json.JSONDecodeError):
        return None
