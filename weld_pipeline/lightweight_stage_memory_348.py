"""Scalar Windows memory events with deduplicated known CPU backing storage."""
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


class _Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage",
        "PeakPagefileUsage", "PrivateUsage")]


class _System(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
        "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


def known_cpu_storage_summary(objects):
    intervals, logical, visited = [], {}, set()
    def visit(name, value):
        if value is None:
            return
        if torch.is_tensor(value):
            if value.device.type != "cpu":
                raise ValueError("Registered CPU storage profiler received another device")
            logical[name] = int(value.numel() * value.element_size())
            store = value.storage()
            size, pointer = int(store.nbytes()), int(store.data_ptr())
            if size:
                intervals.append((pointer, pointer + size))
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return
            logical[name] = int(value.nbytes)
            base = value
            while isinstance(base.base, np.ndarray):
                base = base.base
            if torch.is_tensor(base.base):
                store = base.base.storage()
                pointer, size = int(store.data_ptr()), int(store.nbytes())
            else:
                pointer, size = int(base.__array_interface__["data"][0]), int(base.nbytes)
            if size:
                intervals.append((pointer, pointer + size))
        elif isinstance(value, torch.nn.Module):
            for key, tensor in value.named_parameters():
                visit(name + ".parameter." + key, tensor)
            for key, tensor in value.named_buffers():
                visit(name + ".buffer." + key, tensor)
        elif isinstance(value, dict):
            if id(value) in visited:
                return
            visited.add(id(value))
            for key, tensor in value.items():
                visit(name + "." + str(key), tensor)
        elif isinstance(value, (tuple, list)):
            if id(value) in visited:
                return
            visited.add(id(value))
            for index, tensor in enumerate(value):
                visit(name + "." + str(index), tensor)
    visit("known", objects)
    unique = 0
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    unique = sum(end - start for start, end in merged)
    compact_logical = {}
    for name, count in logical.items():
        category = name.split(".parameter.", 1)[0].split(".buffer.", 1)[0]
        compact_logical[category] = compact_logical.get(category, 0) + count
    return dict(logical_cpu_bytes_including_views_and_repeated_references=sum(logical.values()),
        unique_known_cpu_backing_storage_bytes=unique, unique_known_cpu_backing_intervals=len(merged),
        logical_cpu_bytes_by_object=compact_logical, complete_python_or_native_heap_not_enumerated=True)


class StageMemoryEvents:
    """Streams records; holds no transient tensor, view, event history or profiler stack."""
    def __init__(self, path, persistent=None):
        self.path = Path(path).resolve()
        root = Path(__file__).resolve().parents[1]
        if not self.path.is_relative_to(root):
            raise ValueError("Stage-memory output must remain project-local")
        self.stream = self.path.open("x", encoding="utf-8", newline="\n", buffering=65536)
        self.persistent = dict(persistent or {})
        self.event_count = 0
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.GetCurrentProcess.restype = ctypes.c_void_p
        self.handle = self.kernel.GetCurrentProcess()
        self.kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_uint64)] * 4
        times = [ctypes.c_uint64() for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.handle, *[ctypes.byref(v) for v in times]):
            raise ctypes.WinError(ctypes.get_last_error())
        self.identity = dict(process_id=os.getpid(), creation_filetime_ticks=times[0].value)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]

    def emit(self, stage, edge, known_objects=None, **values):
        counters = _Counters(); counters.cb = ctypes.sizeof(counters)
        system = _System(); system.dwLength = ctypes.sizeof(system)
        if not self.psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        if not self.kernel.GlobalMemoryStatusEx(ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        known = known_cpu_storage_summary(dict(persistent=self.persistent, stage_objects=known_objects))
        row = dict(utc=datetime.now(timezone.utc).isoformat(), monotonic_ns=time.monotonic_ns(),
            stage=stage, edge=edge, identity=self.identity, event_ordinal=self.event_count,
            private_commit_bytes=int(counters.PrivateUsage), working_set_bytes=int(counters.WorkingSetSize),
            lifecycle_peak_commit_bytes=int(counters.PeakPagefileUsage),
            peak_working_set_bytes=int(counters.PeakWorkingSetSize), available_physical_bytes=int(system.ullAvailPhys),
            remaining_commit_bytes=int(system.ullAvailPageFile),
            unassigned_private_commit_minus_known_cpu_storage_bytes=int(counters.PrivateUsage) - known["unique_known_cpu_backing_storage_bytes"],
            unassigned_difference_is_not_an_allocator_cache_measurement=True,
            external_original_Popen_watch_remains_resource_gate_authority=True, known_storage=known, **values)
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        self.event_count += 1
        if edge == "END":
            self.stream.flush()

    def close(self):
        self.stream.close()
        self.persistent.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
