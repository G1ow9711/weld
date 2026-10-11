"""Lightweight local runtime IO and original Windows resource limits."""
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def _json_native(value):
    import numpy as np
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=_json_native),encoding='utf-8')
def utc():return datetime.now(timezone.utc).isoformat()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def resources():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_uint32),('PageFaultCount',ctypes.c_uint32)]+[(n,ctypes.c_size_t) for n in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    class System(ctypes.Structure):
        _fields_=[('dwLength',ctypes.c_uint32),('dwMemoryLoad',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in
            ('ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
    api=ctypes.WinDLL('psapi',use_last_error=True);api.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32]
    v=Counters();v.cb=ctypes.sizeof(v);s=System();s.dwLength=ctypes.sizeof(s)
    if not api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(v),v.cb):raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(s)):raise ctypes.WinError(ctypes.get_last_error())
    return dict(available_physical_bytes=int(s.ullAvailPhys),remaining_commit_bytes=int(s.ullAvailPageFile),
                private_commit_bytes=int(v.PagefileUsage),working_set_bytes=int(v.WorkingSetSize))

def check_resources():
    m=resources()
    if m['available_physical_bytes']<1342177280 or m['remaining_commit_bytes']<2684354560:
        raise RuntimeError('Resource guard: insufficient physical RAM or remaining commit '+str(m))
    if m['private_commit_bytes']>12884901888:raise RuntimeError('Resource guard: worker commit exceeds 12 GiB '+str(m))
    if shutil.disk_usage(ROOT).free<5368709120:raise RuntimeError('Resource guard: less than 5 GiB disk free')
    return m

def reclaim_own_idle_pages():
    """Trim only this worker's pageable working set; limits remain unchanged."""
    import gc
    gc.collect();before=resources()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
    api=ctypes.WinDLL('psapi',use_last_error=True);api.EmptyWorkingSet.argtypes=[ctypes.c_void_p]
    api.EmptyWorkingSet.restype=ctypes.c_int
    if not api.EmptyWorkingSet(kernel.GetCurrentProcess()):raise ctypes.WinError(ctypes.get_last_error())
    return dict(before=before,after=resources(),scope='calling_worker_only_no_other_process_touched')
