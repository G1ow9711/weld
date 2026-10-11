"""Optional process-memory telemetry; unavailable values remain None."""
import ctypes
import os


def process_memory_mb():
    if os.name!='nt':return None
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in
            ['peak_working','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
    value=Counters();value.cb=ctypes.sizeof(value)
    get_process=ctypes.windll.kernel32.GetCurrentProcess;get_process.restype=ctypes.c_void_p
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(get_process()),ctypes.byref(value),value.cb):return None
    return {'working_set_mb':value.working/2**20,'peak_working_set_mb':value.peak_working/2**20}
