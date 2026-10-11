"""Supervise only a newly created Windows Job Object and its process tree.

Reference: https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
The worker starts suspended, joins the private job, then resumes. No process-name
kills, reused PID termination, or association of pre-existing applications.
"""
import ctypes as C
from ctypes import wintypes as W
import json
import shutil
import subprocess
import time
from weld_pipeline.sjg_local_runtime_904 import ROOT, resources, utc
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write


def atomic_json(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    atomic_binary_write(path, lambda stream: stream.write(data))


class Startup(C.Structure):
    _fields_ = [('cb', W.DWORD), ('reserved', W.LPWSTR), ('desktop', W.LPWSTR), ('title', W.LPWSTR)] + [
        (n, W.DWORD) for n in ('x','y','xsize','ysize','xcount','ycount','fill','flags')] + [
        ('show', W.WORD), ('reserved2_size', W.WORD), ('reserved2', C.c_void_p),
        ('stdin', W.HANDLE), ('stdout', W.HANDLE), ('stderr', W.HANDLE)]


class ProcessInfo(C.Structure):
    _fields_ = [('process', W.HANDLE), ('thread', W.HANDLE), ('pid', W.DWORD), ('tid', W.DWORD)]


class BasicLimit(C.Structure):
    _fields_ = [('process_time', C.c_int64), ('job_time', C.c_int64), ('flags', W.DWORD),
                ('min_ws', C.c_size_t), ('max_ws', C.c_size_t), ('active', W.DWORD),
                ('affinity', C.c_size_t), ('priority', W.DWORD), ('scheduling', W.DWORD)]


class ExtendedLimit(C.Structure):
    _fields_ = [('basic', BasicLimit), ('io', C.c_uint64 * 6), ('process_memory', C.c_size_t),
                ('job_memory', C.c_size_t), ('peak_process', C.c_size_t), ('peak_job', C.c_size_t)]


class ProcessIDs(C.Structure):
    _fields_ = [('assigned', W.DWORD), ('count', W.DWORD), ('ids', C.c_size_t * 128)]


class Memory(C.Structure):
    _fields_ = [('cb', W.DWORD), ('faults', W.DWORD)] + [(n, C.c_size_t) for n in
        ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile','private')]


def supervise(command, directory, wall_seconds):
    k = C.WinDLL('kernel32', use_last_error=True)
    p = C.WinDLL('psapi', use_last_error=True)
    declarations = {
        'CreateJobObjectW': ([C.c_void_p, W.LPCWSTR], W.HANDLE),
        'SetInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
        'QueryInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p], W.BOOL),
        'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL),
        'IsProcessInJob': ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
        'TerminateJobObject': ([W.HANDLE, W.UINT], W.BOOL),
        'TerminateProcess': ([W.HANDLE, W.UINT], W.BOOL),
        'CloseHandle': ([W.HANDLE], W.BOOL),
        'GetStdHandle': ([W.DWORD], W.HANDLE),
        'CreateProcessW': ([W.LPCWSTR, W.LPWSTR, C.c_void_p, C.c_void_p, W.BOOL, W.DWORD,
                            C.c_void_p, W.LPCWSTR, C.POINTER(Startup), C.POINTER(ProcessInfo)], W.BOOL),
        'ResumeThread': ([W.HANDLE], W.DWORD),
        'GetProcessTimes': ([W.HANDLE] + [C.POINTER(C.c_uint64)] * 4, W.BOOL),
        'WaitForSingleObject': ([W.HANDLE, W.DWORD], W.DWORD),
        'GetExitCodeProcess': ([W.HANDLE, C.POINTER(W.DWORD)], W.BOOL),
        'OpenProcess': ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
    }
    for name, (args, result) in declarations.items():
        getattr(k, name).argtypes = args
        getattr(k, name).restype = result
    p.GetProcessMemoryInfo.argtypes = [W.HANDLE, C.c_void_p, W.DWORD]
    p.GetProcessMemoryInfo.restype = W.BOOL

    def checked(ok):
        if not ok:
            raise C.WinError(C.get_last_error())

    def creation(handle):
        fields = [C.c_uint64() for _ in range(4)]
        checked(k.GetProcessTimes(handle, *[C.byref(x) for x in fields]))
        return fields[0].value

    job = k.CreateJobObjectW(None, None)
    checked(job)
    pi = ProcessInfo()
    assigned = False
    started = None
    record = dict(command=command, wall_limit_seconds=wall_seconds,
                  supervision_before_worker_execution=True, no_preexisting_process_assigned=True)
    try:
        limit = ExtendedLimit()
        limit.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        checked(k.SetInformationJobObject(job, 9, C.byref(limit), C.sizeof(limit)))
        si = Startup()
        si.cb = C.sizeof(si)
        si.flags = 0x100  # STARTF_USESTDHANDLES, preserve desktop launcher output.
        si.stdin, si.stdout, si.stderr = (k.GetStdHandle(v) for v in (-10, -11, -12))
        started = time.perf_counter()
        record['started_utc'] = utc()
        checked(k.CreateProcessW(str(command[0]), C.create_unicode_buffer(subprocess.list2cmdline(command)),
                                 None, None, True, 0x4, None, str(ROOT), C.byref(si), C.byref(pi)))
        identity = creation(pi.process)
        record.update(root_pid=int(pi.pid), root_creation_filetime=identity)
        checked(k.AssignProcessToJobObject(job, pi.process))
        assigned = True
        atomic_json(directory / 'watchdog_started.json', record)
        if k.ResumeThread(pi.thread) == 0xffffffff:
            raise C.WinError(C.get_last_error())
        k.CloseHandle(pi.thread)
        pi.thread = None
        last_print = 0.
        seen = {}
        reason = None
        while True:
            elapsed = time.perf_counter() - started
            ids = ProcessIDs()
            checked(k.QueryInformationJobObject(job, 3, C.byref(ids), C.sizeof(ids), None))
            if not ids.count:
                break
            if elapsed >= wall_seconds:
                reason = 'EXTERNAL_WALL_DEADLINE'
            if reason is None:
                m = resources()
                if m['available_physical_bytes'] < 1342177280 or m['remaining_commit_bytes'] < 2684354560:
                    reason = 'ORIGINAL_SYSTEM_RESOURCE_GUARD'
                if shutil.disk_usage(ROOT).free < 5368709120:
                    reason = 'ORIGINAL_DISK_RESOURCE_GUARD'
                for pid in list(ids.ids)[:ids.count]:
                    handle = k.OpenProcess(0x410, False, pid)
                    if not handle:
                        if C.get_last_error() == 87:  # A child exited between enumeration and open.
                            continue
                        raise C.WinError(C.get_last_error())
                    try:
                        belongs = W.BOOL()
                        checked(k.IsProcessInJob(handle, job, C.byref(belongs)))
                        if not belongs.value:
                            continue
                        born = creation(handle)
                        memory = Memory()
                        memory.cb = C.sizeof(memory)
                        measured = p.GetProcessMemoryInfo(handle, C.byref(memory), memory.cb)
                        if not measured and k.WaitForSingleObject(handle, 0) == 0:
                            continue
                        checked(measured)
                        seen[str(pid)] = dict(creation_filetime=born, private_commit_bytes=int(memory.private))
                        if memory.private > 12884901888:
                            reason = 'ORIGINAL_WORKER_COMMIT_GUARD'
                    finally:
                        k.CloseHandle(handle)
                record['last_resources'] = m
            if reason:
                assert creation(pi.process) == identity
                record.update(termination_reason=reason, termination_requested_utc=utc(),
                              termination_requested_seconds=time.perf_counter()-started,
                              owned_processes_at_termination=[int(v) for v in list(ids.ids)[:ids.count]])
                checked(k.TerminateJobObject(job, 124))
                atomic_json(directory / 'watchdog_termination_requested.json', record)
                print('WATCHDOG_TERMINATED_OWN_JOB', reason, flush=True)
                # Keep the original handles until exit is observable. Never reopen by PID to kill.
                for _ in range(40):
                    remaining = ProcessIDs()
                    checked(k.QueryInformationJobObject(job, 3, C.byref(remaining), C.sizeof(remaining), None))
                    if not remaining.count:
                        break
                    time.sleep(.25)
                record['remaining_job_processes_after_wait'] = int(remaining.count)
                break
            if elapsed - last_print >= 10:
                print('RECOVERY_WALL_SECONDS', round(elapsed, 2), '/', wall_seconds, flush=True)
                last_print = elapsed
            time.sleep(.25)
        exit_code = W.DWORD()
        checked(k.GetExitCodeProcess(pi.process, C.byref(exit_code)))
        record.update(status='OWNED_JOB_EXIT_OBSERVED', exit_code=int(exit_code.value),
                      exit_observed_utc=utc(), wall_seconds=time.perf_counter()-started,
                      observed_processes=seen, termination_reason=reason)
        atomic_json(directory / 'watchdog_receipt.json', record)
        return record
    except BaseException:
        if pi.process:
            # Both handles were returned for our own newly created, initially suspended process.
            if assigned:
                k.TerminateJobObject(job, 125)
            else:
                k.TerminateProcess(pi.process, 125)
        raise
    finally:
        if pi.thread:
            k.CloseHandle(pi.thread)
        if pi.process:
            k.CloseHandle(pi.process)
        k.CloseHandle(job)
