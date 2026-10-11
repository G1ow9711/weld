"""Bounded wait for asynchronous Windows CUDA startup page reclamation."""
import time
import torch
from weld_pipeline.sjg_cuda_startup_pages_1040 import startup_pages
from weld_pipeline.sjg_local_runtime_904 import resources,check_resources,reclaim_own_idle_pages


def start_cuda():
    try:
        return startup_pages()
    except RuntimeError as error:
        # Only resource pressure *after* CUDA initialization can be awaited.
        # No resource threshold is changed and no other process is touched.
        if not str(error).startswith('Resource guard: insufficient physical RAM or remaining commit') or not torch.cuda.is_initialized():
            raise
        observations=[];message=str(error)
        for attempt in range(15):
            print('WAIT_CUDA_STARTUP_MEMORY',attempt+1,'/15',resources(),flush=True)
            time.sleep(2)
            reclaim_own_idle_pages()
            try:
                ready=check_resources()
                return dict(initial_guard_failure=message,recovery_observations=observations,after=ready,
                    unchanged_resource_thresholds=True,optimizer_updates=0,other_processes_touched=False)
            except RuntimeError as retry:
                if not str(retry).startswith('Resource guard: insufficient physical RAM or remaining commit'):raise
                observations.append(resources())
        raise RuntimeError('Original CUDA resource guard still fails after bounded 30-second wait: '+str(resources())) from error
