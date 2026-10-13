"""Wait for one authenticated input-only pool, then fit/export its TRAIN control."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.export_full_dataset_split_view import local_name
from weld_pipeline.sjg_local_runtime_904 import ROOT, read, write


def run(args):
    base = ROOT / 'server_rebuild/full_dataset_v2'
    out = base / local_name(args.controller_name); out.mkdir(exist_ok=False)
    pool = base / local_name(args.pool_name) / 'pools' / args.job_key
    if Path(args.job_key).name != args.job_key: raise ValueError('Local registered input key required')
    status = dict(status='WAITING_FOR_AUTHENTICATED_COMPLETE_INPUT_POOL', controller_pid=os.getpid(),
                  pool_pid=args.pool_pid, pool_start_ticks=args.pool_start_ticks,
                  official_validation_or_test_started=False, full_optimization_rounds_added=0)
    write(out/'progress.json',status)
    while not (pool/'receipt.json').exists():
        if (pool/'failure.json').exists():
            write(out/'failure.json',dict(stage='INPUT_POOL',failure=read(pool/'failure.json')));return
        path = Path('/proc') / str(args.pool_pid) / 'stat'
        if not path.exists():
            write(out/'failure.json',dict(stage='INPUT_POOL',error='Authenticated input process exited without a complete pool'));return
        parts = path.read_text().rsplit(')',1)[1].split()
        if parts[19] != args.pool_start_ticks or parts[0] in ('Z','X'):
            write(out/'failure.json',dict(stage='INPUT_POOL',error='Original process identity is terminal or changed'));return
        time.sleep(10)
    command=[sys.executable,'-u','-m','tools.fit_supported_interval_parent_pool_control',
        '--pool-name',local_name(args.pool_name),'--experiment-name',local_name(args.fit_name),
        '--job-key',args.job_key,'--partial-training-control','--device','cpu','--steps','8']
    with (out/'fit.log').open('ab') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status.update(status='FITTING_COMPLETE_TRAIN_SET_AND_EXPORTING_CONTROL',child_pid=child.pid,command=command)
        write(out/'progress.json',status);code=child.wait()
    status.update(exit_code=code,status='TRAIN_OUTPUT_CONTROL_COMPLETED' if code==0 else 'TRAIN_OUTPUT_CONTROL_FAILED')
    write(out/'receipt.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller-name',required=True);p.add_argument('--pool-name',required=True)
    p.add_argument('--fit-name',required=True);p.add_argument('--job-key',required=True)
    p.add_argument('--pool-pid',required=True,type=int);p.add_argument('--pool-start-ticks',required=True);run(p.parse_args())
