"""Give the registered visual optimization stage exclusive active GPU training.

Suspend only a specifically authenticated comparison process; preserve its
memory/checkpoint and restore it when the new stage exits, including failure.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.export_full_dataset_split_view import local_name
from weld_pipeline.sjg_local_runtime_904 import ROOT, read, write


def process(pid):
    folder = Path('/proc') / str(pid)
    if not folder.exists(): return None
    stat = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, state=stat[0], start_ticks=stat[19], pgid=int(stat[2]),
                argv=(folder / 'cmdline').read_bytes().replace(b'\0', b' ').decode())


def run(args):
    out = ROOT / 'server_rebuild/full_dataset_v2' / local_name(args.controller_name)
    out.mkdir(exist_ok=False)
    original = process(args.comparison_pid)
    if (original is None or original['start_ticks'] != args.comparison_start_ticks
            or original['pgid'] != args.comparison_pid
            or 'tools.train_reciprocal_full_recall_parent_dataset' not in original['argv']
            or 'full_interval_endpoint_parent_v5_round04' not in original['argv']
            or original['state'] in ('Z', 'X', 'T')):
        raise ValueError('Comparison process identity/state must be authenticated before suspension')
    command = [sys.executable, '-u', '-m', 'tools.train_supported_interval_full_dataset',
               '--experiment-name', local_name(args.experiment_name), '--device', 'cuda', '--steps-per-view', '8']
    receipt = dict(status='GIVING_GPU_PRIORITY_TO_VISUAL_INTERVAL_TRAINING', controller_pid=os.getpid(),
                   comparison_before=original, command=command, complete_rounds_added=0,
                   official_validation_or_test_automatically_started=False)
    write(out / 'receipt.json', receipt)
    os.killpg(args.comparison_pid, signal.SIGSTOP)
    try:
        for _ in range(100):
            if process(args.comparison_pid)['state'] == 'T': break
            time.sleep(.05)
        else: raise RuntimeError('Comparison did not stop; refuse simultaneous active GPU trainers')
        receipt['comparison_suspended'] = process(args.comparison_pid)
        with (out / 'training.log').open('ab') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt.update(child_pid=child.pid, child_start_ticks=process(child.pid)['start_ticks'])
            write(out / 'receipt.json', receipt)
            code = child.wait()
        receipt.update(exit_code=code, status='VISUAL_OPTIMIZATION_EXITED_COMPLETE_OUTPUT_STILL_REQUIRED')
        write(out / 'receipt.json', receipt)
        if code: raise RuntimeError(f'Visual optimization exited with code {code}')
    finally:
        current = process(args.comparison_pid)
        if current and current['start_ticks'] == args.comparison_start_ticks and current['state'] == 'T':
            os.killpg(args.comparison_pid, signal.SIGCONT)
            receipt['comparison_resumed_signal_sent'] = True
            write(out / 'receipt.json', receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--controller-name', required=True)
    p.add_argument('--experiment-name', required=True); p.add_argument('--comparison-pid', required=True, type=int)
    p.add_argument('--comparison-start-ticks', required=True); run(p.parse_args())
