"""Keep completed TRAIN read cache from starving the ongoing full-data work.

Uses scoped read-only file advice, preserves the original memory guards and
records every advice operation. It does not evict prediction pools or mutate
data/model files. Stops only when every explicit completion receipt exists.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from tools.reclaim_completed_training_read_cache import BASE, ROOT, GIB, main as reclaim, memory_snapshot


def main(args):
    if (not args.experiment_name or Path(args.experiment_name).name != args.experiment_name
            or args.trigger_headroom_gib < 2.5 or args.interval_seconds < 5):
        raise ValueError('A local watcher name and explicit conservative resource policy are required')
    out = BASE/args.experiment_name
    out.mkdir(exist_ok=False)
    until = [Path(value).resolve() for value in args.until_receipt]
    for path in until:
        path.relative_to(ROOT)
    active_progress = list(dict.fromkeys([BASE/'preparation/progress.json',
        BASE/'full_parent_dataset_streamed_v1/training_progress.json',
        *[path.parent/'training_progress.json' for path in until]]))
    sources = [Path(__file__), ROOT/'tools/reclaim_completed_training_read_cache.py']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    protocol = dict(status='SCOPED_COMPLETED_TRAIN_READ_CACHE_WATCHER', sources=hashes,
        trigger_headroom_gib=args.trigger_headroom_gib, interval_seconds=args.interval_seconds,
        until_receipts=[str(p.relative_to(ROOT)) for p in until],
        excluded_jobs=args.exclude_job, resource_guards_unchanged=True,
        global_cache_configuration_unchanged=True, model_or_input_bytes_unchanged=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    counter = 0
    try:
        while not all(path.exists() for path in until):
            if hashes != {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
                raise RuntimeError('Watcher source changed; use a new immutable version')
            snapshot = memory_snapshot()
            if snapshot['conservative_headroom_bytes'] < args.trigger_headroom_gib*GIB:
                excluded = set(args.exclude_job)
                for path in active_progress:
                    if path.exists():
                        job = json.loads(path.read_text()).get('job_key')
                        if job:
                            excluded.add(job)
                counter += 1
                reclaim(argparse.Namespace(experiment_name=args.experiment_name+f'_advice_{counter:04d}',
                                           exclude_job=sorted(excluded)))
            state = dict(status='WATCHING_SCOPED_COMPLETED_TRAIN_CACHE', controller_pid=os.getpid(),
                advice_calls=counter, snapshot=memory_snapshot(), optimizer_updates=0,
                unix_time=time.time())
            stage=out/'progress.incomplete';stage.write_text(json.dumps(state,indent=2)+'\n')
            os.replace(stage,out/'progress.json')
            time.sleep(args.interval_seconds)
        (out/'receipt.json').write_text(json.dumps(dict(status='ALL_DECLARED_COMPLETION_RECEIPTS_EXIST',
            advice_calls=counter,optimizer_updates=0,unix_time=time.time()),indent=2)+'\n')
    except BaseException:
        (out/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),unix_time=time.time()),indent=2)+'\n')
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-name',required=True)
    parser.add_argument('--until-receipt',required=True,action='append')
    parser.add_argument('--exclude-job',action='append',default=[])
    parser.add_argument('--trigger-headroom-gib',type=float,default=10.)
    parser.add_argument('--interval-seconds',type=int,default=30)
    main(parser.parse_args())
