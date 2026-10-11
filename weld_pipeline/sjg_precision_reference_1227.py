"""Frozen376 DEV64 comparison using exactly run379's upstream output, no fit."""
import argparse
import os
import shutil
import time

from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest
from weld_pipeline.sjg_owned_watchdog_1101 import supervise


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run', required=True); args=ap.parse_args()
    dest=OUT/args.run; dest.mkdir(exist_ok=False); start=time.perf_counter()
    write(dest/'registered_protocol.json',dict(status='FROZEN376_SAME_INPUT_DEV64_REFERENCE',
        execution_budget_seconds=150,fit_calls=0,source_run='run376',input_run='run379',source='64',
        no_extra_training=True,GT_inference_reads=0,full_contract_pass=False))
    old=OUT/'run376'; new=OUT/'run379'; frozen=read(new/'selector_training.json')['frozen_files']
    for name, sha in frozen.items():
        assert digest(old/name)==digest(new/name)==sha
        os.link(old/name,dest/name)
    shutil.copy2(old/'selector_training.json',dest/'selector_training.json')
    folder=dest/'test_64';folder.mkdir()
    for name in ('unary.npy','decoded.json','compact_parents.json','parent_points.npy','parent_metadata.json'):
        os.link(new/'test_64'/name,folder/name)
    for stage,module in (('select','weld_pipeline.sjg_incompatibility_worker_1217'),('score','weld_pipeline.sjg_selector_trial_worker_1174')):
        remaining=150-(time.perf_counter()-start)
        assert remaining>0
        command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',module,
            '--run',args.run,'--stage',stage,'--source','64']
        if stage=='select':command+=['--limit-seconds',str(min(remaining,120))]
        log=dest/f'job_{stage}';log.mkdir()
        result=supervise(command,log,min(remaining,120))
        assert result['exit_code']==0 and not result['termination_reason'],result
        assert read(dest/f'stage_{stage}_64_complete.json')['status']=='STAGE_COMPLETE'
    reference=read(folder/'evaluation.json')['report'];current=read(new/'test_64/evaluation.json')['report']
    write(dest/'receipt.json',dict(status='SAME_INPUT_DEV64_COMPARISON_COMPLETE',baseline376=reference,
        modified379=current,upstream_decoded_sha256=digest(folder/'decoded.json'),
        normalized_input_source_sha256=digest(folder/'unary.npy'),fit_calls=0,
        seconds=time.perf_counter()-start,full_contract_pass=False))
    print('FROZEN376_DEV64_REFERENCE',reference,flush=True)


if __name__=='__main__':main()
