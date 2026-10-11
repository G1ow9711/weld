"""Apply frozen TRAIN feature definitions to DEV64, without supervision access."""
import argparse
import gc
import shutil
import time
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');args=ap.parse_args()
    dest=OUT/'run335'
    if args.worker:
        from weld_pipeline.sjg_relative_geometry_batch_1136 import prepare as relative
        from weld_pipeline.sjg_observed_face_features_1138 import prepare as observed
        started=time.perf_counter()
        base=OUT/'run334';base.mkdir(exist_ok=False)
        a=relative('64',base);gc.collect()
        write(base/'receipt.json',dict(status='PURE_DEV_RELATIVE142_READY',cases=[a],GT_reads=0,fit_calls=0))
        b=observed('64',dest,prior_run='run334')
        write(dest/'worker_receipt.json',dict(status='PURE_DEV_RELATIVE152_READY',cases=[b],GT_reads=0,STEP_reads=0,
            fit_calls=0,seconds=time.perf_counter()-started,full_contract_pass=False))
        return
    for name in ('run331','run332','run333'):
        e=read(OUT/name/'evaluation.json')
        assert e['report']['development_continuation_gate_pass']
    dest.mkdir(exist_ok=False);snap=dest/'source_snapshot';snap.mkdir()
    for name in ('sjg_relative_dev_prepare_1143.py','sjg_relative_geometry_batch_1136.py','sjg_observed_face_features_1138.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,snap)
    write(dest/'registered_protocol.json',dict(status='FROZEN_MODEL_DEV_INPUT_ONLY',source='64',model_run='run330',
        model_sha256=digest(OUT/'run330/model.pkl'),GT_reads=0,STEP_reads=0,fit_calls=0,wall_seconds=180,utc=utc()))
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_relative_dev_prepare_1143','--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='DEV_INPUT_PREPARATION_INCOMPLETE',watchdog=result))
        raise RuntimeError('DEV preparation did not finish')
    receipt=read(dest/'worker_receipt.json');receipt['external_wall_seconds']=result['wall_seconds']
    write(dest/'receipt.json',receipt);print('DEV_RELATIVE_INPUT_READY',receipt['external_wall_seconds'],flush=True)


if __name__=='__main__':main()
