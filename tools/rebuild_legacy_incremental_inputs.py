"""Original input-only whole-carrier and local support features for run446."""
import argparse,time
from pathlib import Path
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

def run(key,stage):
    runtime.CASE=runtime.BASE/'legacy_runtime'/('source_'+key)
    assert read(runtime.CASE/'observed_features_reconstruction_receipt.json')['status']=='RECONSTRUCTED_STAGE_COMPLETE'
    if stage=='admission':
        source=runtime.ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776'
        for name,origin in [('run420','run420'),('run384','run414/admission')]:
            dest=runtime.CASE/name;dest.mkdir(exist_ok=True)
            files=['presence_model.pkl','training_receipt.json'] if name=='run420' else ['whole_joint_model.pkl','training_receipt.json']
            for f in files:
                target=source/origin/f;link=dest/f
                if not link.exists():link.symlink_to(target)
                assert digest(link)==digest(target)
        m=runtime.module('sjg_corrected_presence_train_1260');m.infer(runtime.CASE/'run420',key)
        return
    whole='run414/whole' if key in ('25','36') else 'run380'
    local='run414/local' if key in ('25','36') else 'run383'
    name='sjg_whole_joint_features_1228' if stage=='whole' else 'sjg_local_cross_section_1233'
    m=runtime.module(name);dest=runtime.CASE/(whole if stage=='whole' else local);dest.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if stage=='whole':m.prepare(key,dest)
    else:m.prepare(key,dest,whole_run=whole)
    receipt=read(dest/('source_'+key)/'receipt.json');assert receipt['GT_reads']==receipt['STEP_reads']==0
    write(runtime.CASE/(stage+'_incremental_receipt.json'),dict(status='ORIGINAL_INCREMENTAL_INPUT_STAGE_COMPLETE',source_key=key,stage=stage,seconds=time.perf_counter()-start,source_sha256=digest(__file__),original_module_sha256=digest(m.__file__),output_receipt_sha256=digest(dest/('source_'+key)/'receipt.json'),GT_reads=0,STEP_reads=0,historical_parity_proven=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);ap.add_argument('--stage',required=True,choices=['whole','local','admission']);args=ap.parse_args()
    with threadpool_limits(limits=2):run(args.key,args.stage)
