"""Five-case interval pilot; original269 targets, fixed final epoch, no DEV fit."""
import argparse
import shutil
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest
from weld_pipeline.sjg_native_whole_presence_1259 import TRAIN,paths


def routes(key):
    return dict(context_run=paths(key)[0],sequence_run='run421/sequence56',occupancy_run='run421/occupancy')


def initialize(dest,stage):
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
    for name in ('sjg_five_case_interval_pilot_1264.py','sjg_native_interval_set_training_1253.py',
                 'sjg_structured_interval_set_1251.py','sjg_joint_pair_model_1248.py','sjg_interval_set_targets_1249.py',
                 'sjg_lossless_training_buffers_1268.py'):
        src=ROOT/'weld_pipeline'/name;bindings[name]=digest(src);shutil.copy2(src,snapshot/name)
    write(dest/'registered_protocol.json',dict(stage=stage,TRAIN=list(TRAIN),DEV=['64'],original_G=269,
        module_sha256=bindings,input_run='run421/endpoints',input_routes={k:routes(k) for k in TRAIN},
        native_run='run422',set_targets_run='run423',initial_model_run='run408',
        epochs=12,batch=64,learning_rate=.0003,optimizer='fresh Adam',seed=1031,
        same_input_rule_all_cases=True,final_epoch_only=True,keep_best379=True,buffer_inputs=True,
        material_role_rules_not_fully_implemented=True,external_consultation_paused=True,full_contract_pass=False))


def execute(stage,key=None,resume=False):
    from weld_pipeline import sjg_native_interval_set_training_1253 as targets
    from weld_pipeline import sjg_structured_interval_set_1251 as trainer
    dest=OUT/('run423' if stage=='prepare' else 'run424')
    if stage in ('prepare','train') and not resume:
        audit=read(OUT/'run422/independent_audit_summary.json')
        assert audit['status']=='PASS' and audit['original_G']==269
        if stage=='train':assert read(OUT/'run423/receipt.json')['G']==269
        initialize(dest,stage)
    protocol=read(dest/'registered_protocol.json')
    for name,sha in protocol['module_sha256'].items():assert digest(ROOT/'weld_pipeline'/name)==sha
    if stage=='prepare':
        targets.alias_controls(dest)
        targets.prepare(dest,train=TRAIN,native_run='run422',input_run='run421/endpoints',expected_g=269)
    elif stage=='train':
        targets.alias_controls(dest)
        trainer.train(dest,resume,supervision_run='run423',input_run='run421/endpoints',
            input_routes={k:routes(k) for k in TRAIN},expected_g=269,initial_run='run408',buffer_inputs=True)
    else:
        assert key in (*TRAIN,'64')
        trainer.base.infer(dest,key,input_run='run421/endpoints',**routes(key))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','train','infer'],required=True)
    p.add_argument('--source');p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:execute(a.stage,a.source,a.resume)
    except BaseException:
        d=OUT/('run423' if a.stage=='prepare' else 'run424')
        if d.exists():write(d/f'failure_{a.stage}_{a.source or "all"}.json',dict(error=traceback.format_exc()))
        raise
