"""Fixed five-case interval experiment after native occupancy repair."""
import argparse
import shutil
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest

TRAIN=('21','BM79A_native_local','10','25','36')


def routes(key):
    return dict(context_run='run414/whole' if key in ('25','36') else 'run380',
        sequence_run='run421/sequence56',occupancy_run='run429')


def execute(stage,key=None,resume=False):
    if stage=='prepare':
        from weld_pipeline.sjg_endpoint_occupancy_overlay_1272 import prepare as overlay
        from weld_pipeline.sjg_refreshed_pair_initialization_1273 import prepare as initialize
        overlay();initialize();return
    from weld_pipeline import sjg_structured_interval_set_1251 as trainer
    dest=OUT/'run432'
    if stage=='train' and not resume:
        assert read(OUT/'run431/prefit_binding_audit.json')['status']=='PASS'
        dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
        for name in ('sjg_repaired_interval_pilot_1274.py','sjg_structured_interval_set_1251.py',
            'sjg_joint_pair_model_1248.py','sjg_endpoint_role_train_1246.py','sjg_endpoint_occupancy_overlay_1272.py',
            'sjg_refreshed_pair_initialization_1273.py','sjg_supervision_dependency_guard_1269.py',
            'sjg_lossless_training_buffers_1268.py'):
            path=ROOT/'weld_pipeline'/name;shutil.copy2(path,snapshot/name);bindings[name]=digest(path)
        write(dest/'registered_protocol.json',dict(module_sha256=bindings,TRAIN=list(TRAIN),DEV=['64'],G=269,
            input_run='run430',supervision_run='run431',input_routes={k:routes(k) for k in TRAIN},
            initialization_file='run431/pair_initialization.pt',fresh_current_TRAIN_normalization=True,
            epochs=12,batch=64,learning_rate=.0003,seed=1031,optimizer='fresh Adam',
            all_pairs_every_epoch=True,final_epoch_only=True,buffer_inputs=True,keep_best379=True,
            geometry_maps_and_full_native_supervision_unchanged=True,external_consultation_paused=True,full_contract_pass=False))
    protocol=read(dest/'registered_protocol.json')
    for name,sha in protocol['module_sha256'].items():assert digest(ROOT/'weld_pipeline'/name)==sha
    if stage=='train':
        trainer.train(dest,resume,supervision_run='run431',input_run='run430',input_routes={k:routes(k) for k in TRAIN},
            expected_g=269,initial_run=None,buffer_inputs=True,initialization_file='run431/pair_initialization.pt')
    else:
        assert key in (*TRAIN,'64');trainer.base.infer(dest,key,input_run='run430',**routes(key))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','train','infer'],required=True)
    p.add_argument('--source');p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:execute(a.stage,a.source,a.resume)
    except BaseException:
        write(ROOT/'.codex-local'/f'repaired_interval_{a.stage}_failure.json',dict(error=traceback.format_exc()));raise
