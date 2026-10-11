"""Fixed five-case interval experiment after native occupancy repair."""
import argparse
import shutil
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest

TRAIN=('21','BM79A_native_local','10','25','36')


def routes(key):
    return dict(context_run='run414/whole' if key in ('25','36') else 'run380',
        sequence_run='run421/sequence56',occupancy_run='run437')


def execute(stage,key=None,resume=False):
    if stage=='prepare':
        from weld_pipeline.sjg_density_endpoint_overlay_1279 import prepare as overlay
        from weld_pipeline.sjg_density_pair_initialization_1280 import prepare as initialize
        overlay();initialize();return
    from weld_pipeline import sjg_density_structured_interval_1281 as trainer
    dest=OUT/'run440'
    if stage=='train' and not resume:
        assert read(OUT/'run439/prefit_binding_audit.json')['status']=='PASS'
        dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
        for name in ('sjg_density_interval_pilot_1282.py','sjg_density_structured_interval_1281.py',
            'sjg_joint_pair_model_1248.py','sjg_endpoint_role_train_1246.py','sjg_density_endpoint_overlay_1279.py',
            'sjg_density_pair_initialization_1280.py','sjg_supervision_dependency_guard_1269.py',
            'sjg_lossless_training_buffers_1268.py','sjg_density_feature_runtime_1277.py',
            'sjg_endpoint_occupancy_overlay_1272.py'):
            path=ROOT/'weld_pipeline'/name;shutil.copy2(path,snapshot/name);bindings[name]=digest(path)
        write(dest/'registered_protocol.json',dict(module_sha256=bindings,TRAIN=list(TRAIN),DEV=['64'],G=269,
            input_run='run438',supervision_run='run439',input_routes={k:routes(k) for k in TRAIN},
            initialization_file='run439/pair_initialization.pt',fresh_current_TRAIN_normalization=True,
            density_augmentation_run='run436',density_manifest_sha256=digest(OUT/'run436/receipt.json'),
            augmentation='per carrier Bernoulli 0.5; only 45 endpoint dimensional-neighborhood columns replaced',
            original_geometry_targets_weights_and_all_pairs_retained=True,
            epochs=12,batch=64,learning_rate=.0003,seed=1031,optimizer='fresh Adam',
            all_pairs_every_epoch=True,final_epoch_only=True,buffer_inputs=True,keep_best379=True,
            geometry_maps_and_full_native_supervision_unchanged=True,external_consultation_paused=True,full_contract_pass=False))
    protocol=read(dest/'registered_protocol.json')
    assert digest(OUT/'run436/receipt.json')==protocol['density_manifest_sha256']
    for name,sha in protocol['module_sha256'].items():assert digest(ROOT/'weld_pipeline'/name)==sha
    if stage=='train':
        trainer.train(dest,resume,supervision_run='run439',input_run='run438',input_routes={k:routes(k) for k in TRAIN},
            expected_g=269,initial_run=None,buffer_inputs=True,initialization_file='run439/pair_initialization.pt')
    else:
        assert key in (*TRAIN,'64');trainer.base.infer(dest,key,input_run='run438',**routes(key))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','train','infer'],required=True)
    p.add_argument('--source');p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:execute(a.stage,a.source,a.resume)
    except BaseException:
        write(ROOT/'.codex-local'/f'repaired_interval_{a.stage}_failure.json',dict(error=traceback.format_exc()));raise
