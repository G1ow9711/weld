"""Full original-parent scoring route for the repaired occupancy experiment."""
import argparse
import shutil
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_repaired_interval_pilot_1274 import TRAIN,routes as pair_routes
from weld_pipeline import sjg_additive_parent_precision_1254 as raw
from weld_pipeline import sjg_child_parent_supervision_1255 as supervision
from weld_pipeline import sjg_child_parent_model_1256 as model
from weld_pipeline.sjg_five_case_child_pilot_1258 import audit as audit_supervision

KEYS=(*TRAIN,'64')


def routes():
    return {key:dict(baseline_run='run413' if key in ('25','36') else 'run379',proposal_run='run432',feature_run='run433',
        pair_inputs=dict(input_run='run430',**pair_routes(key)),
        **(dict(known_background_file=f'run422/background_{key}.json') if key in TRAIN else {})) for key in KEYS}


def initialize(dest,stage):
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
    for name in ('sjg_repaired_child_parent_pilot_1275.py','sjg_additive_parent_precision_1254.py',
        'sjg_child_parent_supervision_1255.py','sjg_child_parent_model_1256.py','sjg_five_case_child_pilot_1258.py',
        'sjg_endpoint_role_train_1246.py','sjg_endpoint_occupancy_overlay_1272.py'):
        path=ROOT/'weld_pipeline'/name;bindings[name]=digest(path);shutil.copy2(path,snapshot/name)
    write(dest/'registered_protocol.json',dict(stage=stage,module_sha256=bindings,TRAIN=list(TRAIN),DEV=['64'],G=269,
        input_routes=routes(),fixed_threshold=.5,trees_each=64,keep_best379_full_parents=True,
        original_parent_order_gaps_closure_preserved=True,material_role_rules_not_fully_implemented=True,
        external_consultation_paused=True,full_contract_pass=False))


def execute(stage,key=None):
    if stage=='prepare':
        check_resources()
        for source in KEYS:
            r=read(OUT/f'run432/proposal_{source}/prediction_receipt.json')
            assert r['GT_reads']==r['STEP_reads']==0
            assert digest(OUT/f'run432/proposal_{source}/decoded.json')==r['decoded_sha256']
        dest=OUT/'run433';initialize(dest,'prepare_parent_features');raw.prepare(dest,keys=KEYS,routes=routes())
        labels=OUT/'run434';initialize(labels,'prepare_child_features')
        write(labels/'input_routes.json',dict(routes=routes(),TRAIN=list(TRAIN),DEV=['64'],G=269))
        supervision.prepare(labels,keys=KEYS)
    elif stage=='targets':supervision.targets(OUT/'run434',train=TRAIN,original_g=269)
    elif stage=='audit':audit_supervision(label_run='run434',train=TRAIN,routebook=routes())
    elif stage=='train':
        assert read(OUT/'run434/prefit_audit.json')['status']=='PASS'
        dest=OUT/'run435';initialize(dest,'train');model.train(dest,OUT/'run434')
    else:
        assert key in KEYS;model.infer(OUT/'run435',key)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','targets','audit','train','infer'],required=True);p.add_argument('--source');a=p.parse_args()
    try:execute(a.stage,a.source)
    except BaseException:
        write(ROOT/'.codex-local'/f'repaired_final_{a.stage}_failure.json',dict(error=traceback.format_exc()));raise
