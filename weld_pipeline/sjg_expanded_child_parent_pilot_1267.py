"""Five-case final child selection and original-parent assembly after424.

Freeze all six pure-input feature sets before accessing five TRAIN targets.
The 379 baseline remains intact. No DEV targets, new threshold search, or
fabricated plate roles are used to fit any head.
"""
import argparse
import shutil
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_native_whole_presence_1259 import TRAIN,paths
from weld_pipeline import sjg_additive_parent_precision_1254 as raw
from weld_pipeline import sjg_child_parent_supervision_1255 as supervision
from weld_pipeline import sjg_child_parent_model_1256 as model
from weld_pipeline.sjg_five_case_child_pilot_1258 import audit as audit_supervision

KEYS=(*TRAIN,'64')


def routes():
    return {key:dict(baseline_run='run413' if key in ('25','36') else 'run379',proposal_run='run424',feature_run='run425',
        pair_inputs=dict(input_run='run421/endpoints',context_run=paths(key)[0],sequence_run='run421/sequence56',occupancy_run='run421/occupancy'),
        **(dict(known_background_file=f'run422/background_{key}.json') if key in TRAIN else {})) for key in KEYS}


def initialize(dest,stage):
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir();bindings={}
    for name in ('sjg_expanded_child_parent_pilot_1267.py','sjg_additive_parent_precision_1254.py',
        'sjg_child_parent_supervision_1255.py','sjg_child_parent_model_1256.py','sjg_five_case_child_pilot_1258.py'):
        p=ROOT/'weld_pipeline'/name;bindings[name]=digest(p);shutil.copy2(p,snapshot/name)
    write(dest/'registered_protocol.json',dict(stage=stage,module_sha256=bindings,TRAIN=list(TRAIN),DEV=['64'],G=269,
        input_routes=routes(),fixed_threshold=.5,trees_each=64,keep_best379_full_parents=True,
        original_parent_order_gaps_closure_preserved=True,material_role_rules_not_fully_implemented=True,
        external_consultation_paused=True,full_contract_pass=False))


def prepare():
    check_resources()
    for key in KEYS:
        r=read(OUT/f'run424/proposal_{key}/prediction_receipt.json')
        assert r['GT_reads']==r['STEP_reads']==0
        assert digest(OUT/f'run424/proposal_{key}/decoded.json')==r['decoded_sha256']
    dest=OUT/'run425';initialize(dest,'prepare_parent_features')
    raw.prepare(dest,keys=KEYS,routes=routes())
    labels=OUT/'run426';initialize(labels,'prepare_child_features')
    write(labels/'input_routes.json',dict(routes=routes(),TRAIN=list(TRAIN),DEV=['64'],G=269))
    supervision.prepare(labels,keys=KEYS)


def execute(stage,key=None):
    if stage=='prepare':prepare()
    elif stage=='targets':supervision.targets(OUT/'run426',train=TRAIN,original_g=269)
    elif stage=='audit':audit_supervision(label_run='run426',train=TRAIN,routebook=routes())
    elif stage=='train':
        assert read(OUT/'run426/prefit_audit.json')['status']=='PASS'
        dest=OUT/'run427';initialize(dest,'train');model.train(dest,OUT/'run426')
    else:
        assert key in KEYS;model.infer(OUT/'run427',key)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','targets','audit','train','infer'],required=True);p.add_argument('--source');a=p.parse_args()
    try:execute(a.stage,a.source)
    except BaseException:
        write(ROOT/'.codex-local'/f'expanded_final_{a.stage}_failure.json',dict(error=traceback.format_exc()))
        raise
