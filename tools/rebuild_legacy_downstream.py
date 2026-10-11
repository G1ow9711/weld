"""Isolated reconstruction of original graph and 128/142/152 feature stages.

Run-number directories here are routing conventions for new artifacts, never
claims that historical tensors were recovered or numerically reproduced.
"""
import argparse,importlib,sys,time,traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
ADAPTER=install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(__import__('os').environ.get('WELD_RESTORE_ROOT',str(ROOT/'server_rebuild/best_baseline_restore'))).resolve()
BASE.relative_to(ROOT)

def module(name):
    m=importlib.import_module('weld_pipeline.'+name)
    for k,v in list(sys.modules.items()):
        if k.startswith('weld_pipeline.') and v is not None and hasattr(v,'OUT'):v.OUT=CASE
    return m

def graphs(key):
    dest=CASE/'run285';dest.mkdir()
    m=module('sjg_joint_graph_capacity_1105');m.TRAIN_KEYS=(key,);rows=m.freeze_graphs(dest)
    profiles=read(CASE/f'run185/source_{key}/profiles.json')
    with np.load(dest/rows[0]['geometry_file']) as z:offsets=z['offsets'];spans=z['spans']
    rep,prep,merges=module('sjg_joint_alias_normalization_1108').normalize(profiles,offsets,spans)
    dest=CASE/'run288';dest.mkdir();p=dest/f'source_{key}_alias_map.npz'
    np.savez(p,fragment_representative=rep,profile_representative=prep,canonical_fragments=np.unique(rep))
    write(dest/'all_alias_maps_frozen_before_labels.json',dict(cases=[dict(source=key,map_sha256=digest(p))],labels_read=0))
    write(dest/f'source_{key}_proven_aliases.json',dict(merges=merges))
    request=module('sjg_request_competition_1125');Graph=module('sjg_joint_segment_graph_1104').JointSegmentGraph
    dest=CASE/'run312';dest.mkdir();p=dest/f'source_{key}_input_requests.json'
    write(p,request.freeze_requests(key,Graph(profiles,offsets,spans)))
    write(dest/'all_input_requests_frozen_before_labels.json',dict(cases=[dict(source=key,sha256=digest(p))],GT_reads=0))

def bind_models():
    source=ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run379'
    dest=CASE/'run379';dest.mkdir()
    names=['model.pkl','upstream_verified.json','selector_training.json','precision_model.pkl','selector.npz','frozen_duplicate.npz','frozen_residual.npz','base_selector.npz','selector_normalization.npz']
    for name in names:(dest/name).symlink_to(source/name)
    write(dest/'reconstruction_binding.json',dict(source=str(source),files={n:digest(dest/n) for n in names},weights_frozen=True,historical_parity_proven=False))

def screen(key):
    models,binding=module('sjg_recall_set_worker_1162').load_upstream(CASE/'run379')
    runtime=module('sjg_relative_ordinal_runtime_1150');case=runtime.load_case(key);value=runtime.predict_instances(case,models)
    value.update(model_sha256=binding['model_sha256'],feature_receipt_sha256=case['feature_receipt_sha256'],GT_reads=0,fit_calls=0,source=key)
    dest=CASE/'run379';path=dest/f'screen_{key}.json';write(path,value)
    write(dest/f'screen_{key}_receipt.json',dict(screen_sha256=digest(path),nodes=len(value['nodes']),invalid=len(value['invalid']),model_sha256=value['model_sha256'],feature_receipt_sha256=value['feature_receipt_sha256']))

def forest_stage(key,stage):
    dest=CASE/'run379'
    if stage=='select':return module('sjg_precision_residual_1225').select(dest,key,120.)
    m=module('sjg_sparse_recovery_worker_1170')
    if stage=='edge_graph':
        recovery=module('sjg_recall_recovery_worker_1165')
        models,_=module('sjg_recall_set_worker_1162').load_upstream(dest)
        recovery.component=lambda name:models[name]
        m.build_graph(dest,key,True)
    elif stage=='solve':m.solve(dest,key,True)
    elif stage=='decode':m.decode(dest,key,True)

def run(key,stage):
    if stage=='bind_models':bind_models()
    elif stage=='screen':screen(key)
    elif stage in ('edge_graph','solve','decode','select'):forest_stage(key,stage)
    elif stage=='graphs':graphs(key)
    elif stage=='object_features':
        # Original feature routing for extension TRAIN cases.
        module('sjg_joint_object_features_1106').prepare(key,'run308' if key in ('25','36') else 'run287')
    elif stage=='relative_features':
        dest=CASE/'run327';dest.mkdir();module('sjg_relative_geometry_batch_1136').prepare(key,dest)
    elif stage=='observed_features':
        dest=CASE/('run344' if key in ('25','36') else 'run329');dest.mkdir();module('sjg_observed_face_features_1138').prepare(key,dest)
    else:raise ValueError(stage)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);ap.add_argument('--stage',required=True,choices=['bind_models','graphs','object_features','relative_features','observed_features','screen','edge_graph','solve','decode','select']);args=ap.parse_args()
    CASE=BASE/'legacy_runtime'/('source_'+args.key);assert CASE.exists()
    marker=CASE/(args.stage+'_reconstruction_receipt.json');assert not marker.exists()
    started=time.perf_counter()
    try:
        with threadpool_limits(limits=2):run(args.key,args.stage)
        write(marker,dict(status='RECONSTRUCTED_STAGE_COMPLETE',stage=args.stage,source_key=args.key,seconds=time.perf_counter()-started,source_sha256=digest(__file__),adapter=ADAPTER,GT_reads=0,STEP_reads=0,historical_parity_proven=False))
    except BaseException:
        write(CASE/(args.stage+'_failure.json'),dict(error=traceback.format_exc()));raise
