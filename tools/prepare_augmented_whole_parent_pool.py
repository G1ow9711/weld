"""Actual original-event outputs + preserved native parents, with shared features.

All geometry and features freeze before TRAIN reference utility is computed.
No filtered reference-selected candidate or grid oracle is used as input.
"""
from pathlib import Path
import time
import numpy as np
from weld_pipeline.server_whole_parent_features import encode_parents
from weld_pipeline.server_bound_parent_pool import load_parent_pool
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.server_offline_parent_set_reference import compile_reference_table,evaluate_reference_subset
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'

def main():
 control=read(BASE/'whole_parent_feature_parity_v1/receipt.json')
 if control['module_sha256']!=digest('weld_pipeline/server_whole_parent_features.py'):
  control=read(BASE/'whole_parent_projection_limits_v1/receipt.json')
 assert control['status']=='PASS' and control['module_sha256']==digest('weld_pipeline/server_whole_parent_features.py')
 keys=read(BASE/'reference_query_task_v1/protocol.json')['training_keys']
 assert 'new_cfc732e5f068' not in keys and 'new_ceeae49d6777' not in keys
 dest=BASE/'augmented_whole_parent_pool_v1';dest.mkdir(exist_ok=False)
 write(dest/'protocol.json',dict(keys=keys,feature_schema='SIDE_SYMMETRIC_REAL_CARRIER_CHILD466_PARENT1458_V1',scope='Six already registered expansion TRAIN families; current candidate parents, not independent acceptance',new_line_source='reference_detector_balanced_endpoint_full_output_v1',native_parent_policy='Preserve previously frozen whole-parent pool when available; otherwise complete unscreened native run379 parents',targets_read_after_input_freeze=True,reference_utility_not_physical_absence=True,DEV_reads=0,fit_calls=0,modules={n:digest('weld_pipeline/'+n+'.py') for n in ['server_whole_parent_features','server_bound_parent_pool','server_offline_parent_set_reference']}))
 reports=[]
 for key in keys:
  start=time.perf_counter();out=dest/key;out.mkdir();source=BASE/'frozen_features/legacy_runtime'/('source_'+key)
  prepared=BASE/'unified_parent_scoring_v1'/('source_'+key)
  native=prepared/'parents.json' if (prepared/'receipt.json').exists() else source/'run379'/('test_'+key)/'decoded.json'
  if native.parent==prepared:
   assert digest(native)==read(prepared/'receipt.json')['files']['parents.json']
  else:
   assert (native.parent/'solve_receipt.json').exists()
  lines=BASE/'reference_detector_balanced_endpoint_full_output_v1'/key;freeze=read(lines/'prediction_freeze.json');assert freeze['GT_reads']==0
  assert freeze['compact_decoder_sha256']==digest('weld_pipeline/server_compact_line_predictions.py')
  linepath=lines/'compact_predictions.npz';assert digest(linepath)==freeze['files'][linepath.name]
  manifest=dict(native_parents=dict(path=str(native.relative_to(ROOT)),sha256=digest(native)),line_predictions=dict(path=str(linepath.relative_to(ROOT)),sha256=digest(linepath)),line_freeze_sha256=digest(lines/'prediction_freeze.json'),GT_reads=0)
  write(out/'parent_pool_manifest.json',manifest)
  pool,native_count=load_parent_pool(ROOT,manifest);inp=FixedQueries(source,key)
  print('FEATURES',key,'native',native_count,'total',len(pool['parents']),flush=True)
  encoded=encode_parents(inp,pool['parents']);np.save(out/'features.npy',encoded['features']);np.save(out/'child_features.npy',encoded['child_features']);write(out/'parent_children.json',encoded['parent_children']);write(out/'child_metadata.json',encoded['child_metadata'])
  write(out/'input_freeze.json',dict(parent_pool_manifest_sha256=digest(out/'parent_pool_manifest.json'),input_bindings=inp.bindings,features_sha256=digest(out/'features.npy'),parent_children_sha256=digest(out/'parent_children.json'),child_metadata_sha256=digest(out/'child_metadata.json'),native_parents=native_count,new_line_parents=len(pool['parents'])-native_count,GT_reads=0,STEP_reads=0,fit_calls=0))
  inp.close();del encoded
  labels=BASE/'labels'/key;lr=read(labels/'receipt.json');assert lr['role']=='EXPANSION_TRAIN';targetpath=labels/'targets.json';assert digest(targetpath)==lr['targets_sha256'];target=read(targetpath)
  print('REFERENCE_TABLE',key,flush=True);table,score=compile_reference_table(pool,target,[False]*len(target['parents']));np.savez_compressed(out/'reference_table.npz',**table)
  report=dict(key=key,parents=len(pool['parents']),original_parent_count=native_count,new_line_parent_count=len(pool['parents'])-native_count,reference_children=target['original_children'],full_pool_score=score['categories']['overall'],native_only_score=evaluate_reference_subset(table,np.arange(native_count)),new_lines_only_score=evaluate_reference_subset(table,np.arange(native_count,len(pool['parents']))),input_freeze_sha256=digest(out/'input_freeze.json'),table_sha256=digest(out/'reference_table.npz'),target_sha256=digest(targetpath),fit_calls=0,DEV_reads=0,source_sha256=digest(__file__),seconds=time.perf_counter()-start)
  write(out/'receipt.json',report);reports.append(report);print('READY',key,report['full_pool_score'],flush=True)
 write(dest/'receipt.json',dict(status='SIX_TRAIN_AUGMENTED_WHOLE_PARENT_POOLS_READY',cases=reports,fit_calls=0,DEV_reads=0))
if __name__=='__main__':main()
