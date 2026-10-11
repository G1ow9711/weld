"""Bind input-frozen whole-parent features to exact TRAIN set utility tables.

Uses the original parent pool including its retained baseline parents, curves,
child order and gaps. Unmatched candidates have a REFERENCE-policy cost; they
are not converted into physical no-weld labels. Runtime never reads the table.
"""
from pathlib import Path
import time
import numpy as np
from weld_pipeline.server_offline_parent_set_reference import compile_reference_table,evaluate_reference_subset
from weld_pipeline.server_array_cached_d3_evaluation import score_native_parents_cached
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild';DEST=BASE/'expansion_v1/whole_parent_reference_set_v1'

def main():
 control=BASE/'expansion_v1/parent_set_reference_controls_v1/receipt.json'
 assert read(control)['module_sha256']==digest('weld_pipeline/server_offline_parent_set_reference.py')
 cases=[(key,BASE/'unified_parent_scoring_v1'/('source_'+key),BASE/'labels_v1'/key) for key in ['21','BM79A_native_local','10','25','36']]
 key='new_ac2008a03607';cases.append((key,BASE/'expansion_v1/unified_parent_scoring_v1'/('source_'+key),BASE/'expansion_v1/labels'/key))
 DEST.mkdir(exist_ok=False)
 write(DEST/'protocol.json',dict(keys=[k for k,_,_ in cases],objective='Exact existing D3 TP - FP for selected whole-parent sets',unknown_semantics='Unmatched output cost belongs to annotated reference policy; not physical absence supervision',full_target_scope_held_fixed=True,direction_semantics=False,original_candidate_coordinates_unchanged=True,original_whole_parents_preserved=True,keep_invalid_output_ledger=True,full_chain_independence=False,read_protected_DEV=False,fit_calls=0,modules={n:digest('weld_pipeline/'+n+'.py') for n in ['server_offline_parent_set_reference','contract_geometry_d3_409']}))
 # Bind all input artifacts BEFORE reference tables are compiled.
 bindings=[]
 for key,folder,labels in cases:
  r=read(folder/'receipt.json');assert r['GT_reads']==0
  for name,sha in r['files'].items():assert digest(folder/name)==sha
  input_names=['parents.json','features.npy','parent_children.json','child_records.json','default_configuration.json']
  bindings.append(dict(key=key,folder=str(folder.relative_to(ROOT)),files={name:digest(folder/name) for name in input_names},input_receipt_sha256=digest(folder/'receipt.json')))
 write(DEST/'input_bindings.json',dict(cases=bindings,GT_reads=0,source_sha256=digest(__file__)))
 reports=[]
 for key,folder,labels in cases:
  begin=time.perf_counter();out=DEST/key;out.mkdir();label_receipt=read(labels/'receipt.json');assert label_receipt['role'] in ['TRAIN','EXPANSION_TRAIN']
  targetpath=labels/'targets.json';assert digest(targetpath)==label_receipt['targets_sha256'];target=read(targetpath)
  pool=read(folder/'parents.json');print('COMPILE',key,len(pool['parents']),len(target['parents']),flush=True)
  table,score=compile_reference_table(pool,target,[False]*len(target['parents']))
  np.savez_compressed(out/'reference_table.npz',**table)
  default=read(folder/'default_configuration.json')['parent_indices'];rng=np.random.default_rng(1031)
  samples=[[],default,rng.choice(len(pool['parents']),min(8,len(pool['parents'])),replace=False).tolist()]
  controls=[]
  for ids in samples:
   part=dict(pool,parents=[pool['parents'][i] for i in ids],total_prediction_children=int(table['child_counts'][ids].sum())+int(table['invalid_children']))
   actual=score_native_parents_cached(part,target,[False]*len(target['parents']))['categories']['overall'];cached=evaluate_reference_subset(table,ids)
   assert all(actual[k]==cached[k] for k in ['TP','FP','predicted']);controls.append(dict(parent_indices=ids,score=cached))
  prior=read(folder/'d3_supervision.json');assert prior['target_sha256']==digest(targetpath)
  unknown=sum(not r['parent_loss_valid'] for r in prior['rows'])
  report=dict(key=key,parents=len(pool['parents']),reference_parents=len(target['parents']),G=target['original_children'],candidate_children=int(table['child_counts'].sum()),unknown_geometry_parents_under_previous_mask=unknown,full_pool_score=score['categories']['overall'],subset_controls=controls,table_sha256=digest(out/'reference_table.npz'),input_bindings_sha256=digest(DEST/'input_bindings.json'),target_sha256=digest(targetpath),scope='Offline exact subset supervision for fixed historical candidate pool; not new-grid integration or deployed selection',fit_calls=0,DEV_reads=0,seconds=time.perf_counter()-begin)
  write(out/'receipt.json',report);reports.append({k:v for k,v in report.items() if k not in ['subset_controls']});print('COMPILED',key,report['full_pool_score'],report['seconds'],flush=True)
 write(DEST/'receipt.json',dict(status='WHOLE_PARENT_REFERENCE_TABLES_READY',cases=reports,source_sha256=digest(__file__),fit_calls=0,DEV_reads=0))
if __name__=='__main__':main()
