"""D3 supervision for frozen expansion parents; unknown geometry is masked.

All child correspondences are preserved, including both traversal directions.
No support-tube or absence-of-label negative expansion is performed.
"""
import argparse,collections,time
from pathlib import Path
import numpy as np
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets,score_native_parents
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'server_rebuild/expansion_v1';BASE=EXP/'unified_parent_scoring_v1';LABELS=EXP/'labels'
def run(key):
 start=time.perf_counter();folder=BASE/('source_'+key);receipt=read(folder/'receipt.json');out=folder/'d3_supervision.json';assert not out.exists()
 for name,sha in receipt['files'].items():assert digest(folder/name)==sha
 targetpath=LABELS/key/'targets.json';target=read(targetpath);assert digest(targetpath)==read(targetpath.parent/'receipt.json')['targets_sha256']
 parents=read(folder/'parents.json')['parents'];ids=read(folder/'parent_children.json');segments=read(folder/'child_segments.json');truth=native_targets(target);children=[]
 for pi,p in enumerate(truth['parents']):
  for ci,c in enumerate(p['children']):children.append((pi,ci,c,reference_segment(c),bool(p['is_closed'] and len(p['children'])==1)))
 ends=np.array([np.asarray(r['points'])[[0,-1]] for pi,ci,c,r,cl in children]);closed=np.array([v[-1] for v in children]);owners=[]
 for s in segments:
  e=np.asarray(s['points'])[[0,-1]];err=np.minimum(np.linalg.norm(ends-e,axis=2).max(1),np.linalg.norm(ends-e[::-1],axis=2).max(1));eligible=np.flatnonzero((err<=3.)|closed);pair=[]
  for reverse in (False,True):
   ss=dict(s,**reverse_segment(s)) if reverse else s;matches=set()
   for j in eligible:
    pi,ci,c,r,cl=children[j]
    if child_error(ss,c,r,cl,False) is not None:matches.add((pi,ci))
   pair.append(matches)
  owners.append(pair)
 signatures=collections.defaultdict(list)
 for i,p in enumerate(target['parents']):signatures[p['original_child_count'],bool(p['is_closed']),tuple(p['gaps'])].append(i)
 rows=[];control_classes=set();controls=[]
 for hi,(p,indices) in enumerate(zip(parents,ids)):
  n=p['child_count'];assert n==len(indices)==len(p['segments']);gaps=tuple(p['gaps']);scores={}
  for reverse in (False,True):
   order=indices[::-1] if reverse else indices;gg=gaps[::-1] if reverse else gaps
   for shift in (range(n) if p['is_closed'] and n>1 else [0]):
    view=order[shift:]+order[:shift];vg=gg
    if shift:ring=gg+('SHARED',);vg=(ring[shift:]+ring[:shift])[:-1]
    for ti in signatures[n,bool(p['is_closed']),vg]:
     tp=sum((ti,ci) in owners[ix][int(reverse)] for ci,ix in enumerate(view));scores[ti]=max(scores.get(ti,0),tp)
  best=max(scores.values(),default=0);known=sum(bool(owners[ix][0] or owners[ix][1]) for ix in indices);mask=known==n
  row=dict(hypothesis=hi,children=n,D3_TP=best,D3_FP=n-best,known_geometry_children=known,parent_loss_valid=mask,compatible_best_parents=sorted(ti for ti,tp in scores.items() if tp==best and tp),annotated_parent_scope=sorted({ti for ix in indices for ti,ci in owners[ix][0]|owners[ix][1]}),positive_correspondences=[dict(parent=ti,TP=tp) for ti,tp in sorted(scores.items()) if tp]);rows.append(row)
  cls=(n,best,mask,bool(p['is_closed']))
  if cls not in control_classes and len(controls)<32:
   control_classes.add(cls);score=score_native_parents(dict(parents=[p],invalid_predictions=[],total_prediction_children=n),target,[False]*len(target['parents']))['categories']['overall'];assert score['TP']==best and score['FP']==n-best;controls.append(dict(hypothesis=hi,TP=best,FP=n-best))
 write(out,dict(source=key,rows=rows,oriented_child_owners=[[sorted(a),sorted(b)] for a,b in owners],parents_sha256=digest(folder/'parents.json'),features_sha256=digest(folder/'features.npy'),target_sha256=digest(targetpath),input_receipt_sha256=digest(folder/'receipt.json'),controls=controls,partial_TP_not_collapsed_to_binary=True,unknown_geometry_parent_loss_masked=True,negative_from_absence=False,support_tube_expansion=False,fit_calls=0,DEV_reads=0,source_sha256=digest(__file__),seconds=time.perf_counter()-start))
 print(dict(source=key,parents=len(rows),known_rows=sum(r['parent_loss_valid'] for r in rows),reference_controls=len(controls)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();cohort=next(c for c in read(EXP/'coverage_cohort_protocol.json')['cases'] if c['key']==a.key);assert digest(LABELS/a.key/'receipt.json')==cohort['labels_receipt_sha256'];run(a.key)
