"""Offline TRAIN diagnostic of unknown candidate geometry versus finite labels."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 reports=[]
 for key in ['10','25','36']:
  f=BASE/('source_'+key);selection=read(BASE/'set_competition_v1'/('source_'+key)/'receipt.json')['selected_parent_indices'];parents=read(f/'parents.json')['parents'];supervision=read(f/'d3_supervision.json')['rows'];targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json';target=native_targets(read(targetpath));targets=[]
  for pi,p in enumerate(target['parents']):
   for ci,c in enumerate(p['children']):
    ref=reference_segment(c)
    if ref['kind']=='line':targets.append((pi,ci,c['weld_type'],np.asarray(ref['points'])[[0,-1]]))
  rows=[]
  for i in selection:
   if supervision[i]['parent_loss_valid']:continue
   children=[]
   for s in parents[i]['segments']:
    if s['kind']!='line':children.append(dict(kind=s['kind'],diagnostic='unexamined'));continue
    ends=np.asarray(s['points'])[[0,-1]];choices=[]
    for pi,ci,wt,e in targets:
     if wt!=s['weld_type']:continue
     v=e[1]-e[0];length=np.linalg.norm(v);v/=length;q=(ends-e[0])@v;closest=e[0]+np.clip(q,0,length)[:,None]*v;distance=np.linalg.norm(ends-closest,axis=1).max();choices.append((float(distance),pi,ci,float(q.min()/length),float(q.max()/length)))
    best=min(choices);children.append(dict(kind='line',finite_GT_support_max_mm=best[0],target_parent=best[1],target_child=best[2],projected_fraction=[best[3],best[4]],contained_in_annotated_line_tube=best[0]<=3.))
   rows.append(dict(candidate_parent=i,children=children,all_children_within_annotated_finite_line_tubes=all(c.get('contained_in_annotated_line_tube',False) for c in children)))
  report=dict(source=key,rows=rows,unknown_selected=len(rows),all_children_contained=sum(r['all_children_within_annotated_finite_line_tubes'] for r in rows),target_sha256=digest(targetpath),GT_diagnostic_only=True,no_new_negative_labels_created=True,not_a_material_identity_proof=True)
  write(BASE/'set_competition_v1'/('source_'+key)/'unknown_support_diagnostic.json',report);reports.append({k:report[k] for k in ['source','unknown_selected','all_children_contained']});print(reports[-1],flush=True)
 write(BASE/'set_competition_v1/unknown_support_summary.json',reports)
if __name__=='__main__':main()
