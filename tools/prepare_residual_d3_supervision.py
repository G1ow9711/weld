"""D3-count supervision for frozen hypotheses; unknown geometry stays masked."""
import argparse,collections
from pathlib import Path
import numpy as np
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/parent_hypotheses_v1'
def run(key):
 folder=BASE/('source_'+key);receipt=read(folder/'receipt.json')
 for name,sha in receipt['files'].items():assert digest(folder/name)==sha
 targetpath=ROOT/'server_rebuild/labels_v1'/key/'targets.json';truth=read(targetpath);assert digest(targetpath)==read(targetpath.parent/'receipt.json')['targets_sha256']
 originfolder=ROOT/'server_rebuild/parent_residual_v1'/('source_'+key);binding=read(originfolder/'child_ownership.json');assert binding['target_sha256']==digest(targetpath)
 atoms=read(folder/'atoms.json');owners=[set() for _ in range(2*len(atoms['parents']))]
 for row,r in enumerate(binding['rows']):
  for o in r['owners']:owners[atoms['row_states'][row]^int(o['reversed'])].add((o['parent'],o['child']))
 for alias_rows in atoms['aliases']:
  reference=None
  for row in alias_rows:
   state=atoms['row_states'][row];r=binding['rows'][row];local=[set(),set()]
   for o in r['owners']:local[(state%2)^int(o['reversed'])].add((o['parent'],o['child']))
   if reference is None:reference=local
   else:assert local==reference,'Exact alias changed target compatibility'
 signatures=collections.defaultdict(list)
 for ti,t in enumerate(truth['parents']):signatures[t['original_child_count'],t['is_closed'],tuple(t['gaps'])].append(ti)
 paths=read(folder/'paths.json');paths=paths['native']+paths['added'];hypotheses=read(folder/'hypotheses.json')['parents'];rows=[];control_classes=set();controls=[]
 for hi,(path,h) in enumerate(zip(paths,hypotheses)):
  states=path['states'];n=len(states);assert n==h['child_count'];gaps=tuple(h['gaps']);best=0;best_parents=set();scores={}
  for reverse in (False,True):
   order=tuple(s^1 for s in states[::-1]) if reverse else tuple(states);gg=gaps[::-1] if reverse else gaps
   for shift in (range(n) if h['is_closed'] and n>1 else [0]):
    view=order[shift:]+order[:shift];vg=gg
    if shift:ring=gg+('SHARED',);vg=(ring[shift:]+ring[:shift])[:-1]
    for ti in signatures[n,h['is_closed'],vg]:
     tp=sum((ti,ci) in owners[state] for ci,state in enumerate(view));scores[ti]=max(scores.get(ti,0),tp)
     if tp>best:best=tp;best_parents={ti}
     elif tp==best and tp:best_parents.add(ti)
  known=sum(bool(owners[s] or owners[s^1]) for s in states);mask=known==n
  owner_parents=sorted({ti for s in states for ti,ci in owners[s]|owners[s^1]})
  rows.append(dict(hypothesis=hi,children=n,D3_TP=best,D3_FP=n-best,known_geometry_children=known,parent_loss_valid=mask,compatible_best_parents=sorted(best_parents),annotated_parent_scope=owner_parents,positive_correspondences=[dict(parent=ti,TP=tp) for ti,tp in sorted(scores.items()) if tp]))
  control_class=(n,best,mask,h['is_closed'])
  if control_class not in control_classes and len(controls)<32:
   control_classes.add(control_class);decoded=dict(parents=[h],invalid_predictions=[],total_prediction_children=n);score=score_native_parents(decoded,truth,[False]*len(truth['parents']))
   assert score['categories']['overall']['TP']==best and score['categories']['overall']['FP']==n-best
   controls.append(dict(hypothesis=hi,children=n,TP=best,FP=n-best))
 report=dict(source=key,rows=rows,oriented_atom_owners=[sorted(x) for x in owners],hypotheses_sha256=digest(folder/'hypotheses.json'),target_sha256=digest(targetpath),ownership_sha256=digest(originfolder/'child_ownership.json'),controls=controls,partial_TP_not_collapsed_to_binary=True,unknown_geometry_parent_loss_masked=True,fit_calls=0,DEV_reads=0)
 write(folder/'d3_supervision.json',report);print(dict(source=key,hypotheses=len(rows),known_parent_loss_rows=sum(r['parent_loss_valid'] for r in rows),partial_TP_rows=sum(0<r['D3_TP']<r['children'] for r in rows),D3_reference_controls=len(controls)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);run(ap.parse_args().key)
