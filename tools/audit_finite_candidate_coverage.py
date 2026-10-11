"""Evaluation-only geometry coverage, never an extraction accuracy score."""
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from weld_pipeline.world_set_evaluation_285 import reference_segment,distance_to_primitive
ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('--version',required=True);a=ap.parse_args()
d=ROOT/('server_rebuild/candidates_'+a.version)/a.key
assert (d/'receipt.json').exists()
candidates=json.loads((d/'candidates.json').read_text());target=json.loads((ROOT/'server_rebuild/labels_v1'/a.key/'targets.json').read_text())
cache=[]
for c in candidates:
 rep=c['representation'];cache.append(reference_segment(dict(original_kind=rep['kind'],world_representation=rep)))
rows=[]
for pi,p in enumerate(target['parents']):
 for ci,c in enumerate(p['children']):
  ref=reference_segment(dict(c,world_representation=c['representation']));q=np.asarray(ref['points']);dist=np.full(len(q),np.inf);best=None
  for k,(raw,pred) in enumerate(zip(candidates,cache)):
   if raw['representation']['kind']!=c['original_kind']:continue
   forward=distance_to_primitive(q,pred);dist=np.minimum(dist,forward)
   endpoints=np.asarray(pred['points'])[[0,-1]]
   err=min(np.linalg.norm(q[[0,-1]]-endpoints,axis=1).max(),np.linalg.norm(q[[0,-1]]-endpoints[::-1],axis=1).max())
   if err>3:continue
   whole=max(float(forward.max()),float(distance_to_primitive(np.asarray(pred['points']),ref).max()))
   if whole<=3 and (best is None or whole<best['error_mm']):best=dict(candidate=k,error_mm=whole)
  maximum=float(dist.max());rows.append(dict(parent=pi,child=ci,kind=c['original_kind'],type=c['weld_type'],complete_candidate=best,union_support_max_mm=maximum if np.isfinite(maximum) else None,union_support_3mm=maximum<=3))
result=dict(source=a.key,G=target['original_children'],candidate_count=len(candidates),complete_geometry_candidates=sum(x['complete_candidate'] is not None for x in rows),union_support_children=sum(x['union_support_3mm'] for x in rows),rows=rows,model_accuracy_test=False,weld_type_side_material_and_parent_assembly_ignored=True,meaning='optimistic input geometry coverage only; no TP/FP/precision claims',fit_calls=0)
(d/'coverage_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
