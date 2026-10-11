"""Conditional positive-query child geometry probes, not final extraction scores."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';MODEL=BASE/'positive_query_head_v1/model_query_local_distance_seven'
def main():
 mr=read(MODEL/'receipt.json');reports=[]
 for key,sha in mr['bindings'].items():
  folder=BASE/'positive_query_head_v1'/key;assert digest(folder/'receipt.json')==sha;records=read(folder/'records.json');labels=(ROOT/'server_rebuild/labels_v1') if key in ('21','BM79A_native_local','10','25','36') else BASE/'labels';targets=native_targets(read(labels/key/'targets.json'));refs={(pi,ci):reference_segment(c) for pi,p in enumerate(targets['parents']) for ci,c in enumerate(p['children'])};runtime=(ROOT/'server_rebuild/best_baseline_restore/legacy_runtime') if key in ('21','BM79A_native_local','10','25','36') else BASE/'frozen_features/legacy_runtime';profiles=read(runtime/('source_'+key)/'run185'/('source_'+key)/'profiles.json');curves={};known={tuple(o) for r in records for o in r['owners']}
  for split in ('training','heldout'):
   path=MODEL/(key+'_'+split+'.npz');assert digest(path)==mr['files'][path.name]
   with np.load(path) as z:ratios=z['interval_ratios'];roles=z['weld_type']
   hits={};valid=0;invalid=0;matching_queries=0
   for qi,(row,bounds,role) in enumerate(zip(records,ratios,roles)):
    pid=row['profile']
    if pid not in curves:curves[pid]=CurveCarrier(profiles[pid]['representation'])
    curve=curves[pid];lo,hi=bounds*curve.length
    if not np.isfinite(bounds).all() or not 0<=lo<hi<=curve.length:invalid+=1;continue
    valid+=1;ends=curve.at(np.array([lo,hi]));weld_type='horizontal' if role==0 else 'vertical';eligible=[]
    for owner in row['owners']:
     pi,ci=owner;c=targets['parents'][pi]['children'][ci];ref=refs[pi,ci];closed=bool(targets['parents'][pi]['is_closed'] and len(targets['parents'][pi]['children'])==1)
     if c['weld_type']!=weld_type:continue
     e=np.asarray(ref['points'])[[0,-1]];distance=min(np.linalg.norm(ends-e,axis=1).max(),np.linalg.norm(ends[::-1]-e,axis=1).max())
     if distance<=3. or closed:eligible.append((pi,ci,c,ref,closed))
    if not eligible:continue
    rep=curve.segment(float(lo),float(hi));segment=line_segment(*rep['endpoints_mm']) if curve.kind=='line' else sample_arc(CircularArc.from_parameters(rep['arc_parameters']));segment['weld_type']=weld_type;matched=False
    for pi,ci,c,ref,closed in eligible:
     for reverse in (False,True):
      s=dict(segment,**reverse_segment(segment)) if reverse else segment;error=child_error(s,c,ref,closed,False)
      if error is not None:hits.setdefault((pi,ci),dict(query=qi,profile=pid,error=error));matched=True;break
    matching_queries+=int(matched)
   report=dict(key=key,split=split,known_positive_queries=len(records),represented_GT_children=len(known),children_with_geometry_passing_proposal=len(hits),matching_query_rows=matching_queries,valid_queries=valid,invalid_queries=invalid,missing_represented_children=sorted(known-set(hits)),hits=[dict(parent=pi,child=ci,**v) for (pi,ci),v in sorted(hits.items())],prediction_sha256=digest(path),scope='Only positive fixed-input-grid query subset is scored; no false-positive denominator or parent assembly. Not full extraction accuracy/acceptance. Training split tests fit capacity; heldout shares frozen upstream.',fit_calls=0,DEV_reads=0);write(MODEL/(key+'_'+split+'_conditional_geometry.json'),report);reports.append({k:v for k,v in report.items() if k not in ('hits','missing_represented_children')});print(reports[-1],flush=True)
 write(MODEL/'conditional_geometry_summary.json',dict(cases=reports,status='CONDITIONAL_PROPOSAL_DIAGNOSTIC_ONLY',baseline_unchanged=True))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
