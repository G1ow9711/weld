"""Offline positive interval witnesses on frozen input carriers, never predictions.

Each witness passes the original D3 child geometry test. The table provides
positive annotated occupancy, not complete local counts or unlabeled negatives.
"""
import argparse,time,collections
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 started=time.perf_counter();entry=next(c for c in read(BASE/'additional_batch01_protocol.json')['cases'] if c['key']==key);labels=BASE/'labels'/key;assert digest(labels/'receipt.json')==entry['labels_receipt_sha256'];targetpath=labels/'targets.json';truth=native_targets(read(targetpath))
 source=BASE/'frozen_features/legacy_runtime'/('source_'+key);profilepath=source/'run185'/('source_'+key)/'profiles.json';profiles=read(profilepath);coveragepath=BASE/'exhaustive_carrier_supervision_v1/coverage'/('source_'+key)/'raw_carrier_coverage.json';coverage=read(coveragepath);assert coverage['profiles_sha256']==digest(profilepath) and coverage['target_sha256']==digest(targetpath)
 binding=next(r for r in read(source/'run285/all_geometry_frozen_before_labels.json')['cases'] if r['source']==key);geometry=source/'run285'/binding['geometry_file'];assert digest(geometry)==binding['geometry_sha256']
 with np.load(geometry) as z:offsets=z['offsets'];spans=z['spans']
 with np.load(source/'run288'/('source_'+key+'_alias_map.npz')) as z:representative=z['fragment_representative'];canonical=z['canonical_fragments']
 records=[];rejected=[];missing=[]
 for row in coverage['rows']:
  pi,ci=row['parent'],row['child'];p=truth['parents'][pi];c=p['children'][ci];reference=reference_segment(c);full_closed=bool(p['is_closed'] and len(p['children'])==1);count=0
  for near in row['nearest']:
   if near['max_distance_mm']>3.:continue
   pid=near['profile'];curve=CurveCarrier(profiles[pid]['representation']);q,_=curve.project(np.asarray(reference['points']))
   if full_closed and curve.kind=='arc' and curve.arc.is_closed:lo,hi=0.,curve.length
   else:
    lo,hi=sorted((float(q[0]),float(q[-1])))
    if not hi>lo:rejected.append(dict(parent=pi,child=ci,profile=pid,reason='EMPTY_OR_WRAPPED_PROJECTED_INTERVAL'));continue
    if curve.kind=='arc' and np.any((q<lo-1e-8)|(q>hi+1e-8)):
     rejected.append(dict(parent=pi,child=ci,profile=pid,reason='ARC_WRAP_REQUIRES_EXPLICIT_WITNESS'));continue
   rep=curve.segment(lo,hi)
   segment=line_segment(*rep['endpoints_mm']) if curve.kind=='line' else sample_arc(CircularArc.from_parameters(rep['arc_parameters']))
   segment['weld_type']=c['weld_type'];error=None;orientation=None
   for reverse in (False,True):
    ss=dict(segment,**reverse_segment(segment)) if reverse else segment
    e=child_error(ss,c,reference,full_closed,False)
    if e is not None:error=e;orientation=reverse;break
   if error is None:rejected.append(dict(parent=pi,child=ci,profile=pid,reason='ORIGINAL_D3_CHILD_TEST_REJECTED'));continue
   a,b=offsets[pid:pid+2];local=spans[a:b];lengths=np.maximum(0,np.minimum(local[:,1],hi)-np.maximum(local[:,0],lo));take=np.flatnonzero(lengths>0);fragments=[]
   for j in take:
    fragment=int(a+j);canonical_id=int(representative[fragment]);location=int(np.searchsorted(canonical,canonical_id));assert int(canonical[location])==canonical_id
    left,right=local[j,:2];fragments.append(dict(fragment=fragment,canonical_fragment=canonical_id,feature_rows_both_unresolved_sides=[2*location,2*location+1],positive_overlap_mm=[float(max(left,lo)),float(min(right,hi))],positive_ratios=[float((max(left,lo)-left)/(right-left)),float((min(right,hi)-left)/(right-left))],local_total_count_known=False,physical_side_label_known=False))
   assert np.isclose(sum(f['positive_overlap_mm'][1]-f['positive_overlap_mm'][0] for f in fragments),hi-lo)
   records.append(dict(parent=pi,child=ci,kind=curve.kind,weld_type=c['weld_type'],profile=pid,interval_mm=[lo,hi],direction_reversed=orientation,original_D3_error=error,fragments=fragments));count+=1
  if not count:missing.append(dict(parent=pi,child=ci,kind=c['original_kind'],weld_type=c['weld_type']))
 counts=collections.Counter((r['parent'],r['child']) for r in records)
 for r in records:r['within_child_alternative_weight']=1/counts[r['parent'],r['child']]
 dest=BASE/'exhaustive_carrier_supervision_v1/positive_intervals'/key;dest.mkdir(parents=True,exist_ok=False);write(dest/'witnesses.json',dict(positive_witnesses=records,missing_children=missing,rejected_witnesses=rejected))
 report=dict(status='POSITIVE_ONLY_TRAIN_GEOMETRY_WITNESSES',key=key,children=len(coverage['rows']),children_with_witnesses=len(counts),witnesses=len(records),fragment_events=sum(len(r['fragments']) for r in records),original_D3_checked_every_witness=True,target_sha256=digest(targetpath),profile_sha256=digest(profilepath),geometry_sha256=digest(geometry),coverage_sha256=digest(coveragepath),witnesses_sha256=digest(dest/'witnesses.json'),count_supervision_complete=False,unknown_unlabeled_not_negative=True,not_runtime_prediction=True,GT_only_offline_TRAIN=True,DEV_reads=0,fit_calls=0,source_sha256=digest(__file__),seconds=time.perf_counter()-started);write(dest/'receipt.json',report);print(report,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
 with threadpool_limits(limits=2):run(a.key)
