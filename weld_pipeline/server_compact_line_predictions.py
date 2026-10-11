"""Lossless line prediction freeze with original 1mm samples retained for D3.

Compact arrays are intermediate predictions, not a customer CSV export.
"""
import math
import numpy as np
from .sjg_curve_queries_907 import CurveCarrier

def line_segment_array(start,end):
 start,end=np.asarray(start,np.float64),np.asarray(end,np.float64)
 length=float(np.linalg.norm(end-start))
 if not math.isfinite(length) or length<=0:raise ValueError('Degenerate predicted line')
 count=int(math.ceil(length))+1
 if count>1_000_001:raise ValueError('Predicted line exceeds registered export sample capacity')
 points=np.linspace(start,end,count,dtype=np.float64);points[0],points[-1]=start,end
 return dict(kind='line',points=points,endpoints_mm=[start.tolist(),end.tolist()],length_mm=length,is_closed=False,sampling_audit=dict(max_step_mm=length/(count-1)))

def build_compact_lines(profiles,profile_ids,interval_mm,roles,query_ids,event_ids):
 seen=set();kept=[];ends=[]
 for i,(pid,(lo,hi),role) in enumerate(zip(profile_ids,interval_mm,roles)):
  signature=(int(pid),float(lo),float(hi),int(role))
  if signature in seen:continue
  seen.add(signature);kept.append(i)
  if hi<=lo:ends.append(np.full((2,3),np.nan));continue
  curve=CurveCarrier(profiles[pid]['representation']);assert curve.kind=='line';ends.append(curve.at(np.array([lo,hi])))
 keep=np.asarray(kept,np.int64);compact=dict(selected_prediction_indices=keep,profile_ids=np.asarray(profile_ids)[keep],interval_mm=np.asarray(interval_mm)[keep],weld_type=np.asarray(roles)[keep],query_ids=np.asarray(query_ids)[keep],event_ids=np.asarray(event_ids)[keep],endpoints_mm=np.asarray(ends,np.float64).reshape(-1,2,3))
 return decode_compact_lines(compact),compact

def decode_compact_lines(compact):
 parents=[];invalid=[]
 for pid,(lo,hi),role,query,event_ids,ends in zip(compact['profile_ids'],compact['interval_mm'],compact['weld_type'],compact['query_ids'],compact['event_ids'],compact['endpoints_mm']):
  if hi<=lo:
   invalid.append(dict(counted_false_children=1,predicted_weld_types=[int(role)],predicted_kinds=[1],reason='EMPTY_EVENT_INTERVAL'));continue
  segment=line_segment_array(*ends);segment.update(weld_type='horizontal' if role==0 else 'vertical',predicted_side='UNKNOWN');parents.append(dict(child_count=1,is_closed=False,gaps=[],segments=[segment],query_id=int(query),input_profile=int(pid),chosen_event_ids=event_ids.tolist()))
 return dict(parents=parents,invalid_predictions=invalid,total_prediction_children=len(parents)+len(invalid),GT_reads=0,STEP_reads=0,parent_assembly_implemented=False)
