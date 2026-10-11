"""Free decoding of incremental parent events; no truth or cache loader.

Children inherit learned parent identity. Shared edges reference one stored
coordinate; positive gaps keep distinct coordinates. Invalid positives are
counted hypotheses, including zero children and all out-of-domain events.
"""
import math
import numpy as np
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.legacy_incremental_effective_nodes_422 import effective_nodes_numpy

VERSION='FULL_ORIGINAL_CANDIDATE_FREE_ORDERED_INCREMENTAL_EVENTS_v1'

def as_numpy(value):
 if hasattr(value,'detach'):return value.detach().cpu().numpy()
 return np.asarray(value)
def probability(logits):
 z=np.clip(np.asarray(logits,float),-80.,80.);return 1./(1.+np.exp(-z))
def segment_at(seq,left,right,closed=False):
 assert math.isfinite(left) and math.isfinite(right) and right>left
 if seq.get('curve_kind')=='arc':
  c=seq['curve'];radius=float(c['radius_mm']);angle=float(c['sequence_start_angle_rad'])+left/radius
  arc=CircularArc([*c['center_xy_mm'],c['height_mm']],radius,[0.,0.,1.],
   [math.cos(angle),math.sin(angle),0.],(right-left)/radius,closed)
  return sample_arc(arc)
 assert not closed,'A straight monotone segment cannot be a closed loop'
 start=np.asarray(seq['xyz'][0],float)+left*np.asarray(seq['direction'],float)
 end=np.asarray(seq['xyz'][0],float)+right*np.asarray(seq['direction'],float)
 length=float(np.linalg.norm(end-start));assert length>0 and math.isfinite(length)
 count=max(2,int(math.ceil(length))+1);assert count<=1000001
 return dict(kind='line',points=np.linspace(start,end,count).tolist(),length_mm=length,
  is_closed=False,sampling_audit=dict(max_step_mm=length/(count-1)))

def reverse_segment(segment):
 if segment['kind']=='line':return dict(segment,points=segment['points'][::-1])
 c=segment['arc_parameters'];arc=CircularArc.from_parameters(c)
 last=arc.point_at(np.asarray(arc.length_mm))
 return sample_arc(CircularArc(arc.center_mm,arc.radius_mm,arc.plane_normal,
  (last-arc.center_mm)/arc.radius_mm,-arc.sweep_rad,arc.is_closed))

def decode_candidate_events(sequences,output,threshold=.5,published_types=('vertical',)):
 arrays={k:as_numpy(output[k]) for k in ['presence_logits','count_logits','positions_mm','gap_logits','direction_logits','closed_logits']}
 b=len(sequences);presence=probability(arrays['presence_logits']);q=presence.shape[1]
 positions=arrays['positions_mm'];capacity=positions.shape[2]
 assert presence.shape==(b,q) and positions.shape==(b,q,capacity,2)
 assert arrays['count_logits'].shape==(b,q,capacity+1)
 assert arrays['gap_logits'].shape==(b,q,max(0,capacity-1))
 assert arrays['direction_logits'].shape==arrays['closed_logits'].shape==(b,q)
 predictions=[];invalid=[];parents=[]
 for si,seq in enumerate(sequences):
  if seq['type'] not in published_types:continue
  span=(len(seq['xyz'])-1)*float(seq['sample_step_mm'])
  for query in range(q):
   if not math.isfinite(float(presence[si,query])):
    count=1
   elif presence[si,query]<threshold:continue
   else:count=int(arrays['count_logits'][si,query].argmax())
   base=dict(sequence_index=si,parent_query_index=query,declared_children=max(1,count),
    type=seq['type'],kind='arc' if seq.get('curve_kind')=='arc' else 'line',
    presence_probability=float(presence[si,query]) if math.isfinite(float(presence[si,query])) else None)
   try:
    assert math.isfinite(base['presence_probability']) and 0<count<=capacity
    relevant=[arrays[k][si,query] for k in ['count_logits','positions_mm','gap_logits','direction_logits','closed_logits']]
    assert all(np.isfinite(v).all() for v in relevant)
    p=positions[si,query,:count].astype(float).copy()
    assert np.all(p>=0.) and np.all(p<=span),'Predicted event outside complete candidate domain'
    gaps=arrays['gap_logits'][si,query,:count-1]>=0.
    p=effective_nodes_numpy(p,gaps)
    for child in range(1,count):
     if gaps[child-1]:assert p[child,0]>p[child-1,1],'Positive gap is collapsed or overlapping'
    assert np.all(p[:,1]>p[:,0]),'Child event has zero or negative length'
    assert np.all(p[1:,0]>=p[:-1,1]),'Ordered child path overlaps or turns back'
    closed=bool(arrays['closed_logits'][si,query]>=0.)
    segments=[segment_at(seq,float(a),float(z),closed and count==1) for a,z in p]
    boundaries=[]
    for child,s in enumerate(segments):
     if child and not gaps[child-1]:
      left_ref=segments[child-1]['boundary_refs'][1];s['points'][0]=boundaries[left_ref]
     else:left_ref=len(boundaries);boundaries.append(s['points'][0])
     if closed and child==count-1:
      first=boundaries[segments[0]['boundary_refs'][0]] if child else boundaries[left_ref]
      assert np.linalg.norm(np.asarray(s['points'][-1])-np.asarray(first))<=1e-6,'Closed path does not complete its curve'
      right_ref=segments[0]['boundary_refs'][0] if child else left_ref;s['points'][-1]=first
     else:right_ref=len(boundaries);boundaries.append(s['points'][-1])
     s['boundary_refs']=[left_ref,right_ref]
    negative_direction=bool(arrays['direction_logits'][si,query]<0.)
    if negative_direction:
     segments=[reverse_segment(s) for s in segments[::-1]]
     gaps=gaps[::-1];boundaries=[]
     for child,s in enumerate(segments):
      if child and not gaps[child-1]:left_ref=segments[child-1]['boundary_refs'][1];s['points'][0]=boundaries[left_ref]
      else:left_ref=len(boundaries);boundaries.append(s['points'][0])
      if closed and child==count-1:
       right_ref=segments[0]['boundary_refs'][0] if child else left_ref;s['points'][-1]=boundaries[right_ref]
      else:right_ref=len(boundaries);boundaries.append(s['points'][-1])
      s['boundary_refs']=[left_ref,right_ref]
    for ci,s in enumerate(segments):s.update(segment_index=ci,segment_count=count,weld_type=seq['type'])
    pred=dict(id='event_%05d_%03d'%(si,query),type=seq['type'],segments=segments,
     confidence=base['presence_probability'],is_closed=closed,boundaries_mm=boundaries,
     gaps=['POSITIVE_GAP' if value else 'SHARED' for value in gaps],
     original_candidate_sequence_index=si,learned_parent_query_index=query)
    predictions.append(pred);parents.append(dict(**base,boundary_refs=[s['boundary_refs'] for s in segments],
     gaps=pred['gaps'],chart_direction=-1 if negative_direction else 1,is_closed=closed))
   except (AssertionError,ValueError,KeyError,OverflowError,TypeError) as error:
    invalid.append(dict(**base,reason='INVALID_POSITIVE_INCREMENTAL_EVENT_PATH: '+str(error)))
 return dict(version=VERSION,predictions=predictions,invalid_positive_ledger=invalid,parent_relations=parents,
  total_declared_prediction_children=sum(len(p['segments']) for p in predictions)+sum(p['declared_children'] for p in invalid),
  original_GT_or_STEP_reads=0,reference_assisted_orientation_or_child_count_selection=False,
  segmentation_is_learned_not_fixed_one_or_three=True,full_contract_pass=False)
