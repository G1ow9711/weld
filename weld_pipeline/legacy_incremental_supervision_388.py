"""Offline TRAIN-only mapping of complete labels to fresh original factories.

No target-derived candidate is added. Every unmapped original child remains
in the original geometry/E16 denominator with an explicit representation miss.
"""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np

VERSION='FRESH_LEGACY_FULL_DOMAIN_INCREMENTAL_SUPERVISION_v1'
GEOMETRY_TOLERANCE_MM=.25

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def load_fresh_training_interfaces(directory,expected_manifest_sha256):
 directory=Path(directory);assert digest(directory/'manifest.json')==expected_manifest_sha256
 manifest=read(directory/'manifest.json')
 assert manifest['capture_files_are_training_outputs_never_runtime_prediction_inputs']
 assert manifest['GT_reads']==manifest['STEP_reads']==0
 arrays={}
 for name,row in manifest['files'].items():
  assert Path(name).name==name and digest(directory/name)==row['file_sha256']
  a=np.load(directory/name,mmap_mode='r',allow_pickle=False)
  assert list(a.shape)==row['shape'] and str(a.dtype)==row['dtype']
  arrays[Path(name).stem]=a
 sequences=[];previous=0
 for row in manifest['sequences']:
  start,stop=row['start'],row['stop'];assert start==previous and stop>start
  seq=dict(row['fields'])
  assert not set(seq)&{'xyz','features'}
  seq.update(xyz=np.asarray(arrays['xyz'][start:stop],dtype=row['original_xyz_dtype']),
   features=np.asarray(arrays['features'][start:stop],dtype=row['original_features_dtype']))
  seq['direction']=np.asarray(seq['direction'],float)
  assert seq['features'].shape==(stop-start,84) and len(seq['xyz'])==stop-start
  sequences.append(seq);previous=stop
 assert previous==manifest['total_rows']==len(arrays['features'])
 return dict(sequences=sequences,arrays=arrays,manifest=manifest)

def candidate_positions(seq,points):
 """Project to the unchanged candidate chart, without rotating the GT."""
 points=np.asarray(points,np.float64);assert points.ndim==2 and points.shape[1]==3
 if seq.get('curve_kind')=='arc':
  from weld_pipeline.curved_sequences import arc_positions
  return arc_positions(seq,points)
 origin=np.asarray(seq['xyz'][0],np.float64);direction=np.asarray(seq['direction'],np.float64)
 assert abs(np.linalg.norm(direction)-1.)<=1e-6
 positions=(points-origin)@direction
 represented=origin+positions[:,None]*direction
 return positions,np.linalg.norm(represented-points,axis=1)

def map_complete_targets(sequences,targets,geometry_tolerance_mm=GEOMETRY_TOLERANCE_MM):
 """Keep distinct parent identities, child order, signed order and positive gaps."""
 from weld_pipeline.world_set_evaluation_285 import reference_segment
 assert sum(len(p['children']) for p in targets['parents'])==targets['original_children']
 mapped=[];misses=[];by_sequence={};categories=Counter()
 for pi,parent in enumerate(targets['parents']):
  children=parent['children'];assert len(children)==parent['original_child_count']
  for c in children:categories[(c['weld_type'],c['original_kind'])]+=1
  kinds={c['original_kind'] for c in children};types={c['weld_type'] for c in children}
  choices=[]
  if len(kinds)==len(types)==1:
   kind=next(iter(kinds));typ=next(iter(types))
   points=[np.asarray(reference_segment(c)['points'],np.float64) for c in children]
   ends=np.concatenate([p[[0,-1]] for p in points]);all_points=np.concatenate(points)
   for si,seq in enumerate(sequences):
    if seq['type']!=typ or (seq.get('curve_kind')=='arc')!=(kind=='arc'):continue
    position,error=candidate_positions(seq,all_points)
    span=(len(seq['xyz'])-1)*float(seq['sample_step_mm'])
    if error.max()>geometry_tolerance_mm or position.min()<-1e-8 or position.max()>span+1e-8:continue
    endpoints,endpoint_errors=candidate_positions(seq,ends)
    if np.any(np.abs(endpoints.reshape(-1,2)[:,1]-endpoints.reshape(-1,2)[:,0])<=1e-10):continue
    choices.append((float(error.max()),si,np.clip(endpoints.reshape(-1,2),0.,span),span))
  if not choices:
   misses.append(dict(original_parent_index=pi,original_parent_id=parent['original_parent_id'],
    original_child_ids=[c['original_child_id'] for c in children],original_child_count=len(children),
    status='REPRESENTATION_MISS',reason='NO_FULL_DOMAIN_ORIGINAL_MIXED_CANDIDATE',
    all_children_retained_in_geometry_and_E16_denominators=True))
   continue
  residual,si,positions,span=min(choices,key=lambda r:(r[0],r[1]))
  deltas=positions[:,1]-positions[:,0];direction=1 if deltas[0]>0 else -1
  if np.any(deltas*direction<=0) or np.any(np.diff(positions[:,0])*direction<0):
   misses.append(dict(original_parent_index=pi,original_parent_id=parent['original_parent_id'],
    original_child_ids=[c['original_child_id'] for c in children],original_child_count=len(children),
    status='REPRESENTATION_MISS',reason='ORIGINAL_CHILD_ORDER_NOT_ONE_ORDERED_CANDIDATE_PATH',
    all_children_retained_in_geometry_and_E16_denominators=True));continue
  gap_widths=direction*(positions[1:,0]-positions[:-1,1])
  record=dict(original_parent_index=pi,original_parent_id=parent['original_parent_id'],
   original_child_ids=[c['original_child_id'] for c in children],sequence_index=si,
   child_parameter_endpoints_mm=positions.tolist(),original_child_count=len(children),
   original_gap_status=parent['gaps'],gap_widths_mm=gap_widths.tolist(),
   parent_chart_direction=direction,parent_is_closed=parent['is_closed'],
   entire_candidate_domain_mm=[0.,span],maximum_representation_residual_mm=residual,
   supervision_uses_original_order_without_reversing_or_merging=True)
  by_sequence.setdefault(si,[]).append(len(mapped));mapped.append(record)
 all_ids=[c['original_child_id'] for p in targets['parents'] for c in p['children']]
 declared=[cid for p in mapped+misses for cid in p['original_child_ids']]
 assert Counter(all_ids)==Counter(declared) and len(all_ids)==targets['original_children']
 return dict(version=VERSION,original_G=targets['original_children'],original_parents=len(targets['parents']),
  mapped_parents=mapped,representation_misses=misses,sequence_parent_mapping=by_sequence,
  mapped_children=sum(p['original_child_count'] for p in mapped),
  representation_miss_children=sum(p['original_child_count'] for p in misses),
  category_original_G={typ+'_'+kind:n for (typ,kind),n in categories.items()},
  candidate_count_including_all_background=len(sequences),
  maximum_distinct_parents_on_one_candidate=max([len(p) for p in by_sequence.values()] or [0]),
  maximum_original_children_per_parent=max([p['original_child_count'] for p in targets['parents']] or [0]),
  E16_unmapped_children_not_removed=True,all_original_child_ids_retained=True,
  all_candidate_samples_available_in_full_domain=True,full_contract_pass=False)
