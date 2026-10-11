"""Common real carrier features for native, recombined, and kept parents.

Both source families use current XYZ-derived fragment features. This is a new
contract; neither old 526-column interval vectors nor old child vectors are padded.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import tools.rebuild_legacy_downstream as runtime
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_child_parent_supervision_1255 import singleton,state_geometry,link_features
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'server_rebuild/expansion_v1/unified_parent_scoring_v1'

def geometry_key(s):
 fields=('kind','points','weld_type','profile','arc_parameters','is_closed','predicted_side')
 return hashlib.sha256(json.dumps({k:s[k] for k in fields if k in s},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def intervals(curve,segment):
 q,d=curve.project(np.asarray(segment['points']));pieces=[]
 for a,b in zip(q[:-1],q[1:]):
  if a==b:continue
  if curve.kind=='arc' and curve.arc.is_closed and abs(a-b)>curve.length/2:pieces.extend([(0.,float(min(a,b))),(float(max(a,b)),curve.length)])
  else:pieces.append((float(min(a,b)),float(max(a,b))))
 merged=[]
 for a,b in sorted(pieces):
  if b<=a:continue
  if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
  else:merged.append((a,b))
 assert merged,'No finite carrier projection; unavailable features must not be fabricated'
 return merged,float(d.max())

def run(key):
 started=time.perf_counter();dest=DEST/('source_'+key);dest.mkdir(parents=True,exist_ok=False)
 runtime.BASE=ROOT/'server_rebuild/expansion_v1/frozen_features';runtime.CASE=runtime.BASE/'legacy_runtime'/('source_'+key);case=runtime.module('sjg_relative_ordinal_runtime_1150').load_case(key);graph=case['graph']
 with np.load(runtime.CASE/'run288'/('source_'+key+'_alias_map.npz')) as z:representative=z['fragment_representative']
 hypotheses=ROOT/'server_rebuild/expansion_v1/parent_hypotheses_v1'/('source_'+key);hr=read(hypotheses/'receipt.json');assert digest(hypotheses/'hypotheses.json')==hr['files']['hypotheses.json']
 pool=read(hypotheses/'hypotheses.json');basefolder=runtime.CASE/'run446'/('test_'+key);br=read(basefolder/'prediction_receipt.json');assert digest(basefolder/'final_decoded.json')==br['decoded_sha256']
 baseline=read(basefolder/'final_decoded.json');parents=pool['parents']+baseline['parents'];n_hyp=len(pool['parents']);cache={};records=[];child_vectors=[];child_metadata=[];child_segments=[]
 def child(s):
  signature=geometry_key(s)
  if signature in cache:return cache[signature]
  pid=s['profile'];curve=graph.curve(pid);pieces,projection_error=intervals(curve,s);lo,hi=graph.offsets[pid:pid+2];span=graph.spans[lo:hi];w=np.zeros(len(span))
  for a,b in pieces:w+=np.maximum(0,np.minimum(span[:,1],b)-np.maximum(span[:,0],a))
  selected=np.flatnonzero(w>0);assert len(selected);fragments=selected+lo;reps=representative[fragments];locations=np.searchsorted(case['canonical'],reps);assert np.array_equal(case['canonical'][locations],reps)
  rows=np.column_stack((2*locations,2*locations+1)).ravel();weights=np.repeat(w[selected]/2,2);xx=np.asarray(case['x'][rows],np.float64)
  closed=bool(s.get('is_closed',s.get('arc_parameters',{}).get('is_closed',False)))
  extra=[np.log1p(s['length_mm']),np.log1p(curve.length),s['length_mm']/curve.length,float(weights.sum()/s['length_mm']),np.log1p(projection_error),float(s['kind']=='line'),float(s['kind']=='arc'),float(closed),float(s['weld_type']=='horizontal'),float(s['weld_type']=='vertical')]
  vector=np.r_[(xx*weights[:,None]).sum(0)/weights.sum(),xx.min(0),xx.max(0),extra];assert vector.shape==(466,) and np.isfinite(vector).all()
  faces=graph.profiles[pid].get('face_indices',[])[:2];md=dict(face_pairs=[faces],physical_sides=[s.get('predicted_side','UNKNOWN')],side_known=s.get('predicted_side','UNKNOWN') not in ('UNKNOWN',None),length_mm=s['length_mm'],line_fraction=float(s['kind']=='line'))
  # No probability field is invented: state_geometry uses only actual geometry and face references.
  index=len(records);cache[signature]=index;records.append(dict(profile=pid,projected_intervals_mm=pieces,projection_max_error_mm=projection_error,fragments=fragments.tolist(),canonical_feature_rows=rows.tolist(),signature=signature));child_vectors.append(vector);child_metadata.append(md);child_segments.append(s);return index
 parent_children=[];vectors=[]
 for p in parents:
  ids=[child(s) for s in p['segments']];parent_children.append(ids);x=np.array([child_vectors[i] for i in ids]);length=np.array([s['length_mm'] for s in p['segments']]);total=length.sum();count=len(ids)
  state=state_geometry(ParentRelations([singleton(child_segments[i]) for i in ids],[child_metadata[i] for i in ids]));edges=[(2*i,2*(i+1)) for i in range(count-1)]
  if p['is_closed'] and count>1:edges.append((2*(count-1),0))
  if edges:
   forward=link_features(state,edges);reverse=link_features(state,[(b^1,a^1) for a,b in edges]);relations=np.r_[((forward+reverse)/2).sum(0),abs(forward-reverse).sum(0)]
  else:relations=np.zeros(48) # Defined empty sums; edge count below explicitly records applicability.
  distances=[float(np.linalg.norm(state['ends'][a,1]-state['ends'][b,0])) for a,b in edges]
  extra=[np.log1p(count),np.log1p(total),float(p['is_closed']),float(np.mean([s['kind']=='line' for s in p['segments']])),float(np.mean([s['weld_type']=='horizontal' for s in p['segments']])),p['gaps'].count('POSITIVE_GAP'),p['gaps'].count('SHARED'),length.max()/total,length.min()/total,np.log1p(np.linalg.norm(state['ends'][0,0]-state['ends'][2*(count-1),1])),np.log1p(max(distances,default=0.)),len(edges)]
  vectors.append(np.r_[(x*length[:,None]).sum(0)/total,x.min(0),x.max(0),relations,extra])
 features=np.asarray(vectors,np.float32);assert features.shape==(len(parents),1458) and np.isfinite(features).all();case['x']._mmap.close()
 np.save(dest/'features.npy',features);np.save(dest/'child_features.npy',np.asarray(child_vectors,np.float32));write(dest/'child_records.json',records);write(dest/'child_metadata.json',child_metadata);write(dest/'child_segments.json',child_segments);write(dest/'parent_children.json',parent_children)
 write(dest/'parents.json',dict(parents=parents,invalid_predictions=baseline['invalid_predictions'],total_prediction_children=sum(p['child_count'] for p in parents)+sum(v['counted_false_children'] for v in baseline['invalid_predictions'])))
 write(dest/'default_configuration.json',dict(parent_indices=list(range(n_hyp,len(parents))),baseline_sha256=digest(basefolder/'final_decoded.json'),invalid_predictions=baseline['invalid_predictions']))
 report=dict(source=key,schema='SIDE_SYMMETRIC_REAL_CARRIER_CHILD466_PARENT1458_V1',columns=1458,hypotheses=n_hyp,default_parents=len(baseline['parents']),unique_children=len(records),files={name:digest(dest/name) for name in ['features.npy','child_features.npy','child_records.json','child_metadata.json','child_segments.json','parent_children.json','parents.json','default_configuration.json']},original_feature_receipt_sha256=case['feature_receipt_sha256'],hypothesis_sha256=digest(hypotheses/'hypotheses.json'),baseline_sha256=digest(basefolder/'final_decoded.json'),source_sha256=digest(__file__),GT_reads=0,STEP_reads=0,DEV_reads=0,fit_calls=0,no_interval_padding=True,missing_carrier_features_raise=True,seconds=time.perf_counter()-started)
 write(dest/'receipt.json',report);print({k:v for k,v in report.items() if k!='files'},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=[c['key'] for c in read(ROOT/'server_rebuild/expansion_v1/coverage_cohort_protocol.json')['cases']]);a=ap.parse_args()
 with threadpool_limits(limits=2):run(a.key)
