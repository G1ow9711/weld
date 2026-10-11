"""Shared XYZ-carrier features for complete parent candidates, no label access.

Native and new interval parents use the same measured fragment features. No
source key, reference owner or CAD identity is a learned feature. This preserves
the existing 466-column child / 1458-column parent contract.
"""
import hashlib
import json
from types import SimpleNamespace
import numpy as np
from .sjg_curve_queries_907 import CurveCarrier
from .sjg_child_parent_supervision_1255 import singleton,state_geometry,link_features
from .sjg_parent_set_features_1160 import ParentRelations

def geometry_key(s):
 fields=('kind','points','weld_type','profile','arc_parameters','is_closed','predicted_side')
 return hashlib.sha256(json.dumps({k:s[k] for k in fields if k in s},sort_keys=True,separators=(',',':'),default=lambda value:value.tolist()).encode()).hexdigest()

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
 return merged,float(d.max())

def encode_parents(inp, parents):
 graph=SimpleNamespace(offsets=inp.offsets,spans=inp.spans,profiles=inp.profiles,
                       curve=lambda pid:CurveCarrier(inp.profiles[int(pid)]['representation']))
 representative=inp.representative
 case={'canonical':inp.canonical,'x':inp.raw}
 cache={};records=[];child_vectors=[];child_metadata=[];child_segments=[]
 def child(s):
  signature=geometry_key(s)
  if signature in cache:return cache[signature]
  pid=s['profile'];curve=graph.curve(pid);pieces,projection_error=intervals(curve,s);lo,hi=graph.offsets[pid:pid+2];span=graph.spans[lo:hi];w=np.zeros(len(span))
  for a,b in pieces:w+=np.maximum(0,np.minimum(span[:,1],b)-np.maximum(span[:,0],a))
  collapsed=False
  if not pieces:
   # A positive segment shorter than coordinate roundoff can project to one
   # representable arclength value. Use the measured containing fragment as
   # the normalized integral's point limit; never lengthen the output curve.
   points=np.asarray(s['points'],np.float64);scale=max(1.,curve.length,float(np.abs(points).max()));bound=8*np.spacing(scale)
   assert 0<s['length_mm']<=bound and np.linalg.norm(points[-1]-points[0])<=bound,'No finite carrier projection outside numerical point limit'
   q,_=curve.project(points);assert np.all(q==q[0]) and len(span)
   index=min(int(np.searchsorted(span[:,1],q[0],side='right')),len(span)-1)
   assert span[index,0]<=q[0]<=span[index,1],'Collapsed point has no measured fragment'
   w[index]=s['length_mm'];pieces=[(float(q[0]),float(q[0]))];collapsed=True
  selected=np.flatnonzero(w>0);assert len(selected);fragments=selected+lo;reps=representative[fragments];locations=np.searchsorted(case['canonical'],reps);assert np.array_equal(case['canonical'][locations],reps)
  rows=np.column_stack((2*locations,2*locations+1)).ravel();weights=np.repeat(w[selected]/2,2);xx=np.asarray(case['x'][rows],np.float64)
  closed=bool(s.get('is_closed',s.get('arc_parameters',{}).get('is_closed',False)))
  extra=[np.log1p(s['length_mm']),np.log1p(curve.length),s['length_mm']/curve.length,float(weights.sum()/s['length_mm']),np.log1p(projection_error),float(s['kind']=='line'),float(s['kind']=='arc'),float(closed),float(s['weld_type']=='horizontal'),float(s['weld_type']=='vertical')]
  vector=np.r_[(xx*weights[:,None]).sum(0)/weights.sum(),xx.min(0),xx.max(0),extra];assert vector.shape==(466,) and np.isfinite(vector).all()
  faces=graph.profiles[pid].get('face_indices',[])[:2];md=dict(face_pairs=[faces],physical_sides=[s.get('predicted_side','UNKNOWN')],side_known=s.get('predicted_side','UNKNOWN') not in ('UNKNOWN',None),length_mm=s['length_mm'],line_fraction=float(s['kind']=='line'))
  # No probability field is invented: state_geometry uses only actual geometry and face references.
  index=len(records);cache[signature]=index;record=dict(profile=pid,projected_intervals_mm=pieces,projection_max_error_mm=projection_error,fragments=fragments.tolist(),canonical_feature_rows=rows.tolist(),signature=signature)
  if collapsed:record['numerically_collapsed_projection']=True
  records.append(record);child_vectors.append(vector);child_metadata.append(md);child_segments.append(s);return index
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
 features=np.asarray(vectors,np.float32).reshape(-1,1458);assert features.shape==(len(parents),1458) and np.isfinite(features).all()
 return dict(features=features,child_features=np.asarray(child_vectors,np.float32).reshape(-1,466),
             child_records=records,child_metadata=child_metadata,
             child_segments=child_segments,parent_children=parent_children)
