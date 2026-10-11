"""Analytic per-child arclength competition features; schema V3, retraining required."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.server_parent_geometry_sampling import parent_geometry, endpoint_correspondence

def build_pair_features(parents,meta,pc):
 clouds=[];ends=[];centres=[];attrs=[];faces=[];profiles=[]
 for p,ids in zip(parents,pc):
  segments=p['segments'];length=np.array([s['length_mm'] for s in segments]);total=length.sum();sample,e,centre=parent_geometry(p)
  clouds.append(sample);ends.append(e);centres.append(centre);attrs.append([p['child_count'],total,float(p['is_closed']),p['gaps'].count('SHARED'),p['gaps'].count('POSITIVE_GAP'),sum(s['length_mm'] for s in segments if s['kind']=='line')/total,sum(s['length_mm'] for s in segments if s['weld_type']=='horizontal')/total]);faces.append({tuple(sorted(pair)) for i in ids for pair in meta[i]['face_pairs'] if len(pair)==2 and min(pair)>=0});profiles.append({s['profile'] for s in segments})
 centres=np.asarray(centres);attrs=np.asarray(attrs);tree=cKDTree(centres);distances,_=tree.query(centres,k=min(17,len(parents)));pairs=set()
 # Complete the kth-distance tie, with a declared numerical tolerance in mm.
 # The resulting degree may exceed 16; no row-index tie breaker is applied.
 for a,d in enumerate(distances):
  radius=max(25.,float(np.max(d)))+1e-7
  for b in tree.query_ball_point(centres[a],radius):
   if a!=b:pairs.add(tuple(sorted((a,b))))
 pairs=np.asarray(sorted(pairs),np.int32);features=[]
 for a,b in pairs:
  distance=np.linalg.norm(clouds[a][:,None,:]-clouds[b][None,:,:],axis=2);da=distance.min(1);db=distance.min(0);quant=np.r_[np.quantile(da,[0,.25,.5,.75,1]),np.quantile(db,[0,.25,.5,.75,1])].reshape(2,5);symmetric=np.r_[quant.min(0),quant.max(0)]
  equal=len(ends[a])==len(ends[b]);ordered=0.
  if equal:ordered=endpoint_correspondence(ends[a],ends[b],parents[a]['is_closed'] and parents[b]['is_closed'])
  fc=faces[a]&faces[b];pcross=profiles[a]&profiles[b]
  features.append(np.r_[np.log1p(np.linalg.norm(centres[a]-centres[b])),np.log1p(symmetric),np.log1p(abs(attrs[a]-attrs[b])),np.log1p(np.minimum(attrs[a],attrs[b])),float(equal),np.log1p(ordered),len(fc)/max(1,len(faces[a]|faces[b])),len(pcross)/max(1,len(profiles[a]|profiles[b]))])
 x=np.round(np.asarray(features,np.float64),6).astype(np.float32);assert x.shape==(len(pairs),29) and np.isfinite(x).all()
 return pairs,x
