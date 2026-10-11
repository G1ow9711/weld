"""Pure parent competition feature computation, extracted without numerical changes."""
import numpy as np
from scipy.spatial import cKDTree

def build_pair_features(parents,meta,pc):
 clouds=[];ends=[];centres=[];attrs=[];faces=[];profiles=[]
 for p,ids in zip(parents,pc):
  segments=p['segments'];length=np.array([s['length_mm'] for s in segments]);total=length.sum();e=np.array([np.asarray(s['points'])[[0,-1]] for s in segments]);centre=np.average(e.mean(1),axis=0,weights=length);pts=np.concatenate([np.asarray(s['points']) for s in segments]);sample=pts[np.linspace(0,len(pts)-1,33).astype(int)]
  clouds.append(sample);ends.append(e);centres.append(centre);attrs.append([p['child_count'],total,float(p['is_closed']),p['gaps'].count('SHARED'),p['gaps'].count('POSITIVE_GAP'),sum(s['length_mm'] for s in segments if s['kind']=='line')/total,sum(s['length_mm'] for s in segments if s['weld_type']=='horizontal')/total]);faces.append({tuple(sorted(pair)) for i in ids for pair in meta[i]['face_pairs'] if len(pair)==2 and min(pair)>=0});profiles.append({s['profile'] for s in segments})
 centres=np.asarray(centres);attrs=np.asarray(attrs);tree=cKDTree(centres);pairs=set(tree.query_pairs(25.));_,nearest=tree.query(centres,k=min(17,len(parents)))
 for a,neighbors in enumerate(nearest):
  for b in np.atleast_1d(neighbors):
   if a!=int(b):pairs.add(tuple(sorted((a,int(b)))))
 pairs=np.asarray(sorted(pairs),np.int32);features=[]
 for a,b in pairs:
  distance=np.linalg.norm(clouds[a][:,None,:]-clouds[b][None,:,:],axis=2);da=distance.min(1);db=distance.min(0);quant=np.r_[np.quantile(da,[0,.25,.5,.75,1]),np.quantile(db,[0,.25,.5,.75,1])].reshape(2,5);symmetric=np.r_[quant.min(0),quant.max(0)]
  equal=len(ends[a])==len(ends[b]);ordered=0.
  if equal:ordered=min(np.linalg.norm(ends[a]-ends[b],axis=2).max(),np.linalg.norm(ends[a]-ends[b][::-1,::-1],axis=2).max())
  fc=faces[a]&faces[b];pcross=profiles[a]&profiles[b]
  features.append(np.r_[np.log1p(np.linalg.norm(centres[a]-centres[b])),np.log1p(symmetric),np.log1p(abs(attrs[a]-attrs[b])),np.log1p(np.minimum(attrs[a],attrs[b])),float(equal),np.log1p(ordered),len(fc)/max(1,len(faces[a]|faces[b])),len(pcross)/max(1,len(profiles[a]|profiles[b]))])
 x=np.asarray(features,np.float32);assert x.shape==(len(pairs),29) and np.isfinite(x).all()
 return pairs,x
