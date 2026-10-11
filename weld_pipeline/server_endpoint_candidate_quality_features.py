"""Input-only symmetric quality features for actual compact line predictions.

No label, source ID, world coordinate, or inferred physical absence is a feature.
"""
import numpy as np
from scipy.spatial import cKDTree
from .server_fixed_fragment_queries import GRID
COLUMNS=['valid_interval','weld_type','log_length_mm','log_carrier_length_mm','length_fraction','carrier_margin_min','carrier_margin_max','log_supporting_queries','detector_score_mean','detector_score_max']
COLUMNS += [f'endpoint_{stat}_{agg}' for stat in ['min_probability','max_probability','probability_product'] for agg in ['mean','max']]
COLUMNS += ['query_center_abs_fraction_mean','query_center_abs_fraction_max']
COLUMNS += [f'endpoint_provenance_{agg}_{i}' for agg in ['min','max'] for i in range(12)]
COLUMNS += ['log_raw_distance_endpoint_min','log_raw_distance_endpoint_max','log_raw_distance_quarter_min','log_raw_distance_quarter_max','log_raw_distance_midpoint']

def features(compact,prediction,detector_scores,detector_roles,provenance,inp,xyz):
 pids=prediction['profile_ids'];bounds=prediction['interval_mm'];ids=prediction['selected_query_ids'];roles=detector_roles[ids];n=len(compact['profile_ids']);key=lambda p,b,r:(int(p),float(b[0]),float(b[1]),int(r));lookup={key(p,b,r):i for i,(p,b,r) in enumerate(zip(compact['profile_ids'],compact['interval_mm'],compact['weld_type']))};assert len(lookup)==n
 group=np.array([lookup[key(p,b,r)] for p,b,r in zip(pids,bounds,roles)],np.int64);count=np.bincount(group,minlength=n);assert np.all(count>0)
 def aggregate(v):
  maximum=np.full(n,-np.inf);np.maximum.at(maximum,group,v);mean=np.bincount(group,weights=v,minlength=n)/count;return mean,maximum
 valid=compact['interval_mm'][:,1]>compact['interval_mm'][:,0];length=np.diff(compact['interval_mm'],axis=1).ravel();full=inp.length[compact['query_ids']//4];assert np.all(full>0);margins=np.column_stack((compact['interval_mm'][:,0]/full,1-compact['interval_mm'][:,1]/full));columns=[valid.astype(float),compact['weld_type'],np.log1p(length),np.log1p(full),length/full,margins.min(1),margins.max(1),np.log1p(count),*aggregate(detector_scores[ids])]
 probability=prediction['endpoint_probability'].astype(np.float64);assert probability.shape==(len(ids),2) and np.isfinite(probability).all()
 for v in (probability.min(1),probability.max(1),probability.prod(1)):columns.extend(aggregate(v))
 spans=inp.spans[inp.canonical[ids//4],:2];q=spans[:,0]+GRID[ids%4]*(spans[:,1]-spans[:,0]);width=bounds[:,1]-bounds[:,0];relative=np.zeros(len(ids));good=width>0;relative[good]=abs((q[good]-bounds[good].mean(1))/width[good]);columns.extend(aggregate(relative))
 pair=provenance[compact['event_ids']];assert pair.shape==(n,2,12)
 for values in (pair.min(1),pair.max(1)):columns.extend(values.T)
 distances=np.zeros((n,5));ends=compact['endpoints_mm'];tree=cKDTree(xyz)
 if valid.any():
  fractions=np.array([0.,.25,.5,.75,1.]);samples=ends[valid,0,None,:]+fractions[None,:,None]*(ends[valid,1,None,:]-ends[valid,0,None,:]);distances[valid]=tree.query(samples.reshape(-1,3),workers=2)[0].reshape(-1,5)
 columns.extend([np.log1p(distances[:,[0,4]].min(1)),np.log1p(distances[:,[0,4]].max(1)),np.log1p(distances[:,[1,3]].min(1)),np.log1p(distances[:,[1,3]].max(1)),np.log1p(distances[:,2])]);x=np.column_stack(columns).astype(np.float32);assert x.shape==(n,len(COLUMNS)) and np.isfinite(x).all();return x,group
