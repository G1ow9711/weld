"""Explain TRAIN-only infinite-plane witnesses rejected by finite observed domains."""
from pathlib import Path
import collections
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.server_finite_plane_endpoint_events import FaceEvents
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def run(key):
 diagnosis=BASE/'training_missing_endpoint_geometry_v1'/key/'report.json';r=read(diagnosis);folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);freeze=read(folder/'input_freeze.json');xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==freeze['input_sha256']==r['input_sha256']
 for name in ['surfaces.json','surface_support.npz','profiles.json']:assert digest(folder/name)==freeze['files'][name]
 xyz=np.load(xyzpath,mmap_mode='r');profiles=read(folder/'profiles.json');faces=read(folder/'surfaces.json');support=np.load(folder/'surface_support.npz');geometry=FaceEvents(xyz,faces,support);trees={};rows=[]
 for row in r['rows']:
  if row['reason']!='FINITE_EVENT_MISSING_DESPITE_INFINITE_PLANE_PAIR':continue
  pid=row['best_infinite_plane_pair']['profile'];ends=np.asarray(profiles[pid]['representation']['endpoints_mm']);direction=ends[1]-ends[0];length=np.linalg.norm(direction);direction/=length;slopes=geometry.normals@direction;valid=np.flatnonzero(abs(slopes)>1e-8);t=(geometry.offsets[valid]-geometry.normals[valid]@ends[0])/slopes[valid];inside=(t>=0)&(t<=length);t=np.r_[0.,length,t[inside]];ids=np.r_[-1,-1,valid[inside]];points=ends[0]+t[:,None]*direction;reference=np.asarray(row['reference_endpoints_mm']);dist=np.linalg.norm(points[:,None,:]-reference[None,:,:],axis=2);head=[]
  for side in (0,1):
   # Retain every near-reference plane intersection, not just the most favorable face.
   candidates=[]
   for j in np.flatnonzero(dist[:,side]<=3.):
    face=int(ids[j]);item=dict(face=face,reference_distance_mm=float(dist[j,side]),position_mm=float(t[j]))
    if face<0:item['finite_status']='CARRIER_END'
    elif face not in geometry.domains:item['finite_status']='NO_OBSERVED_DOMAIN'
    else:
     A,b=geometry.domains[face];violation=float(np.max(A@points[j]+b));item.update(finite_status='ACCEPTED' if violation<=1e-6 else 'OUTSIDE_OBSERVED_DOMAIN',maximum_halfspace_violation=violation)
    if face>=0:
     support_key=faces[face].get('point_ids_key');item['support_key']=support_key
     if support_key in support:
      if face not in trees:trees[face]=cKDTree(np.asarray(xyz[support[support_key]]))
      item['nearest_face_support_mm']=float(trees[face].query(points[j])[0]);item['support_points']=int(trees[face].n);item['abs_normal_dot_carrier']=float(abs(slopes[face]))
    candidates.append(item)
   head.append(dict(side=side,candidates=candidates,raw_nearest_reference_mm=row['raw_nearest_endpoint_mm'][side]))
  rows.append(dict(parent=row['parent'],child=row['child'],profile=pid,heads=head))
 counter=collections.Counter(c['finite_status'] for row in rows for h in row['heads'] for c in h['candidates']);out=BASE/'training_finite_event_rejections_v1'/key;out.mkdir(parents=True,exist_ok=False);write(out/'report.json',dict(rows=rows,status_counts=dict(counter),diagnosis_sha256=digest(diagnosis),source_sha256=digest(__file__),scope='TRAIN GT-guided diagnostic only; all near-reference plane roots on the prior best infinite-plane profile. Outside-domain intersections remain invalid; no candidate changes or threshold fit.',fit_calls=0,heldout_reads=0,DEV_reads=0));print(key,dict(counter),flush=True);support.close();xyz._mmap.close()
if __name__=='__main__':
 with threadpool_limits(limits=2):
  for key in ['new_034230635b7a','new_181887f85097']:run(key)
