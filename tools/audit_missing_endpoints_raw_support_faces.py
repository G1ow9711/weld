"""TRAIN diagnosis: which detected faces contain the nearest raw endpoint point?"""
from pathlib import Path
import collections
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def run(key):
 p=BASE/'training_missing_endpoint_geometry_v1'/key/'report.json';report=read(p);folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);freeze=read(folder/'input_freeze.json');xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==freeze['input_sha256'];xyz=np.load(xyzpath,mmap_mode='r');tree=cKDTree(xyz);faces=read(folder/'surfaces.json');profiles=read(folder/'profiles.json');assert digest(folder/'surface_support.npz')==freeze['files']['surface_support.npz'];rows=[]
 for r in report['rows']:
  best=r['best_event_pair']
  if best is None or best['max_endpoint_error_mm']<=3:continue
  for side,e in enumerate(best['endpoint_errors_mm']):
   if e<=3:continue
   ref=np.asarray(r['reference_endpoints_mm'][side]);d,idx=tree.query(ref);ends=np.asarray(profiles[best['profile']]['representation']['endpoints_mm']);v=ends[1]-ends[0];v/=np.linalg.norm(v);rows.append(dict(parent=r['parent'],child=r['child'],side=side,profile=best['profile'],event_error_mm=e,reference=ref.tolist(),nearest_raw_id=int(idx),raw_distance_mm=float(d),direction=v.tolist(),support_faces=[]))
 wanted={r['nearest_raw_id'] for r in rows};mapping=collections.defaultdict(list)
 with np.load(folder/'surface_support.npz') as support:
  for i,f in enumerate(faces):
   k=f.get('point_ids_key')
   if k not in support:continue
   ids=set(np.asarray(support[k]).tolist())&wanted
   for idx in ids:mapping[idx].append(i)
 for row in rows:
  for i in mapping[row['nearest_raw_id']]:
   f=faces[i];n=np.asarray(f['normal']);norm=np.linalg.norm(n)
   if norm==0:continue
   n=n/norm;row['support_faces'].append(dict(face=i,abs_normal_dot_carrier=float(abs(n@row['direction'])),reference_plane_distance_mm=float(abs(n@row['reference']-float(f['offset'])/norm))))
 summary=dict(missing_heads=len(rows),nearest_raw_within3=sum(r['raw_distance_mm']<=3 for r in rows),nearest_raw_without_detected_face=sum(not r['support_faces'] for r in rows),near_raw_without_detected_face=sum(r['raw_distance_mm']<=3 and not r['support_faces'] for r in rows),near_raw_with_any_parallel_face=sum(r['raw_distance_mm']<=3 and any(f['abs_normal_dot_carrier']<=1e-8 for f in r['support_faces']) for r in rows))
 out=BASE/'missing_endpoints_raw_support_faces_v1'/key;out.mkdir(parents=True,exist_ok=False);write(out/'report.json',dict(rows=rows,summary=summary,diagnosis_sha256=digest(p),scope='TRAIN-only nearest one observed point and its detected-face memberships. Lack of this membership does not prove absence of all nearby faces; no candidate or runtime rule changed.',fit_calls=0,DEV_reads=0));print(key,summary,flush=True);xyz._mmap.close()
if __name__=='__main__':
 for key in ['new_034230635b7a','new_181887f85097']:run(key)
