"""TRAIN-only endpoint candidate geometry diagnosis; does not alter candidates."""
from pathlib import Path
import collections
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 data=BASE/'exhaustive_carrier_supervision_v1/positive_query'/key;records=read(data/'records.json');owners=collections.defaultdict(set)
 for row in records:
  for owner in row['owners']:owners[tuple(owner)].add(row['profile'])
 folder=BASE/'sampled_boundary_anchor_training_v1'/key;r=read(folder/'receipt.json');assert digest(folder/'events.npz')==r['events_sha256'];ev=np.load(folder/'events.npz');pmap={int(p):i for i,p in enumerate(ev['profile_ids'])};source=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);freeze=read(source/'input_freeze.json');profiles=read(source/'profiles.json');faces=read(source/'surfaces.json');normals=np.array([f['normal'] for f in faces]);offsets=np.array([f['offset'] for f in faces]);xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==freeze['input_sha256'];xyz=np.load(xyzpath,mmap_mode='r');tree=cKDTree(xyz);truthpath=BASE/'labels'/key/'targets.json';truth=read(truthpath);results=[]
 for pi,p in enumerate(truth['parents']):
  for ci,c in enumerate(p['children']):
   if c['representation']['kind']!='line':continue
   ref=np.array(c['representation']['endpoints_mm']);rawdist=tree.query(ref,workers=2)[0];best=None;plane_best=None
   for pid in sorted(owners[pi,ci]):
    if pid not in pmap:continue
    ends=np.asarray(profiles[pid]['representation']['endpoints_mm']);direction=ends[1]-ends[0];length=np.linalg.norm(direction);direction/=length;index=pmap[pid];a,b=ev['offsets'][index:index+2];q=ev['positions_mm'][a:b];points=ends[0]+q[:,None]*direction;dist=np.linalg.norm(points[:,None,:]-ref[None,:,:],axis=2);near=dist.argmin(0);errors=dist[near,np.arange(2)];score=float(errors.max());candidate=dict(profile=pid,max_endpoint_error_mm=score,endpoint_errors_mm=errors.tolist(),nearest_event_mm=q[near].tolist(),nearest_event_xyz=points[near].tolist())
    if best is None or score<best['max_endpoint_error_mm']:best=candidate
    slopes=normals@direction;valid=np.flatnonzero(abs(slopes)>1e-8);t=(offsets[valid]-normals[valid]@ends[0])/slopes[valid];inside=(t>=0)&(t<=length);t=np.r_[0.,length,t[inside]];points=ends[0]+t[:,None]*direction;errors=np.linalg.norm(points[:,None,:]-ref[None,:,:],axis=2).min(0);score=float(errors.max())
    if plane_best is None or score<plane_best['max_endpoint_error_mm']:plane_best=dict(profile=pid,max_endpoint_error_mm=score,endpoint_errors_mm=errors.tolist())
   reason='NO_RELIABLE_FIXED_QUERY_PROFILE' if best is None else ('EVENT_PAIR_WITHIN3_GEOMETRY_ONLY' if best['max_endpoint_error_mm']<=3 else ('FINITE_EVENT_MISSING_DESPITE_INFINITE_PLANE_PAIR' if plane_best['max_endpoint_error_mm']<=3 else 'NO_CLOSE_PAIR_IN_EVENT_OR_INFINITE_PLANE_PROPOSALS'))
   results.append(dict(parent=pi,child=ci,reference_endpoints_mm=ref.tolist(),reference_length_mm=float(np.linalg.norm(ref[1]-ref[0])),raw_nearest_endpoint_mm=rawdist.tolist(),query_profile_count=len(owners[pi,ci]),best_event_pair=best,best_infinite_plane_pair=plane_best,reason=reason))
 out=BASE/'training_missing_endpoint_geometry_v1'/key;out.mkdir(parents=True,exist_ok=False);counts=dict(collections.Counter(r['reason'] for r in results));write(out/'report.json',dict(rows=results,counts=counts,event_receipt_sha256=digest(folder/'receipt.json'),target_sha256=digest(truthpath),input_sha256=digest(xyzpath),scope='Offline TRAIN geometric endpoint proximity on reliable query profiles only; ignores side ordering, original D3 parent/type/assembly. Infinite planes diagnostic only, no candidate replacement; raw point distances do not define weldability.',fit_calls=0,DEV_reads=0));print(counts);xyz._mmap.close()
if __name__=='__main__':
 for key in ['new_034230635b7a','new_181887f85097']:run(key)
