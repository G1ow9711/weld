"""Preserve input-only event provenance without altering frozen candidate coordinates."""
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_finite_plane_endpoint_events import FaceEvents
from weld_pipeline.server_face_boundary_endpoint_events import FaceBoundaryEvents
from weld_pipeline.server_sampled_surface_boundary_events import SampledSurfaceBoundaryEvents
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DEST=BASE/'runtime_boundary_event_provenance_v1'
COLUMNS=['carrier_end','finite_plane_crossing','convex_domain_boundary','sampled_surface_boundary','log1p_distinct_source_faces','min_abs_normal_dot_tangent','max_abs_normal_dot_tangent','min_abs_plane_distance_mm','max_abs_plane_distance_mm','max_source_normal_sine','log1p_min_support_count','log1p_max_support_count']
def describe(generator,curve,support_counts):
 q0,s0=FaceEvents.events(generator,curve);q1,s1=FaceBoundaryEvents.events(generator,curve);q,s=generator.events(curve)
 assert np.array_equal(q1[:len(q0)],q0) and np.array_equal(s1[:len(s0)],s0)
 assert np.array_equal(q[:len(q1)],q1) and np.array_equal(s[:len(s1)],s1)
 kinds=np.full(len(q),3);kinds[:len(q1)]=2;kinds[:len(q0)]=1;kinds[s<0]=0
 values,inverse=np.unique(q,return_inverse=True);features=np.zeros((len(values),len(COLUMNS)),np.float32)
 for kind in range(4):np.maximum.at(features[:,kind],inverse,(kinds==kind).astype(np.float32))
 order=np.argsort(inverse,kind='stable');groups=np.split(order,np.flatnonzero(np.diff(inverse[order]))+1)
 for event,group in enumerate(groups):
  faces=np.unique(s[group]);faces=faces[faces>=0]
  if not len(faces):continue
  normals=generator.normals[faces];dot=abs(normals@curve.direction);point=curve.at(values[event:event+1])[0];distance=abs(normals@point-generator.offsets[faces]);normal_sine=np.sqrt(np.maximum(0,1-np.clip(normals@normals.T,-1,1)**2)).max();counts=support_counts[faces]
  features[event,4:]=[np.log1p(len(faces)),dot.min(),dot.max(),distance.min(),distance.max(),normal_sine,np.log1p(counts.min()),np.log1p(counts.max())]
 assert np.isfinite(features).all()
 return values,features

def main():
 DEST.mkdir(exist_ok=False);keys=['new_cfc732e5f068','new_ceeae49d6777'];write(DEST/'protocol.json',dict(keys=keys,columns=COLUMNS,scope='Input-only event provenance on frozen detector-selected runtime candidate caches; no labels or targets read; no new events, snapping or proximity merging',source_kind_names_are_geometric_not_physical=True,physical_contact_certified=False,DEV_reads=0,heldout_label_reads=0))
 for key in keys:
  started=time.perf_counter();out=DEST/key;out.mkdir();cache=BASE/'runtime_boundary_anchor_cache_v1'/key;receipt=read(cache/'receipt.json');assert digest(cache/'events.npz')==receipt['events_sha256'];ev=np.load(cache/'events.npz');source=BASE/'frozen_features/legacy_runtime'/('source_'+key);folder=source/'run185'/('source_'+key);freeze=read(folder/'input_freeze.json');xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==freeze['input_sha256']
  for name in ['surfaces.json','surface_support.npz','profiles.json']:assert digest(folder/name)==freeze['files'][name]
  xyz=np.load(xyzpath,mmap_mode='r');faces=read(folder/'surfaces.json');inp=FixedQueries(source,key)
  with np.load(folder/'surface_support.npz') as support:
   generator=SampledSurfaceBoundaryEvents(xyz,faces,support);counts=np.array([len(support[f['point_ids_key']]) if f.get('point_ids_key') in support else 0 for f in faces])
  chunks=[]
  for i,pid in enumerate(ev['profile_ids']):
   curve=CurveCarrier(inp.profiles[pid]['representation']);assert curve.kind=='line';values,features=describe(generator,curve,counts);a,b=ev['offsets'][i:i+2];assert np.array_equal(values,ev['positions_mm'][a:b]),'Candidate coordinates changed';chunks.append(features)
   if i%100==0:print(key,i,'/',len(ev['profile_ids']),flush=True)
  features=np.concatenate(chunks);np.save(out/'features.npy',features);write(out/'receipt.json',dict(key=key,rows=len(features),columns=COLUMNS,features_sha256=digest(out/'features.npy'),events_receipt_sha256=digest(cache/'receipt.json'),source_sha256=digest(__file__),generator_sha256=digest(ROOT/'weld_pipeline/server_sampled_surface_boundary_events.py'),input_bindings=receipt['input_bindings'],exact_coordinates_preserved=True,source_kind_counts=(features[:,:4]>0).sum(0).tolist(),GT_reads=0,STEP_reads=0,fit_calls=0,seconds=time.perf_counter()-started));inp.close();xyz._mmap.close();print('COMPLETE',key,len(features),flush=True)
 write(DEST/'receipt.json',dict(status='TWO_RUNTIME_INPUT_PROVENANCE_CACHES_COMPLETE',GT_reads=0,STEP_reads=0,fit_calls=0,DEV_reads=0))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
