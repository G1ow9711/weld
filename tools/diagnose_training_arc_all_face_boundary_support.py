"""TRAIN-only compare reference arcs with observed finite planar boundaries."""
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.server_sampled_surface_boundary_events import SampledSurfaceBoundaryEvents
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'training_arc_all_face_boundary_support_diagnostic_v1'
def segment_distances(points,segments):
 a=segments[:,0];v=segments[:,1]-a;vv=np.sum(v*v,axis=1);good=vv>1e-20;a=a[good];v=v[good];vv=vv[good];result=np.full(len(points),np.inf)
 for start in range(0,len(points),8):
  p=points[start:start+8];delta=p[:,None,:]-a[None];t=np.clip(np.einsum('bsi,si->bs',delta,v)/vv,0,1);result[start:start+8]=np.linalg.norm(delta-t[:,:,None]*v[None],axis=2).min(1)
 return result
def main():
 started=time.perf_counter();key='new_ac2008a03607';OUT.mkdir(exist_ok=False);folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);freeze=read(folder/'input_freeze.json');xyzpath=BASE/'inputs'/key/'xyz_mm.npy';assert digest(xyzpath)==freeze['input_sha256']
 for name in ['surfaces.json','surface_support.npz']:assert digest(folder/name)==freeze['files'][name]
 xyz=np.load(xyzpath,mmap_mode='r');faces=read(folder/'surfaces.json')
 with np.load(folder/'surface_support.npz') as support:g=SampledSurfaceBoundaryEvents(xyz,faces,support)
 # Input boundary construction completes before reference geometry is read.
 write(OUT/'input_boundary_freeze.json',dict(key=key,input_sha256=freeze['input_sha256'],surface_sha256=freeze['files']['surfaces.json'],support_sha256=freeze['files']['surface_support.npz'],boundary_faces=len(g.boundaries),module_sha256=digest('weld_pipeline/server_sampled_surface_boundary_events.py'),GT_reads=0))
 truthpath=BASE/'labels'/key/'targets.json';truth=read(truthpath);tree=cKDTree(xyz);rows=[];edge_cache={}
 for pi,parent in enumerate(truth['parents']):
  for ci,c in enumerate(parent['children']):
   if c['representation']['kind']!='arc':continue
   arc=CircularArc.from_parameters(c['representation']['arc_parameters']);points=arc.point_at(np.linspace(0,arc.length_mm,int(np.ceil(arc.length_mm))+1));raw=tree.query(points,workers=2)[0];union=np.full(len(points),np.inf);nearest_face=np.full(len(points),-1,np.int64);best=None;count=0
   for face,(origin,basis,edges) in g.boundaries.items():
    if face not in edge_cache:edge_cache[face]=origin+edges@basis.T
    d=segment_distances(points,edge_cache[face]);nearest_face[d<union]=face;union=np.minimum(union,d);count+=1
    if best is None or d.max()<best['sampled_maximum_distance_mm']:best=dict(face=int(face),sampled_maximum_distance_mm=float(d.max()),median_distance_mm=float(np.median(d)))
   finite=np.isfinite(union);row=dict(parent=pi,child=ci,radius_mm=float(arc.radius_mm),length_mm=float(arc.length_mm),samples=len(points),eligible_observed_boundary_faces=count,raw_distance_quantiles_mm=np.quantile(raw,[0,.5,.95,1]).tolist(),boundary_union_distance_quantiles_mm=np.quantile(union,[0,.5,.95,1]).tolist() if finite.all() else None,nearest_face_normal_dot_arc_normal_quantiles=np.quantile(abs(g.normals[nearest_face]@arc.plane_normal),[0,.5,1]).tolist() if (nearest_face>=0).all() else None,samples_with_boundary_distance_le3=int((union<=3).sum()),best_single_observed_face=best);rows.append(row);print(row,flush=True)
 write(OUT/'report.json',dict(key=key,rows=rows,target_sha256=digest(truthpath),input_freeze_sha256=digest(OUT/'input_boundary_freeze.json'),scope='Post-freeze TRAIN diagnostic with <=1mm reference sampling; nearest finite boundary segments on ALL observed face planes; no orientation or plane-distance prefilter. Union may span disconnected faces, not weld/contact proof, no candidate generation or GT endpoint insertion.',fit_calls=0,DEV_reads=0,heldout_reads=0,seconds=time.perf_counter()-started));xyz._mmap.close()
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
