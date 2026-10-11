"""XYZ-only exploratory circle carriers from an upright cross-section.

The largest observed planar support defines a base hypothesis, not gravity.
Projected carriers are proposals only: observed bottom proximity does not prove
continuous upright-to-bottom contact. No weld segmentation or labels are read.
"""
import argparse,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest,check_resources
from weld_pipeline.contract_curve_geometry import CircularArc
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
CONFIG=dict(seed=1031,section_half_width_mm=.75,base_exclusion_mm=3.,circle_residual_mm=1.,trials_per_round=2048,max_rounds=16,min_consensus_points=12,min_chain_points=3,max_radius_part_diagonals=1000.,no_GT_reads=True,no_STEP_reads=True,contact_proof=False,horizontal_membership_exclusion=True,multiple_observed_parallel_levels=True,parallel_normal_cosine_tolerance=1e-8,level_grouping_tolerance_mm=.01)
def run(key):
 assert key in {c['key'] for c in read(BASE/'coverage_cohort_protocol.json')['cases']};dest=BASE/'curved_upright_probe_v3'/key;dest.mkdir(parents=True,exist_ok=False);write(dest/'protocol.json',dict(CONFIG,source_sha256=digest(__file__)));start=time.perf_counter()
 folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);binding=read(folder/'input_freeze.json');faces=read(folder/'surfaces.json')
 for name in ('surfaces.json','surface_support.npz'):assert digest(folder/name)==binding['files'][name]
 path=BASE/'inputs'/key/'xyz_mm.npy';assert digest(path)==binding['input_sha256'];xyz=np.load(path,mmap_mode='r')
 with np.load(folder/'surface_support.npz') as support:
  sizes=[len(support[f['point_ids_key']]) if f.get('point_ids_key') in support and np.linalg.norm(f.get('normal',[0,0,0]))>0 else 0 for f in faces];fid=int(np.argmax(sizes));baseids=support[faces[fid]['point_ids_key']]
 face=faces[fid];n=np.asarray(face['normal'],float);n/=np.linalg.norm(n);origin=np.asarray(face['center'],float);origin=origin+(float(face['offset'])-origin@n)*n;basis=np.asarray(face['basis'],float);assert basis.shape==(3,2) and np.allclose(basis.T@basis,np.eye(2),atol=1e-6)
 height=(xyz-origin)@n;horizontal=np.zeros(len(xyz),bool);parallel=[]
 with np.load(folder/'surface_support.npz') as support:
  for i,f in enumerate(faces):
   raw=np.asarray(f.get('normal',[0,0,0]),float);norm=np.linalg.norm(raw)
   if norm and abs(raw@n/norm)>=1-1e-8 and f.get('point_ids_key') in support:
    ids=support[f['point_ids_key']];horizontal[ids]=True;parallel.append((float(np.asarray(f['center'])@n),i,len(ids)))
 groups=[]
 for offset,i,count in sorted(parallel):
  if groups and abs(offset-groups[-1]['offset'])<=.01:groups[-1]['members'].append((i,count))
  else:groups.append(dict(offset=offset,members=[(i,count)]))
 for g in groups:g['representative']=max(g['members'],key=lambda v:v[1])[0]
 off=height[(~horizontal)&(abs(height)>CONFIG['base_exclusion_mm'])];assert len(off)
 sign=1 if np.quantile(off,.99)>-np.quantile(off,.01) else -1;eligible=(~horizontal)&(sign*height>CONFIG['base_exclusion_mm']);level=float(np.median(sign*height[eligible]));selected=np.flatnonzero(eligible&(abs(sign*height-level)<=CONFIG['section_half_width_mm']));uv=(xyz[selected]-origin)@basis
 write(dest/'section_receipt.json',dict(horizontal_support_points_excluded=int(horizontal.sum()),eligible_off_plane_points=int(eligible.sum()),section_points=len(selected),parallel_plane_hypotheses=groups,scope='Observed plane support membership, not certified material/bottom identity'))
 uv=np.unique(uv,axis=0);assert len(uv)>=12;nn=cKDTree(uv).query(uv,k=2,workers=2)[0][:,1];spacing=float(np.median(nn[nn>1e-6]));gap=max(6.,3*spacing);diag=float(np.linalg.norm(np.ptp(uv,axis=0)));rng=np.random.default_rng(CONFIG['seed']);remaining=np.arange(len(uv));profiles=[];rounds=[];bottom=cKDTree(np.asarray(xyz[baseids]))
 for iteration in range(CONFIG['max_rounds']):
  if len(remaining)<CONFIG['min_consensus_points']:break
  points=uv[remaining];best=None
  for block in range(CONFIG['trials_per_round']//128):
   triples=rng.integers(0,len(points),size=(128,3));a,b,c=points[triples].transpose(1,0,2);A=2*np.stack((b-a,c-a),axis=1);rhs=np.stack((np.sum(b*b-a*a,axis=1),np.sum(c*c-a*a,axis=1)),axis=1);det=np.linalg.det(A);valid=abs(det)>1e-8
   centers=np.linalg.solve(A[valid],rhs[valid]);radii=np.linalg.norm(a[valid]-centers,axis=1);valid2=(radii>=spacing)&(radii<=CONFIG['max_radius_part_diagonals']*diag)&np.isfinite(radii)
   centers=centers[valid2];radii=radii[valid2]
   if not len(radii):continue
   residual=abs(np.linalg.norm(points[None,:,:]-centers[:,None,:],axis=2)-radii[:,None]);counts=(residual<=CONFIG['circle_residual_mm']).sum(1);j=int(np.argmax(counts));candidate=(int(counts[j]),centers[j],float(radii[j]))
   if best is None or candidate[0]>best[0]:best=candidate
  if best is None or best[0]<CONFIG['min_consensus_points']:break
  count,center,radius=best;mask=abs(np.linalg.norm(points-center,axis=1)-radius)<=CONFIG['circle_residual_mm'];fit=least_squares(lambda p:np.linalg.norm(points[mask]-p[:2],axis=1)-p[2],np.r_[center,radius],loss='soft_l1',f_scale=.5,max_nfev=200);center=fit.x[:2];radius=float(fit.x[2]);mask=abs(np.linalg.norm(points-center,axis=1)-radius)<=CONFIG['circle_residual_mm'];inliers=remaining[mask]
  if len(inliers)<CONFIG['min_consensus_points'] or radius<=0:break
  angles=np.mod(np.arctan2(uv[inliers,1]-center[1],uv[inliers,0]-center[0]),2*np.pi);order=np.argsort(angles);angles=angles[order];inliers=inliers[order];cut=int(np.argmax(np.diff(np.r_[angles,angles[0]+2*np.pi])))+1;angles=np.r_[angles[cut:],angles[:cut]+2*np.pi];inliers=np.r_[inliers[cut:],inliers[:cut]]
  splits=np.r_[0,np.flatnonzero(np.diff(angles)*radius>gap)+1,len(angles)];added=0;intervals=[]
  for lo,hi in zip(splits[:-1],splits[1:]):
   if hi-lo<CONFIG['min_chain_points']:continue
   theta=float(angles[lo]);sweep=float(angles[hi-1]-theta)
   if sweep<=0:continue
   intervals.append([float(radius*(angles[lo]-angles[0])),float(radius*(angles[hi-1]-angles[0]))])
   cc=origin+basis@center;direction=basis@np.array([np.cos(theta),np.sin(theta)]);arc=CircularArc(cc,radius,n,direction,sweep,False);samples=arc.point_at(np.linspace(0,arc.length_mm,65));distance=bottom.query(samples,workers=2)[0]
   profiles.append(dict(representation=dict(kind='arc',arc_parameters=arc.to_parameters()),length_mm=arc.length_mm,face_indices=[fid],support_normal=n.tolist(),support_intervals_mm=[[0.,arc.length_mm]],unobserved_gaps_mm=[],weld_type=None,material_pair=None,status='UNCLASSIFIED_UPRIGHT_SECTION_PROJECTION',proposal_branch='xyz_upright_section_circle_probe',observed_section_height_mm=sign*level,section_points=int(hi-lo),bottom_sample_nearest_max_mm=float(distance.max()),finite_contact_proven=False));added+=1
  # One finite analytic carrier retains all observed intervals and explicit voids.
  # Void evidence is not a welding gap label and does not establish contact.
  if intervals:
   theta=float(angles[0]);sweep=float(angles[-1]-theta)
   arc=CircularArc(origin+basis@center,radius,n,basis@np.array([np.cos(theta),np.sin(theta)]),sweep,False);gaps=[];last=0.
   for left,right in intervals:
    if left>last:gaps.append([last,left])
    last=max(last,right)
   if last<arc.length_mm:gaps.append([last,arc.length_mm])
   profiles.append(dict(representation=dict(kind='arc',arc_parameters=arc.to_parameters()),length_mm=arc.length_mm,face_indices=[fid],support_normal=n.tolist(),support_intervals_mm=intervals,unobserved_gaps_mm=gaps,weld_type=None,material_pair=None,status='UNCLASSIFIED_FINITE_UPRIGHT_SECTION_ENVELOPE',proposal_branch='xyz_upright_section_circle_envelope_probe',observed_section_height_mm=sign*level,section_points=len(inliers),finite_contact_proven=False));added+=1
  rounds.append(dict(round=iteration,inliers=len(inliers),radius_mm=radius,profiles=added));remaining=remaining[~mask];check_resources()
 # Every distinct observed parallel level remains a separate hypothesis.
 # This preserves multiple bottom heights and does not assert material identity.
 original_profiles=profiles;profiles=[]
 with np.load(folder/'surface_support.npz') as support:
  for group in groups:
   representative=group['representative'];plane=faces[representative];plane_offset=float(np.asarray(plane['center'])@n);tree=cKDTree(np.asarray(xyz[support[plane['point_ids_key']]]))
   for p in original_profiles:
    a=CircularArc.from_parameters(p['representation']['arc_parameters']);center=a.center_mm+(plane_offset-a.center_mm@n)*n;arc=CircularArc(center,a.radius_mm,n,a.start_direction,a.sweep_rad,a.is_closed);q=dict(p);q['representation']=dict(kind='arc',arc_parameters=arc.to_parameters());q['face_indices']=[representative];q['base_level_mm']=plane_offset;q['single_plane_bottom_identity_assumed']=False
    distance=tree.query(arc.point_at(np.linspace(0,arc.length_mm,65)),workers=2)[0];q['bottom_sample_nearest_max_mm']=float(distance.max());q['bottom_sample_fraction_within_3mm']=float(np.mean(distance<=3.));profiles.append(q)
 write(dest/'profiles.json',profiles);write(dest/'receipt.json',dict(status='XYZ_ONLY_PROPOSAL_PROBE_NOT_DEPLOYED',key=key,base_face=fid,base_support_points=sizes[fid],base_normal=n.tolist(),section_level_mm=sign*level,section_points=len(uv),parallel_base_level_hypotheses=len(groups),horizontal_support_points_excluded=int(horizontal.sum()),section_spacing_mm=spacing,observed_chain_gap_mm=gap,chain_gap_is_not_weld_segmentation_or_35mm_rule=True,rounds=rounds,profiles=len(profiles),input_sha256=digest(path),profiles_sha256=digest(dest/'profiles.json'),GT_reads=0,STEP_reads=0,fit_calls=0,geometric_circle_fitting=True,finite_contact_proven=False,seconds=time.perf_counter()-start));print('CURVED_UPRIGHT_PROBE',key,len(profiles),rounds,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
 with threadpool_limits(limits=2):run(a.key)
