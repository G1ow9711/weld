"""Input-only arc envelopes on matching observed circles; never weld labels.

Original circle geometry is retained per seed. Observed gaps remain explicit
and are not asserted to be weld intervals, contacts, or 35mm splits.
"""
import argparse,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.contract_curve_geometry import CircularArc
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
def run(key):
 dest=BASE/'observed_arc_envelopes_v1'/key;dest.mkdir(parents=True,exist_ok=False);start=time.perf_counter();folder=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run185'/('source_'+key);binding=read(folder/'input_freeze.json');source=folder/'profiles.json';profiles=read(source);pairs=[(i,CircularArc.from_parameters(p['representation']['arc_parameters'])) for i,p in enumerate(profiles) if p['representation']['kind']=='arc'];centers=np.array([a.center_mm for _,a in pairs]);tree=cKDTree(centers);output=[]
 write(dest/'protocol.json',dict(center_tolerance_mm=.2,radius_tolerance_mm=.2,normal_cosine_tolerance=1e-8,geometry='Each original observed circle retained; compatible arc angular supports form bounded envelope with explicit gaps',no_coordinate_averaging=True,GT_reads=0,STEP_reads=0,not_material_contact_or_weld=True))
 for j,(pid,a) in enumerate(pairs):
  neighbours=tree.query_ball_point(a.center_mm,.2);u=a.start_direction;v=np.cross(a.plane_normal,u);angles=[];sources=[]
  for k in sorted(neighbours):
   oid,b=pairs[k]
   if abs(b.radius_mm-a.radius_mm)>.2 or abs(a.plane_normal@b.plane_normal)<1-1e-8:continue
   pts=b.point_at(np.linspace(0,b.length_mm,max(3,int(abs(b.sweep_rad)/.1)+1)));theta=np.mod(np.arctan2((pts-a.center_mm)@v,(pts-a.center_mm)@u),2*np.pi);angles.extend(theta.tolist());sources.append(oid)
  if len(sources)<2:continue
  angles=np.unique(angles);cut=int(np.argmax(np.diff(np.r_[angles,angles[0]+2*np.pi])))+1;ordered=np.r_[angles[cut:],angles[:cut]+2*np.pi];theta=ordered[0];sweep=float(ordered[-1]-theta)
  if sweep<=0:continue
  arc=CircularArc(a.center_mm,a.radius_mm,a.plane_normal,np.cos(theta)*u+np.sin(theta)*v,sweep,False)
  output.append(dict(representation=dict(kind='arc',arc_parameters=arc.to_parameters()),source_profiles=sources,seed_profile=pid,observed_support_source='Original finite arcs retained in source profiles; envelope does not establish gap contact',finite_contact_proven=False,weld_type=None,material_pair=None))
 write(dest/'profiles.json',output);write(dest/'receipt.json',dict(status='UNCLASSIFIED_OBSERVED_ARC_ENVELOPES',key=key,input_arc_profiles=len(pairs),profiles=len(output),source_profiles_sha256=digest(source),input_sha256=binding['input_sha256'],profiles_sha256=digest(dest/'profiles.json'),GT_reads=0,STEP_reads=0,fit_calls=0,seconds=time.perf_counter()-start));print(key,len(pairs),len(output),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
