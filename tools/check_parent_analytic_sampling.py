"""Real-pool resampling and reversal checks, plus gap/closed topology fixtures."""
import copy,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from weld_pipeline.server_parent_geometry_sampling import parent_geometry,sample_segment,endpoint_correspondence
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def equal_cloud(a,b):
 # Compare unordered quadrature multisets without coordinate-based canonicalization.
 from scipy.spatial import cKDTree
 return max(cKDTree(a).query(b)[0].max(),cKDTree(b).query(a)[0].max())
def main():
 start=time.perf_counter();reports=[];R=Rotation.from_rotvec([.37,-.63,.28]).as_matrix();t=np.array([971.,-438.,723.])
 for key in ['21','BM79A_native_local','10','25','36']:
  parents=read(BASE/('source_'+key)/'parents.json')['parents'];maximum=0.;closed_count=0
  for p in parents:
   cloud,ends,center=parent_geometry(p);q=copy.deepcopy(p)
   for s in q['segments']:s['points']=sample_segment(s,7).tolist()
   c,e,m=parent_geometry(q);maximum=max(maximum,float(np.max(abs(c-cloud))),float(np.max(abs(m-center))))
   q['segments'].reverse()
   for s in q['segments']:s['points']=s['points'][::-1]
   c,e,m=parent_geometry(q);maximum=max(maximum,equal_cloud(c,cloud),float(np.max(abs(m-center))),endpoint_correspondence(ends,e,p['is_closed']))
   q=copy.deepcopy(p)
   for s in q['segments']:
    s['points']=(np.asarray(s['points'])@R.T+t).tolist()
    if s['kind']=='arc':
     a=s['arc_parameters'];a['center_mm']=(R@a['center_mm']+t).tolist();a['plane_normal']=(R@a['plane_normal']).tolist();a['start_direction']=(R@a['start_direction']).tolist()
   c,e,m=parent_geometry(q);maximum=max(maximum,float(np.max(abs(c-(cloud@R.T+t)))),float(np.max(abs(m-(R@center+t)))))
   if p['is_closed']:
    closed_count+=1;q=copy.deepcopy(p);q['segments']=q['segments'][1:]+q['segments'][:1];c,e,m=parent_geometry(q);maximum=max(maximum,equal_cloud(c,cloud),endpoint_correspondence(ends,e,True),float(np.max(abs(m-center))))
  assert maximum<1e-7,(key,maximum)
  row=dict(source=key,parents=len(parents),closed_parent_controls=closed_count,max_error_mm=maximum,PASS=True);reports.append(row);print(row,flush=True)
 fixture={'segments':[dict(kind='line',points=[[0,0,0],[100,0,0]],length_mm=100),dict(kind='line',points=[[102,0,0],[102.01,0,0]],length_mm=.01)]}
 c,e,m=parent_geometry(fixture);assert not np.any((c[:,0]>100)&(c[:,0]<102));assert np.any(c[:,0]==102.01);assert len(c)==66
 write(BASE/'analytic_sampling_controls_v1.json',dict(PASS=True,cases=reports,gap_not_bridged=True,short_child_endpoints_preserved=True,scope='Analytic child sampling and parent centroid/endpoint correspondence, not whole pair feature or model invariance',full_circle_supported=False,seconds=time.perf_counter()-start,module_sha256=digest(ROOT/'weld_pipeline/server_parent_geometry_sampling.py'),source_sha256=digest(__file__)))
if __name__=='__main__':main()
