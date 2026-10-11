"""Input-only V3 pair graph/features: density, direction, rigid motion and order."""
import argparse,copy,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from weld_pipeline.server_parent_geometry_sampling import sample_segment
from weld_pipeline.server_parent_competition_v3 import build_pair_features
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def run(key):
 start=time.perf_counter();folder=BASE/('source_'+key);parents=read(folder/'parents.json')['parents'];meta=read(folder/'child_metadata.json');pc=read(folder/'parent_children.json');pairs,x=build_pair_features(parents,meta,pc);lookup={tuple(p):i for i,p in enumerate(pairs)};reports=[]
 for mode in ['density','reverse','rigid','cyclic','permutation']:
  other=copy.deepcopy(parents);ids=copy.deepcopy(pc);permutation=np.arange(len(parents))
  if mode=='permutation':
   permutation=np.random.default_rng(1031).permutation(len(other));other=[other[i] for i in permutation];ids=[ids[i] for i in permutation]
  else:
   for p,children in zip(other,ids):
    if mode=='reverse':p['segments'].reverse();p['gaps'].reverse();children.reverse()
    if mode=='cyclic' and p['is_closed']:p['segments']=p['segments'][1:]+p['segments'][:1];children[:]=children[1:]+children[:1]
    for s in p['segments']:
     if mode=='density':s['points']=sample_segment(s,7).tolist()
     if mode=='reverse':s['points']=s['points'][::-1]
     if mode=='rigid':
      R=Rotation.from_rotvec([.37,-.63,.28]).as_matrix();t=np.array([971.,-438.,723.]);s['points']=(np.asarray(s['points'])@R.T+t).tolist()
      if s['kind']=='arc':
       a=s['arc_parameters'];a['center_mm']=(R@a['center_mm']+t).tolist();a['plane_normal']=(R@a['plane_normal']).tolist();a['start_direction']=(R@a['start_direction']).tolist()
  pp,xx=build_pair_features(other,meta,ids);mapped=np.sort(permutation[pp],axis=1);matched=[(j,lookup[tuple(p)]) for j,p in enumerate(mapped) if tuple(p) in lookup];a,b=np.array(matched).T;delta=float(np.max(abs(xx[a]-x[b])));ok=len(pp)==len(pairs)==len(matched) and delta<=1e-4;row=dict(mode=mode,pairs_added=len(pp)-len(matched),pairs_removed=len(pairs)-len(matched),max_feature_error=delta,PASS=ok);reports.append(row);print(key,row,flush=True)
 write(BASE/('competition_geometry_v3_'+key+'.json'),dict(source=key,all_pass=all(r['PASS'] for r in reports),controls=reports,schema='PARENT_COMPETITION29_V3',scope='Pair features and retrieval only; no model fitted, no raw XYZ fullchain claim',seconds=time.perf_counter()-start,module_sha256=digest(ROOT/'weld_pipeline/server_parent_competition_v3.py'),sampler_sha256=digest(ROOT/'weld_pipeline/server_parent_geometry_sampling.py')))
 assert all(r['PASS'] for r in reports)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);run(ap.parse_args().key)
