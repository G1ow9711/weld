"""Test extracted production features under physical transformations and ordering."""
import copy,pickle,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.server_parent_competition_v2 import build_pair_features
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 start=time.perf_counter();reports=[]
 with (BASE/'competition_head_v2/model.pkl').open('rb') as f:model=pickle.load(f)
 for key in ['10','36']:
  folder=BASE/('source_'+key);parents=read(folder/'parents.json')['parents'];meta=read(folder/'child_metadata.json');pc=read(folder/'parent_children.json');pairs,x=build_pair_features(parents,meta,pc);assert np.array_equal(pairs,np.load(folder/'competition_v2/pairs.npy'));assert np.array_equal(x,np.load(folder/'competition_v2/features.npy'));baseprob=model.predict_proba(x)[:,1];base={tuple(p):i for i,p in enumerate(pairs)};controls=[]
  for mode in ['rigid','whole_parent_reverse','candidate_permutation']:
   other=copy.deepcopy(parents);ids=copy.deepcopy(pc);permutation=np.arange(len(parents))
   if mode=='rigid':
    rotation=Rotation.from_rotvec([.37,-.63,.28]).as_matrix();offset=np.array([971.,-438.,723.])
    for p in other:
     for seg in p['segments']:seg['points']=(np.asarray(seg['points'])@rotation.T+offset).tolist()
   elif mode=='whole_parent_reverse':
    for p,i in zip(other,ids):
     p['segments']=p['segments'][::-1];p['gaps']=p['gaps'][::-1];i.reverse()
     for seg in p['segments']:seg['points']=seg['points'][::-1]
   else:
    permutation=np.random.default_rng(1031).permutation(len(other));other=[other[i] for i in permutation];ids=[ids[i] for i in permutation]
   pp,xx=build_pair_features(other,meta,ids);mapped=np.sort(permutation[pp],axis=1);common=[(j,base[tuple(p)]) for j,p in enumerate(mapped) if tuple(p) in base];a,b=np.array(common).T;pred=model.predict_proba(xx)[:,1];delta=float(np.max(abs(xx[a]-x[b]))) if len(a) else None;flip=int(np.sum((pred[a]>.5)!=(baseprob[b]>.5)));row=dict(mode=mode,pairs_added=len(pp)-len(common),pairs_removed=len(pairs)-len(common),max_feature_difference=delta,max_probability_difference=float(np.max(abs(pred[a]-baseprob[b]))),classification_changes=flip,pass_with_1e_4_feature_tolerance=len(pp)==len(pairs)==len(common) and delta<=1e-4 and flip==0);controls.append(row);print(key,row,flush=True)
  reports.append(dict(source=key,baseline_feature_bytes_equal=True,controls=controls))
 write(BASE/'competition_geometry_controls_v2_remaining.json',dict(cases=reports,all_pass=all(c['pass_with_1e_4_feature_tolerance'] for r in reports for c in r['controls']),scope='Candidate pair feature builder and competition model only; not upstream XYZ pipeline',seconds=time.perf_counter()-start,source_sha256=digest(__file__),feature_module_sha256=digest(ROOT/'weld_pipeline/server_parent_competition_v2.py')))
if __name__=='__main__':main()
