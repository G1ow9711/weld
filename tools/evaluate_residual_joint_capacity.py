"""TRAIN-only full-parent feasibility oracle under actual input resource rows."""
import argparse,collections,time
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.server_residual_resources import build_resources,hypothesis_resources
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/parent_hypotheses_v1'
def run(key):
 started=time.perf_counter();folder=BASE/('source_'+key);receipt=read(folder/'receipt.json')
 for name,sha in receipt['files'].items():assert digest(folder/name)==sha
 source=ROOT/'server_rebuild/best_baseline_restore/legacy_runtime'/('source_'+key)/'run379'/('test_'+key)
 manifest=read(source/'edge_manifest.json')
 for name in ['resources_offsets.npy','resources_indices.npy']:assert digest(source/name)==manifest['files'][name]['sha256']
 offsets=np.load(source/'resources_offsets.npy');indices=np.load(source/'resources_indices.npy');nodes=[indices[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
 a=read(folder/'atoms.json');originfolder=ROOT/'server_rebuild/child_residual_v1'/('source_'+key);origins=read(originfolder/'origins.json');assert digest(originfolder/'origins.json')==read(originfolder/'receipt.json')['files']['origins.json']
 groups,meta=build_resources(a['parents'],a['metadata'],a['aliases'],origins,nodes);paths=read(folder/'paths.json');paths=paths['native']+paths['added'];resources,invalid=hypothesis_resources(paths,groups)
 write(folder/'resource_constraints.json',dict(groups=groups,hypothesis_resources=resources,internally_conflicting_hypotheses=invalid,**meta))
 compat=read(folder/'full_parent_compatibility.json');target=ROOT/'server_rebuild/labels_v1'/key/'targets.json';assert digest(target)==compat['target_sha256'];truth=read(target);bad=set(invalid)
 variables=[(pi,ti) for pi,targets in enumerate(compat['compatible_targets']) if pi not in bad for ti in targets];width=len(variables);by_parent=collections.defaultdict(list);by_target=collections.defaultdict(list);by_resource=collections.defaultdict(list)
 for j,(pi,ti) in enumerate(variables):
  by_parent[pi].append(j);by_target[ti].append(j)
  for r in resources[pi]:by_resource[r].append(j)
 constraints=list(by_parent.values())+list(by_target.values())+list(by_resource.values());rr=[];cc=[]
 for ri,members in enumerate(constraints):rr.extend([ri]*len(members));cc.extend(members)
 matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(constraints),width)).tocsc();matrix.indices=matrix.indices.astype(np.int32);matrix.indptr=matrix.indptr.astype(np.int32)
 weights=np.array([truth['parents'][ti]['original_child_count'] for pi,ti in variables],float)
 if width:
  solution=milp(-weights,integrality=np.ones(width),bounds=Bounds(0,1),constraints=LinearConstraint(matrix,-np.inf,1.),options={'time_limit':60.,'mip_rel_gap':0.})
  assert solution.status==0 and solution.x is not None,'Joint oracle did not finish optimally: '+solution.message
  chosen=np.flatnonzero(solution.x>.5);selection=np.zeros(width);selection[chosen]=1.;assert np.max(matrix@selection)<=1.
 else:chosen=[]
 selected=[variables[j][0] for j in chosen];covered=[variables[j][1] for j in chosen];assert len(selected)==len(set(selected))
 allparents=read(folder/'hypotheses.json')['parents'];parents=[allparents[i] for i in selected];prediction=dict(parents=parents,invalid_predictions=[],total_prediction_children=sum(p['child_count'] for p in parents));score=score_native_parents(prediction,truth,[False]*len(truth['parents']))
 assert score['categories']['overall']['TP']==int(weights[chosen].sum()) if len(chosen) else score['categories']['overall']['TP']==0
 report=dict(source=key,joint_complete_TP=score['categories']['overall']['TP'],G=truth['original_children'],covered_original_parents=len(covered),missing_original_parents=sorted(set(range(len(truth['parents'])))-set(covered)),selected_hypothesis_indices=selected,internally_conflicting_hypotheses=len(invalid),variables=width,resource_groups=len(groups),diagnostic_only=True,GT_used_for_oracle_selection=True,not_an_extraction_result=True,fit_calls=0,DEV_reads=0,hypothesis_sha256=digest(folder/'hypotheses.json'),resource_sha256=digest(folder/'resource_constraints.json'),target_sha256=digest(target),seconds=time.perf_counter()-started)
 write(folder/'joint_capacity.json',report);print({k:v for k,v in report.items() if k!='selected_hypothesis_indices'},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);run(ap.parse_args().key)
