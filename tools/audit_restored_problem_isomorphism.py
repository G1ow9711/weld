"""Audit complete stored path objectives and constraint families after reordering."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/best_baseline_restore'
def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(root):
 case=root/'legacy_runtime/source_10';folder=case/'run379/test_10';g=read(folder/'selected_graph.json');nodes=g['nodes']
 profiles=read_profile_geometry(case/'run185/source_10/profiles.json',retain_fields=('support_intervals_mm','unobserved_gaps_mm'))
 with np.load(case/'run285/source_10_geometry.npz') as z:offsets=z['offsets'];spans=z['spans']
 keys=[]
 for n in nodes:
  f=n['fragment'];pid=int(np.searchsorted(offsets,f,side='right')-1)
  keys.append(sha(dict(profile=profiles[pid],fragment_span=spans[f].tolist(),node={k:v for k,v in n.items() if k not in ('fragment','object_row')})))
 assert len(set(keys))==len(keys),'Physical mapping ambiguous; do not claim isomorphism'
 order=np.argsort(keys);rank=np.empty(len(keys),np.int64);rank[order]=np.arange(len(keys))
 manifest=read(folder/'edge_manifest.json');count=manifest['edge_count']
 for name,v in manifest['files'].items():assert digest(folder/name)==v['sha256']
 edge=np.memmap(folder/'edges.i32',dtype=np.int32,mode='r',shape=(count,2));mapped=rank[edge];eo=np.lexsort((mapped[:,1],mapped[:,0]));mapped=mapped[eo]
 legal=np.memmap(folder/'legal.u8',dtype=np.uint8,mode='r',shape=(count,3));p=np.memmap(folder/'probabilities.f64',dtype=np.float64,mode='r',shape=(count,4))
 off=np.load(folder/'resources_offsets.npy');ids=np.load(folder/'resources_indices.npy');resources=collections.Counter(tuple(sorted(rank[ids[a:b]].tolist())) for a,b in zip(off[:-1],off[1:]))
 signatures=dict(nodes=sha(sorted(keys)),edges=hashlib.sha256(mapped.tobytes()).hexdigest(),legal_states=hashlib.sha256(legal[eo].tobytes()).hexdigest(),all_probabilities=hashlib.sha256(p[eo].tobytes()).hexdigest(),resource_rows=sha(sorted((list(k),v) for k,v in resources.items())))
 return dict(signatures=signatures,nodes=len(keys),edges=count,resource_rows=len(off)-1,resource_members=len(ids),optional_nodes=not g['selection']['all_positive_nodes_fixed_selected'],integer_bounds=[0,1],constraint_families=['incoming edge sum <= node','outgoing edge sum <= node','resource member sum <= 1'],zero_states_retained=g['selection']['zero_states_retained'],coefficient_formula_source_sha256=digest(ROOT/'weld_pipeline/sjg_sparse_path_solver_1168.py'))
if __name__=='__main__':
 a=load(BASE);b=load(BASE/'historical_cache_order');equal=a==b
 write(BASE/'historical_cache_order/problem_isomorphism_audit.json',dict(original=a,reordered=b,stored_problem_isomorphic=equal,GT_reads=0,fit_calls=0,floating_solver_optimal_status_not_rational_global_certificate=True,scope='Physical bijection plus all stored edge probabilities/legal states, resource row multisets, common exact assembly formulas and bounds; resource display names are not solver constraints'))
 print('STORED_PROBLEM_ISOMORPHIC',equal,flush=True)
