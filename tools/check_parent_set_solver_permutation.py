"""Input-only permutation test of the actual frozen set-selection optimization."""
from pathlib import Path
import collections,hashlib,json,time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def physical_key(p):
 def seg(s,reverse):return dict(kind=s['kind'],weld_type=s['weld_type'],profile=s['profile'],side=s.get('predicted_side'),points=s['points'][::-1] if reverse else s['points'])
 variants=[dict(closed=p['is_closed'],gaps=p['gaps'],segments=[seg(s,False) for s in p['segments']]),dict(closed=p['is_closed'],gaps=p['gaps'][::-1],segments=[seg(s,True) for s in p['segments'][::-1]])]
 return min(json.dumps(v,sort_keys=True,separators=(',',':')) for v in variants)
def run(key):
 start=time.perf_counter();f=BASE/('source_'+key);d=BASE/'set_competition_support_v3'/('source_'+key);receipt=read(d/'receipt.json');res=read(f/'resource_constraints.json');assert digest(f/'resource_constraints.json')==receipt['resources_sha256'];parents=read(f/'parents.json')['parents'];n=len(parents);pred=np.load(d/'predicted_counts.npy');utility=pred[:,0]-pred[:,1];objective=utility.copy();default=res['default_parent_indices'];objective[default]+=1e-9;byresource=collections.defaultdict(list)
 for p,rows in enumerate(res['parent_resources']):
  for r in rows:byresource[r].append(p)
 groups=[v for v in byresource.values() if len(v)>1];pairs=np.load(f/'competition_v2/pairs.npy');prob=np.load(d/'competition_probability.npy');groups+=pairs[prob>.5].tolist();reference=receipt['selected_parent_indices'];reference_physical=collections.Counter(physical_key(parents[i]) for i in reference);reports=[]
 for label,order in [('identity',np.arange(n)),('permutation',np.random.default_rng(1031).permutation(n))]:
  inverse=np.empty(n,int);inverse[order]=np.arange(n);rr=[];cc=[]
  for ri,g in enumerate(groups):rr.extend([ri]*len(g));cc.extend(inverse[g])
  matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(groups),n)).tocsc();matrix.indices=matrix.indices.astype(np.int32);matrix.indptr=matrix.indptr.astype(np.int32);upper=np.ones(n);upper[inverse[res['internally_conflicting_parents']]]=0
  sol=milp(-objective[order],integrality=np.ones(n),bounds=Bounds(0,upper),constraints=LinearConstraint(matrix,-np.inf,1),options={'time_limit':60.,'mip_rel_gap':0.});assert sol.x is not None
  selected=order[np.flatnonzero(sol.x>.5)].tolist()
  if utility[selected].sum()+1e-8<utility[default].sum():selected=default
  physical=collections.Counter(physical_key(parents[i]) for i in selected);same=physical==reference_physical;row=dict(mode=label,solver_status=int(sol.status),selected=len(selected),same_selected_indices=set(selected)==set(reference),same_physical_parent_multiset=same,physical_parents_added=sum((physical-reference_physical).values()),physical_parents_removed=sum((reference_physical-physical).values()),utility=float(utility[selected].sum()),reference_utility=float(utility[reference].sum()),selected_indices=selected);reports.append(row);print(key,{k:v for k,v in row.items() if k!='selected_indices'},flush=True)
 return dict(source=key,controls=reports,seconds=time.perf_counter()-start,reference_receipt_sha256=digest(d/'receipt.json'))
if __name__=='__main__':
 reports=[run(k) for k in ['21','BM79A_native_local','10','25','36']];write(BASE/'set_solver_permutation_controls.json',dict(cases=reports,all_pass=all(c['same_physical_parent_multiset'] for r in reports for c in r['controls']),scope='Frozen candidate optimization only; no GT read and no upstream point-cloud claim',source_sha256=digest(__file__)))
