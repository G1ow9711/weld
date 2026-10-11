"""Canonical ordering for a fixed physical candidate-set optimization problem.

Carrier ids and intrinsic intervals come from the frozen input graph. They are
solver tie keys, never learned identity features or claims of raw-cloud invariance.
No geometry is merged and no model score is changed.
"""
import json
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix

def parent_keys(parents,parent_children,child_records):
 keys=[]
 for p,ids in zip(parents,parent_children):
  alternatives=[]
  for reverse in (False,True):
   ss=p['segments'][::-1] if reverse else p['segments'];ii=ids[::-1] if reverse else ids;gaps=p['gaps'][::-1] if reverse else p['gaps'];ends=[np.asarray(s['points'])[[0,-1]][::-1] if reverse else np.asarray(s['points'])[[0,-1]] for s in ss];parts=[]
   for j,(s,i) in enumerate(zip(ss,ii)):
    r=child_records[i];child=(int(r['profile']),s['kind'],s['weld_type'],s.get('predicted_side'),r['projected_intervals_mm'])
    if j+1<len(ss):edge=(gaps[j],round(float(np.linalg.norm(ends[j][1]-ends[j+1][0])),6))
    elif p['is_closed']:edge=('SHARED',round(float(np.linalg.norm(ends[j][1]-ends[0][0])),6))
    else:edge=(None,None)
    parts.append((child,edge))
   for shift in (range(len(parts)) if p['is_closed'] else [0]):
    alternatives.append(json.dumps((bool(p['is_closed']),parts[shift:]+parts[:shift]),sort_keys=True,separators=(',',':')))
  keys.append(min(alternatives))
 return keys

def solve(utility,groups,invalid,default,keys,time_limit=60.):
 utility=np.asarray(utility,dtype=float)
 if utility.ndim!=1 or len(keys)!=len(utility):raise ValueError('Candidate contract mismatch')
 if len(default)!=len(set(default)) or any(i<0 or i>=len(utility) for i in default) or set(default)&set(invalid):raise ValueError('No valid default configuration')
 if not np.isfinite(utility).all():return list(default),'NONFINITE_UTILITY',dict(fallback=True)
 n=len(utility);default_set=set(default);order=np.array(sorted(range(n),key=lambda i:(keys[i],i not in default_set)),int);inverse=np.empty(n,int);inverse[order]=np.arange(n)
 canonical_groups=sorted({tuple(sorted(int(inverse[p]) for p in g)) for g in groups if len(g)>1});rr=[];cc=[]
 for ri,g in enumerate(canonical_groups):rr.extend([ri]*len(g));cc.extend(g)
 matrix=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(canonical_groups),n)).tocsc();matrix.indices=matrix.indices.astype(np.int32);matrix.indptr=matrix.indptr.astype(np.int32);upper=np.ones(n);upper[inverse[invalid]]=0
 objective=np.asarray(utility).copy();objective[default]+=1e-9
 try:
  result=milp(-objective[order],integrality=np.ones(n),bounds=Bounds(0,upper),constraints=LinearConstraint(matrix,-np.inf,1),options={'time_limit':time_limit,'mip_rel_gap':0.})
 except (RuntimeError,ValueError,AssertionError,TypeError,OverflowError) as exc:
  return list(default),'SOLVER_EXCEPTION',dict(fallback=True,exception_type=type(exc).__name__,message=str(exc))
 report=dict(status=int(result.status),message=result.message,canonical_columns=True,canonical_constraint_rows=True,input_constraint_rows=len(groups),unique_constraint_rows=len(canonical_groups),repeated_intrinsic_keys=n-len(set(keys)),frozen_carrier_identity_required=True,exact_lexicographic_optimality_claimed=False)
 if result.x is None:return list(default),'NO_FEASIBLE_INCUMBENT',report
 solution=np.asarray(result.x)
 if solution.shape!=(n,) or not np.isfinite(solution).all() or np.any(solution < -1e-6) or np.any(solution>1+1e-6) or np.any(abs(solution-np.rint(solution))>1e-6):return list(default),'INVALID_INCUMBENT',report
 chosen=np.flatnonzero(solution>.5);check=np.zeros(n);check[chosen]=1
 if len(canonical_groups) and np.max(matrix@check)>1:return list(default),'INFEASIBLE_INCUMBENT',report
 candidate=order[chosen].tolist()
 if set(candidate)&set(invalid):return list(default),'INTERNALLY_INVALID_INCUMBENT',report
 if np.asarray(utility)[candidate].sum()+1e-8<np.asarray(utility)[default].sum():return list(default),'DEFAULT_HAS_HIGHER_PREDICTED_UTILITY',report
 return candidate,None,report
