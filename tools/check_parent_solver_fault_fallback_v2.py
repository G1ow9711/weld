"""Fault injection on the real frozen 21 candidate problem; no label reads."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import collections
import numpy as np
from weld_pipeline import server_parent_set_optimization_v2 as opt
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 f=BASE/'source_21';parents=read(f/'parents.json')['parents'];resource=read(f/'resource_constraints.json');pc=read(f/'parent_children.json');records=read(f/'child_records.json');keys=opt.parent_keys(parents,pc,records);counts=np.load(BASE/'count_head_support_v2/21_training_counts.npy');u=counts[:,0]-counts[:,1];default=resource['default_parent_indices'];rr=collections.defaultdict(list)
 for i,rs in enumerate(resource['parent_resources']):
  for r in rs:rr[r].append(i)
 groups=[g for g in rr.values() if len(g)>1];reports=[]
 for mode in ['no_incumbent','solver_exception','nonfinite_incumbent','fractional_incumbent','infeasible_incumbent']:
  if mode=='solver_exception':kwargs=dict(side_effect=RuntimeError('Injected solver failure'))
  else:
   x={'no_incumbent':None,'nonfinite_incumbent':np.full(len(u),np.nan),'fractional_incumbent':np.full(len(u),.49),'infeasible_incumbent':np.ones(len(u))}[mode];kwargs=dict(return_value=SimpleNamespace(status=1,message='Injected '+mode,x=x))
  try:
   with patch.object(opt,'milp',**kwargs):chosen,reason,report=opt.solve(u,groups,resource['internally_conflicting_parents'],default,keys)
   row=dict(mode=mode,exception=False,returned_exact_default_indices=chosen==default,reason=reason)
  except Exception as exc:row=dict(mode=mode,exception=True,type=type(exc).__name__,message=str(exc),returned_exact_default_indices=False)
  reports.append(row);print(row,flush=True)
 assert all(r['returned_exact_default_indices'] for r in reports)
 write(BASE/'solver_fault_fallback_v2.json',dict(cases=reports,all_faults_return_complete_default=all(r['returned_exact_default_indices'] for r in reports),scope='Selection helper on real 21 inputs only; no output transaction or raw XYZ fallback claim',GT_reads=0,implementation_sha256=digest(ROOT/'weld_pipeline/server_parent_set_optimization_v2.py'),source_sha256=digest(__file__)))
if __name__=='__main__':main()
