"""Rigid and whole-reversal checks of frozen-carrier solver ordering keys."""
import copy
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from weld_pipeline.server_parent_set_optimization import parent_keys
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 reports=[];rotation=Rotation.from_rotvec([.37,-.63,.28]).as_matrix();shift=np.array([971.,-438.,723.])
 for key in ['21','BM79A_native_local','10','25','36']:
  f=BASE/('source_'+key);parents=read(f/'parents.json')['parents'];pc=read(f/'parent_children.json');records=read(f/'child_records.json');reference=parent_keys(parents,pc,records);controls=[]
  for mode in ['rigid','whole_parent_reverse']:
   moved=[];ids=[]
   for p,cc in zip(parents,pc):
    q=dict(p);q['segments']=[dict(s,points=np.asarray(s['points'])[[0,-1]].tolist()) for s in p['segments']]
    if mode=='rigid':
     for s in q['segments']:s['points']=(np.asarray(s['points'])@rotation.T+shift).tolist()
     ids.append(cc)
    else:
     q['segments']=q['segments'][::-1];q['gaps']=q['gaps'][::-1];ids.append(cc[::-1])
     for s in q['segments']:s['points']=s['points'][::-1]
    moved.append(q)
   actual=parent_keys(moved,ids,records);changed=[i for i,(a,b) in enumerate(zip(reference,actual)) if a!=b];controls.append(dict(mode=mode,changed_keys=changed,PASS=not changed));print(key,mode,len(changed),flush=True)
  reports.append(dict(source=key,candidates=len(parents),controls=controls))
 write(BASE/'set_solver_intrinsic_key_controls.json',dict(cases=reports,all_pass=all(c['PASS'] for r in reports for c in r['controls']),scope='Frozen carrier identities and scalar parameter intervals; not upstream re-extraction from rotated raw XYZ',implementation_sha256=digest(ROOT/'weld_pipeline/server_parent_set_optimization.py')))
if __name__=='__main__':main()
