"""Explicit equivalence proof for the V4 single-condition change on open pools."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 a=ROOT/'weld_pipeline/server_parent_competition_v3.py';b=ROOT/'weld_pipeline/server_parent_competition_v4.py'
 assert a.read_text().replace('schema V3','schema V4').replace("parents[a]['is_closed'] and parents[b]['is_closed']","parents[a]['is_closed'] or parents[b]['is_closed']")==b.read_text()
 for key in ['21','BM79A_native_local','10','36']:
  f=BASE/('source_'+key);control=BASE/('competition_geometry_v3_'+key+'.json');dest=BASE/('competition_geometry_v4_'+key+'.json')
  if dest.exists():continue
  if not control.exists() or not (f/'competition_v4/receipt.json').exists():print('PENDING',key,flush=True);continue
  prior=read(control);assert prior['all_pass'];assert prior['module_sha256']==digest(a);assert prior['sampler_sha256']==digest(ROOT/'weld_pipeline/server_parent_geometry_sampling.py');parents=read(f/'parents.json')['parents'];assert all(not p['is_closed'] for p in parents)
  r3=read(f/'competition_v3/receipt.json');r4=read(f/'competition_v4/receipt.json');assert r3['parents_sha256']==r4['parents_sha256']==digest(f/'parents.json')
  for name in ['pairs.npy','features.npy']:
   assert digest(f/'competition_v3'/name)==r3['files'][name];assert digest(f/'competition_v4'/name)==r4['files'][name];assert np.array_equal(np.load(f/'competition_v3'/name),np.load(f/'competition_v4'/name))
  out=dict(prior,schema='PARENT_COMPETITION29_V4',evidence_kind='Exact single-condition code equivalence for all-open pools plus independent V4 feature-byte equality; inherited transformation tests explicitly V3',module_sha256=digest(b),prior_control_sha256=digest(control),prior_module_sha256=digest(a),all_parents_open=True,independent_V4_features_equal=True,feature_receipt_sha256=digest(f/'competition_v4/receipt.json'),source_sha256=digest(__file__))
  write(dest,out);print('EQUIVALENT',key,flush=True)
if __name__=='__main__':main()
