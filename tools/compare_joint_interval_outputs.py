"""Compare frozen interval predictions without using references to change output."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 rows=[]
 for key in ['new_cfc732e5f068','new_ceeae49d6777']:
  folders={name:BASE/folder/key for name,folder in [('original','bounded_raw_raw_full_output_v1'),('joint','joint_interval_residual_full_output_v1'),('zero','joint_interval_zero_full_output_v1'),('stable_zero','joint_interval_stable_zero_full_output_v1'),('stable_joint','joint_interval_stable_residual_full_output_v1')]}
  available={k:v for k,v in folders.items() if (v/'predictions.npz').exists()}
  arrays={k:np.load(v/'predictions.npz') for k,v in available.items()}
  for name in ('joint','zero','stable_zero','stable_joint'):
   if name not in arrays:continue
   old,new=arrays['original'],arrays[name];assert np.array_equal(old['selected_query_ids'],new['selected_query_ids'])
   a,b=old['interval_mm'],new['interval_mm'];valid=np.isfinite(b).all(1)&(b[:,0]<b[:,1]);diff=np.max(np.abs(a-b),axis=1);changed=diff>0
   report=dict(key=key,branch=name,queries=len(a),changed_queries=int(changed.sum()),valid_intervals=int(valid.sum()),changed_endpoint_max_mm_quantiles=np.quantile(diff[changed],[0,.5,.9,1]).tolist() if changed.any() else [],prediction_sha256=digest(available[name]/'predictions.npz'),original_sha256=digest(available['original']/'predictions.npz'))
   if (available[name]/'new_score.json').exists():report['whole_output_score']=read(available[name]/'new_score.json')['categories']['overall']
   rows.append(report)
  if 'joint' in arrays and 'zero' in arrays:
   assert np.array_equal(arrays['joint']['selected_query_ids'],arrays['zero']['selected_query_ids'])
   rows.append(dict(key=key,branch='learned_residual_vs_zero',changed_queries=int(np.any(arrays['joint']['interval_mm']!=arrays['zero']['interval_mm'],axis=1).sum())))
  for a in arrays.values():a.close()
 out=BASE/'joint_interval_output_comparison_v1';out.mkdir(exist_ok=True)
 write(out/'receipt.json',dict(rows=rows,scope='Same frozen query IDs; prediction differences and existing full-output scores only, no selection using references.',source_sha256=digest(__file__)))
 for r in rows:print(r,flush=True)
if __name__=='__main__':main()
