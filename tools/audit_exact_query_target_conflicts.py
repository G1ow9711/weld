"""TRAIN-only identifiability test of deterministic 457-feature endpoint heads."""
from pathlib import Path
import hashlib,collections
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';MODEL=BASE/'positive_query_head_v1/joint_reference_v1'
def main():
 receipt=read(MODEL/'receipt.json');groups={};total=0;bindings={}
 for key in receipt['training_keys']:
  folder=Path(receipt['binding_paths'][key]);assert digest(folder/'receipt.json')==receipt['bindings'][key];bindings[key]=receipt['bindings'][key];x=np.load(folder/'features.npy',mmap_mode='r');z=np.load(folder/'targets.npz');records=read(folder/'records.json');src=(ROOT/'server_rebuild/best_baseline_restore/legacy_runtime' if key in ('21','BM79A_native_local','10','25','36') else BASE/'frozen_features/legacy_runtime')/('source_'+key);profiles=read(src/'run185'/('source_'+key)/'profiles.json')
  for i,r in enumerate(records):
   p=profiles[r['profile']]
   if p['representation']['kind']!='line':continue
   ends=np.asarray(p['representation']['endpoints_mm']);length=float(np.linalg.norm(ends[1]-ends[0]));y=z['interval_ratios'][i];role=int(z['weld_type'][i]);token=hashlib.sha256(x[i].tobytes()).hexdigest();lo=y-3./length;hi=y+3./length;member=dict(key=key,row=i,owners=r['owners'],target_ratio=y.tolist(),carrier_length_mm=length,role=role)
   if token not in groups:groups[token]=dict(lower=lo,upper=hi,roles={role},members=[member],feature_bytes=x[i].tobytes())
   else:
    g=groups[token];assert g['feature_bytes']==x[i].tobytes();g['lower']=np.maximum(g['lower'],lo);g['upper']=np.minimum(g['upper'],hi);g['roles'].add(role);g['members'].append(member)
   total+=1
  x._mmap.close()
 duplicates=[];conflicts=[]
 for sha,g in groups.items():
  if len(g['members'])<2:continue
  item=dict(feature_sha256=sha,rows=len(g['members']),members=g['members'],generous_3mm_ratio_intersection_lower=g['lower'].tolist(),generous_3mm_ratio_intersection_upper=g['upper'].tolist(),role_conflict=len(g['roles'])>1,interval_conflict=bool(np.any(g['lower']>g['upper'])))
  duplicates.append(item)
  if item['role_conflict'] or item['interval_conflict']:conflicts.append(item)
 out=BASE/'exact_query_target_conflicts_v1';out.mkdir(exist_ok=False);write(out/'conflicts.json',conflicts);write(out/'receipt.json',dict(status='TRAIN_IDENTIFIABILITY_AUDIT_COMPLETE',line_query_rows=total,unique_exact_features=len(groups),duplicate_groups=len(duplicates),rows_in_duplicate_groups=sum(r['rows'] for r in duplicates),incompatible_exact_feature_groups=len(conflicts),rows_in_conflicting_groups=sum(r['rows'] for r in conflicts),bindings=bindings,scope='Exact float32 feature equality only; 3mm along-line tolerance is a generous necessary endpoint condition, not original D3 scoring. No claim about near-feature ambiguity or model accuracy.',heldout_reads=0,DEV_reads=0,fit_calls=0,source_sha256=digest(__file__)));print(read(out/'receipt.json'),flush=True)
if __name__=='__main__':main()
