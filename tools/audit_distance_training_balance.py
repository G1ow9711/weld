"""Measure actual TRAIN loss exposure; no model or threshold selection."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
BASE=ROOT/'server_rebuild/expansion_v1';MODEL=BASE/'reference_distance_comparison_v1'

def main():
 split=read(BASE/'reference_distance_auxiliary_protocol_v1/split.json');rows=[]
 for c in split['cases']:
  if c['role']!='train':continue
  key=c['key'];ids=np.load(MODEL/'crops'/f'{key}.npz')['training'];folder=BASE/'reference_distance_fields_v1'/key;lo=np.load(folder/'lower_mm.npy',mmap_mode='r')[ids];hi=np.load(folder/'upper_mm.npy',mmap_mode='r')[ids]
  for name,sl in [('uniform',slice(0,8)),('near_anchor',slice(8,16))]:
   a=lo[sl];b=hi[sl];counts=dict(near=int((b<=3).sum()),middle_or_boundary=int(((b>3)&(a<6)).sum()),far=int((a>=6).sum()));assert sum(counts.values())==a.size
   rows.append(dict(key=key,anchor_mode=name,observations=a.size,counts=counts,capped_points=int((a>=np.nextafter(np.float32(100),np.float32(-np.inf))).sum()),minimum_crop_near_fraction=float((b<=3).mean(1).min()),maximum_crop_near_fraction=float((b<=3).mean(1).max())))
 summary={}
 for mode in ('uniform','near_anchor','combined'):
  r=[x for x in rows if mode=='combined' or x['anchor_mode']==mode];n=sum(x['observations'] for x in r);counts={k:sum(x['counts'][k] for x in r) for k in ('near','middle_or_boundary','far')};summary[mode]=dict(observations=n,counts=counts,fractions={k:v/n for k,v in counts.items()},capped_fraction=sum(x['capped_points'] for x in r)/n)
 out=MODEL/'training_balance_audit.json';write(out,dict(status='COMPLETE',summary=summary,cases=rows,crop_receipt_sha256=digest(MODEL/'crop_receipt.json'),scope='Actual sampled TRAIN target exposure under unweighted per-point regression. Counts repeated observations, not distinct points; no protected DEV read. Does not alone prove causal explanation for poor performance.'));print(summary,flush=True)

if __name__=='__main__':main()
