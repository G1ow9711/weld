"""Exact float32 representation conflicts in annotation supervision."""
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,read,write,digest
D=ROOT/'server_rebuild/connection_annotation_features_v1'
OUT=ROOT/'server_rebuild/connection_feature_collision_audit_v1'

def audit(x,y):
 # Numeric float32 equality regards -0 and +0 as equal, just as tree splits do.
 x=np.array(x,dtype='<f4',copy=True);x[x==0]=0
 rows=x.view(np.dtype((np.void,128))).reshape(-1)
 _,inverse=np.unique(rows,return_inverse=True)
 positive=np.bincount(inverse,weights=(y>0).astype(np.int64)).astype(np.int64)
 negative=np.bincount(inverse,weights=(y==0).astype(np.int64)).astype(np.int64)
 conflict=(positive>0)&(negative>0)
 return dict(rows=len(y),distinct_vectors=len(positive),conflicting_vectors=int(conflict.sum()),positive_rows_in_conflicts=int(positive[conflict].sum()),negative_rows_in_conflicts=int(negative[conflict].sum()),minimum_unweighted_binary_errors_from_exact_collisions=int(np.minimum(positive,negative).sum()),scope='Lower bound from identical numeric float32 vectors only, not total achievable tree error; unknown samples excluded.')

def main():
 OUT.mkdir(exist_ok=False);xs=[];ys=[];reports=[]
 for key in ('21','10','25','36'):
  r=read(D/key/'receipt.json');n=r['training_rows'];x=np.memmap(D/key/'features.f32',dtype='<f4',mode='r',shape=(n,32));y=np.memmap(D/key/'labels.i8',dtype='i1',mode='r',shape=(n,))
  assert digest(D/key/'features.f32')==r['files']['features.f32'] and digest(D/key/'labels.i8')==r['files']['labels.i8']
  report=dict(source=key,**audit(x,y));reports.append(report);print(report,flush=True);xs.append(x);ys.append(y)
 pooled=audit(np.concatenate(xs),np.concatenate(ys));write(OUT/'receipt.json',dict(status='COMPLETE',cases=reports,pooled=pooled,fit_calls=0,protected_DEV_reads=0));print('POOLED',pooled,flush=True)

if __name__=='__main__':main()
