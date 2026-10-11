"""Fixed physical-distance/type strata of frozen material relation predictions."""
from pathlib import Path
import pickle
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/surface_material_relation_boundary_v2'
def main():
 reports=[]
 for key in ['10','25','36']:
  z=np.load(BASE/(key+'_pairs.npz'));x,y=z['features'],z['labels'];modelpath=BASE/('leave_'+key+'.pkl')
  with modelpath.open('rb') as f:model=pickle.load(f)
  pred=model.predict_proba(x)[:,list(model.classes_).index(1)]>=.5;distance=x[:,0]*1000;dot=x[:,3];rows=[]
  for lo,hi in [(0,10),(10,35),(35,100),(100,300),(300,1000),(1000,float('inf'))]:
   for name,geometry in [('all',np.ones(len(y),bool)),('parallel',dot>=.99),('orthogonal',dot<=.1),('oblique',(dot>.1)&(dot<.99))]:
    mask=(distance>=lo)&(distance<hi)&geometry;truth=y[mask].astype(bool);guess=pred[mask];tp=int((truth&guess).sum());fp=int((~truth&guess).sum());fn=int((truth&~guess).sum());tn=int((~truth&~guess).sum());rows.append(dict(distance_lo_mm=lo,distance_hi_mm=hi if np.isfinite(hi) else None,normal_relation=name,count=int(mask.sum()),TP=tp,FP=fp,FN=fn,TN=tn,false_merge=fp/(fp+tn) if fp+tn else None,same_recall=tp/(tp+fn) if tp+fn else None))
  reports.append(dict(key=key,rows=rows,prediction_model_sha256=digest(modelpath),data_sha256=digest(BASE/(key+'_pairs.npz'))));print(key,[r for r in rows if r['normal_relation']=='all'],flush=True)
 out=ROOT/'server_rebuild/surface_boundary_relation_error_scope_v1';out.mkdir(exist_ok=True);write(out/'receipt.json',dict(cases=reports,fit_calls=0,scope='Diagnostic bins only; 35 is a distance bin, not customer eave splitting or a decision threshold. No threshold tuning.',source_sha256=digest(__file__)))
if __name__=='__main__':main()
