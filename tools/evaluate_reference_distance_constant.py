"""Fit a constant using TRAIN targets only; evaluate on identical input crops."""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';OUT=BASE/'reference_distance_comparison_v1'


def main():
    split=read(BASE/'reference_distance_auxiliary_protocol_v1/split.json');lower=[];upper=[]
    for case in split['cases']:
        if case['role']!='train':continue
        key=case['key'];ids=np.load(OUT/'crops'/f'{key}.npz')['training'];p=BASE/'reference_distance_fields_v1'/key
        lower.append(np.log1p(np.load(p/'lower_mm.npy',mmap_mode='r')[ids]).ravel());upper.append(np.log1p(np.load(p/'upper_mm.npy',mmap_mode='r')[ids]).ravel())
    low=np.concatenate(lower).astype(float);high=np.concatenate(upper).astype(float)
    def objective(x):return float(np.mean(np.maximum(low-x,0)**2+np.maximum(x-high,0)**2))
    fit=minimize_scalar(objective,bounds=(0,np.log1p(100)),method='bounded',options={'xatol':1e-10});assert fit.success
    pred=float(fit.x);mm=float(np.expm1(pred));rows=[]
    for case in split['cases']:
        key=case['key'];ids=np.load(OUT/'crops'/f'{key}.npz')['evaluation'];p=BASE/'reference_distance_fields_v1'/key
        lo=np.load(p/'lower_mm.npy',mmap_mode='r')[ids];hi=np.load(p/'upper_mm.npy',mmap_mode='r')[ids];near=hi<=3;far=lo>=6
        rows.append(dict(key=key,role=case['role'],mean_bounded_log_loss=float(np.mean(np.maximum(np.log1p(lo)-pred,0)**2+np.maximum(pred-np.log1p(hi),0)**2)),mean_interval_distance_mm=float(np.mean(np.maximum(lo-mm,0)+np.maximum(mm-hi,0))),near_count=int(near.sum()),near_hits=int(near.sum()) if mm<=3 else 0,far_count=int(far.sum()),far_false=int(far.sum()) if mm<=3 else 0))
    summary={}
    for role in ('train','development_validation'):
        rr=[r for r in rows if r['role']==role];n=sum(r['near_count'] for r in rr);f=sum(r['far_count'] for r in rr)
        summary[role]=dict(families=len(rr),macro_bounded_log_loss=float(np.mean([r['mean_bounded_log_loss'] for r in rr])),macro_interval_distance_mm=float(np.mean([r['mean_interval_distance_mm'] for r in rr])),near_recall=sum(r['near_hits'] for r in rr)/n if n else None,far_false_near_rate=sum(r['far_false'] for r in rr)/f if f else None)
    write(OUT/'constant_baseline.json',dict(status='COMPLETE',constant_log_distance=pred,constant_mm=mm,TRAIN_objective=float(fit.fun),training_point_observations=len(low),summary=summary,cases=rows,script_sha256=digest(__file__),crop_receipt_sha256=digest(OUT/'crop_receipt.json'),scope='One constant fitted only to same TRAIN crops, evaluated on identical uniform crops; no validation fitting, no geometry inference'))
    print(mm,summary,flush=True)

if __name__=='__main__':main()
