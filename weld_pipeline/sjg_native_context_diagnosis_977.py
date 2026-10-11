"""Conditional TRAIN oracle centers diagnose core/context embedding failures.

Reference labels are used only after all nine model outputs have been frozen.
Nearest reference centers are a diagnostic upper bound, never free predictions.
"""
import argparse
import gc
import numpy as np
from scipy.spatial.distance import cdist
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources


def measured(values,labels,core):
    known=(labels>=0)&core;owners=np.unique(labels[known]);rows={}
    if not len(owners):return None
    centers=np.stack([values[known&(labels==owner)].mean(0) for owner in owners])
    for name,mask in [('core',known),('context',(~core)&np.isin(labels,owners))]:
        idx=np.flatnonzero(mask);expected=np.searchsorted(owners,labels[idx])
        distance=np.linalg.norm(values[idx]-centers[expected],axis=1)
        predicted=cdist(values[idx],centers,'sqeuclidean').argmin(1) if len(idx) else np.empty(0,np.int64)
        rows[name]=dict(points=len(idx),oracle_center_correct=int((predicted==expected).sum()),
            radius_le_02=int((distance<=.2).sum()),radius_le_04=int((distance<=.4).sum()),
            distance_sum=float(distance.sum()),squared_hinge_sum=float((np.maximum(distance-.2,0)**2).sum()))
        rows['multi_instance_'+name]={k:(v if len(owners)>1 else 0) for k,v in rows[name].items()}
    rows['single_instance_blocks']=int(len(owners)==1)
    rows['multi_instance_blocks']=int(len(owners)>1)
    rows['unknown_context_points']=int(((~core)&(labels<0)).sum())
    rows['known_context_owner_absent_from_core']=int(((~core)&(labels>=0)&~np.isin(labels,owners)).sum())
    drift=[]
    for owner,center in zip(owners,centers):
        mask=(~core)&(labels==owner)
        if mask.any():drift.append(float(np.linalg.norm(values[mask].mean(0)-center)))
    rows['center_drift_sum']=sum(drift);rows['center_drift_groups']=len(drift)
    if len(centers)>1:
        d=cdist(centers,centers);off=~np.eye(len(centers),dtype=bool)
        rows['separation_pairs']=int(off.sum());rows['center_pairs_below_1']=int((d[off]<1).sum())
    else:rows['separation_pairs']=rows['center_pairs_below_1']=0
    return rows


def accumulate(total,row):
    for key,value in row.items():
        if isinstance(value,dict):accumulate(total.setdefault(key,{}),value)
        else:total[key]=total.get(key,0)+value


def summarize(total):
    for subset in ('core','context','multi_instance_core','multi_instance_context'):
        row=total[subset];n=row['points']
        for count in ('oracle_center_correct','radius_le_02','radius_le_04'):
            row[count+'_fraction']=row[count]/n if n else None
        row['mean_distance_to_reference_core_center']=row['distance_sum']/n if n else None
    total['mean_core_context_center_drift']=total['center_drift_sum']/max(1,total['center_drift_groups'])
    return total


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    source=OUT/'run165';assert read(source/'receipt.json')['status']=='NATIVE_MATERIAL_SURFACE_FEATURES_FROZEN'
    cases={r['source_key']:r for r in read(source/'all_predictions_frozen.json')['cases']};assert len(cases)==9
    records=read(OUT/'run161/manifest.json')['records'];by_source={}
    for row in records:by_source.setdefault(row['source_key'],[]).append(row)
    write(dest/'registered_protocol.json',dict(stage='TRAIN_CORE_CONTEXT_ORACLE_CENTER_DIAGNOSIS',
        source_sha256=digest(__file__),features='run165',training_labels='run161',
        six_TRAIN_only=True,all_nine_model_outputs_already_frozen=True,
        reference_centers_NOT_FREE_PREDICTIONS=True,optimizer_updates=0,full_contract_pass=False,start_utc=utc()))
    rows=[]
    for key,records in by_source.items():
        entries=cases[key]['blocks'];assert len(entries)==len(records);totals={'face':{},'material':{}}
        for bi,(entry,record) in enumerate(zip(entries,records)):
            if bi%20==0:check_resources()
            feature=source/entry['file'];label=OUT/record['label_file']
            assert digest(feature)==entry['sha256'] and digest(label)==record['label_sha256']
            assert entry['input_block_sha256']==record['input_sha256']
            with np.load(feature) as p,np.load(label) as t:
                core=t['core'];assert int(core.sum())==len(p['original_point_ids'])==record['core_count']
                for name,field,target in [('face','embedding','instance'),('material','material_embedding','reference_volume')]:
                    value=np.r_[p[field],p['context_'+field]];assert len(value)==len(core)
                    result=measured(value,t[target],core)
                    if result is not None:accumulate(totals[name],result)
            if bi%100==0:print('CORE_CONTEXT_DIAGNOSIS',key,bi+1,'/',len(records),flush=True)
        result=dict(source_key=key,blocks=len(entries),face=summarize(totals['face']),material=summarize(totals['material']))
        rows.append(result);write(dest/f'source_{key}_diagnosis.json',result)
        print('CORE_CONTEXT_RESULT',key,'MATERIAL_CORE',result['material']['core']['oracle_center_correct_fraction'],
            'CONTEXT',result['material']['context']['oracle_center_correct_fraction'],
            'DRIFT',result['material']['mean_core_context_center_drift'],flush=True);gc.collect()
    write(dest/'receipt.json',dict(status='NATIVE_TRAIN_CORE_CONTEXT_DIAGNOSED',rows=rows,
        TRAIN_only=True,oracle_not_model_accuracy=True,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
