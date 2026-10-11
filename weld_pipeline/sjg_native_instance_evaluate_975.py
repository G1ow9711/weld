"""TRAIN-only conditional structure diagnosis after all nine predictions freeze."""
import argparse
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources


def partition_metrics(pred,truth):
    known=truth>=0;unknown=int((~known).sum());p=pred[known];t=truth[known]
    if not len(t):return dict(reference_points=0,unknown_reference_points=unknown)
    _,p=np.unique(p,return_inverse=True);owners,t=np.unique(t,return_inverse=True);classes=len(owners)
    codes,count=np.unique(p*classes+t,return_counts=True);r=codes//classes;c=codes%classes
    ps=np.bincount(p);ts=np.bincount(t);square=count.astype(np.float64)**2
    precision=float((square/ps[r]).sum()/len(t));recall=float((square/ts[c]).sum()/len(t))
    iou=count/(ps[r]+ts[c]-count);best=np.zeros(classes,np.float64);np.maximum.at(best,c,iou)
    return dict(reference_points=len(t),unknown_reference_points=unknown,reference_instances=classes,
        predicted_instances_on_known_points=len(ps),B_cubed_precision=precision,B_cubed_recall=recall,
        B_cubed_F1=2*precision*recall/max(1e-12,precision+recall),
        reference_instances_best_IoU_above_90pct=int((best>.9).sum()),
        reference_best_IoU=[dict(reference_id=int(i),IoU=float(v),points=int(n)) for i,v,n in zip(owners,best,ts)])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--prediction-run',default='run169');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);source=OUT/args.prediction_run
    receipt=read(source/'receipt.json');assert receipt['status']=='NATIVE_NINE_INSTANCE_HYPOTHESES_FROZEN' and len(receipt['sources'])==9
    frozen={r['source_key']:r for r in receipt['sources']}
    for row in frozen.values():assert digest(source/row['file'])==row['sha256']
    rows=[]
    for teacher in read(OUT/'run160/manifest.json')['rows']:
        check_resources();key=teacher['source_key'];row=frozen[key]
        assert row['input_sha256']==teacher['input_sha256']
        assert digest(OUT/teacher['teacher_file'])==teacher['teacher_sha256']
        with np.load(source/row['file']) as p,np.load(OUT/teacher['teacher_file']) as t:
            result=dict(source_key=key,points=teacher['points'],face=partition_metrics(p['face_instance'],t['instance']),
                material=partition_metrics(p['material_instance'],t['reference_volume']))
        rows.append(result);print('NATIVE_STRUCTURE_DIAGNOSIS',key,'FACE_BCUBED_F1',result['face']['B_cubed_F1'],
            'MATERIAL_BCUBED_F1',result['material']['B_cubed_F1'],flush=True);gc.collect()
    write(dest/'receipt.json',dict(status='NATIVE_TRAIN_INSTANCE_DIAGNOSIS_COMPLETE',rows=rows,
        prediction_run=args.prediction_run,all_nine_frozen_before_labels=True,
        conditional_offline_reference_not_customer_material_truth=True,TRAIN_only=True,
        multi_owner_contacts_not_evaluated=True,weld_accuracy_not_measured=True,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
