"""Native TRAIN surface-head baseline, explicitly separate from weld accuracy."""
import argparse
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--feature-run',default='run106');args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    assert read(OUT/'run160/receipt.json')['status']=='NATIVE_TRAIN_PLANAR_TEACHERS_READY'
    features=OUT/args.feature_run
    if args.feature_run!='run106':
        frozen=read(features/'receipt.json');assert frozen['status']=='NATIVE_MATERIAL_POINT_FEATURES_READY' and len(frozen['sources'])==9
        # Verify every free case before any reference label content is read.
        for row in frozen['sources']:assert digest(features/row['feature_file'])==row['feature_sha256']
    rows=[];totals={k:0 for k in ('TP','FP','FN','TN')}
    for teacher in read(OUT/'run160/manifest.json')['rows']:
        check_resources();key=teacher['source_key'];f=read(features/f'source_{key}_freeze.json')
        assert teacher['input_sha256']==f['input_sha256']
        assert digest(OUT/teacher['teacher_file'])==teacher['teacher_sha256']
        assert digest(features/f['feature_file'])==f['feature_sha256']
        with np.load(OUT/teacher['teacher_file']) as z:labels={k:z[k] for k in ('instance','normal','boundary_target')}
        with np.load(features/f['feature_file']) as z:normal=z['normal'];boundary=z['boundary']
        assert len(normal)==len(labels['instance'])==teacher['points']
        known=labels['boundary_target']>=0;positive=labels['boundary_target']==1;prediction=boundary>=.5
        counts=dict(TP=int((known&positive&prediction).sum()),FP=int((known&~positive&prediction).sum()),
            FN=int((known&positive&~prediction).sum()),TN=int((known&~positive&~prediction).sum()))
        normal_known=labels['instance']>=0
        angle=np.rad2deg(np.arccos(np.clip(np.abs((normal[normal_known]*labels['normal'][normal_known]).sum(1)),0,1)))
        counts['precision']=counts['TP']/max(1,counts['TP']+counts['FP']);counts['recall']=counts['TP']/max(1,counts['TP']+counts['FN'])
        row=dict(source_key=key,points=teacher['points'],boundary=counts,
            unoriented_normal_quantiles_deg=np.quantile(angle,[.5,.95,.99]).tolist(),
            learned_finite_face_instances_not_evaluated=True,material_head_absent=args.feature_run=='run106',material_partition_not_evaluated=True)
        rows.append(row)
        for k in totals:totals[k]+=counts[k]
        print('NATIVE_SURFACE_BASELINE',key,'BOUNDARY_P_R',counts['precision'],counts['recall'],'NORMAL_P95',row['unoriented_normal_quantiles_deg'][1],flush=True)
        del labels,normal,boundary,angle;gc.collect()
    totals.update(precision=totals['TP']/max(1,totals['TP']+totals['FP']),recall=totals['TP']/max(1,totals['TP']+totals['FN']))
    write(dest/'receipt.json',dict(status='NATIVE_TRAIN_SURFACE_BASELINE_MEASURED' if args.feature_run=='run106' else 'NATIVE_TRAIN_SURFACE_EVALUATED',rows=rows,boundary_total=totals,
        fixed_checkpoint='run26' if args.feature_run=='run106' else 'native_structural_model_bound_by_feature_run',features=args.feature_run,TRAIN_only=True,point_metrics_not_weld_extraction_accuracy=True,
        directed_normal_E16_not_evaluated=True,full_contract_pass=False,end_utc=utc()))

if __name__=='__main__':main()
