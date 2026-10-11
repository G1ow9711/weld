"""TRAIN21 diagnostic: change only local clustering; freeze before reference score."""
import sys
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_native_instance_freeze_974 import main as freeze
from weld_pipeline.sjg_density_modes_978 import density_partition,controls
from weld_pipeline.sjg_native_instance_evaluate_975 import partition_metrics


def main():
    control=controls()
    freeze(partitioner=density_partition,method='density_modes',subset=['21'])
    run=sys.argv[sys.argv.index('--run')+1];dest=OUT/run
    receipt=read(dest/'receipt.json');assert receipt['status']=='NATIVE_SUBSET_INSTANCE_HYPOTHESES_FROZEN'
    row=receipt['sources'][0];assert row['source_key']=='21' and digest(dest/row['file'])==row['sha256']
    teacher=next(r for r in read(OUT/'run160/manifest.json')['rows'] if r['source_key']=='21')
    assert teacher['input_sha256']==row['input_sha256'] and digest(OUT/teacher['teacher_file'])==teacher['teacher_sha256']
    with np.load(dest/row['file']) as p,np.load(OUT/teacher['teacher_file']) as t:
        metrics=dict(face=partition_metrics(p['face_instance'],t['instance']),
            material=partition_metrics(p['material_instance'],t['reference_volume']))
    baseline=next(r for r in read(OUT/'run170/receipt.json')['rows'] if r['source_key']=='21')
    result=dict(status='TRAIN21_DENSITY_PARTITION_DIAGNOSTIC_COMPLETE',controls=control,metrics=metrics,
        baseline=baseline,change='local density modes instead of farthest covering; identical feature checkpoint, overlap thresholds and merge veto',
        selected_TRAIN_source='21',other_eight_structure_predictions_NOT_RUN=True,
        full_contract_G_unchanged=496,supplemental_BM_G=4,reference_is_conditional_offline=True,
        weld_accuracy_not_measured=True,optimizer_updates=0,full_contract_pass=False,end_utc=utc())
    write(dest/'diagnostic_receipt.json',result)
    print('TRAIN21_DENSITY_PARTITION_RESULT','FACE_F1',metrics['face']['B_cubed_F1'],
        'MATERIAL_F1',metrics['material']['B_cubed_F1'],flush=True)


if __name__=='__main__':main()
