"""Recompute PTv3 features from explicit native local XYZ, no old point cache."""
import argparse
import builtins
import gc
import io
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_curve_label_inventory_895 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_local_cohort_902 import load_cohort,load_xyz
from weld_pipeline.sjg_ptv3_runtime_807 import SurfaceFeatureModel,infer_surface_features
from weld_pipeline.sjg_train_776 import check_resources


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    checkpoint=OUT/'run26/last.pt';cohort=load_cohort()
    write(dest/'registered_protocol.json',dict(stage='NATIVE_LOCAL_XYZ_PTV3_FEATURE_BASELINE',
        start_utc=utc(),source_sha256=digest(__file__),cohort_sha256=digest(OUT/'run98/local_cohort.json'),
        checkpoint_sha256=digest(checkpoint),checkpoint_training_cohort=[21,4,10,25,36,42],
        checkpoint_was_trained_on_legacy_world_points=True,
        coordinate_frame='workpiece_local',unit='mm',input_scale=1.,
        model_internal_centering_and_scale='per-block mean then divide by 100 in voxelize; original XYZ retained',
        runtime_sha256=digest(ROOT/'weld_pipeline/sjg_ptv3_runtime_807.py'),
        original_contract_G=496,new_BM79A_candidate_G=4,no_legacy_point_features_reused=True,
        input_channels=['x','y','z'],all_original_points_one_core_each=True,
        targets_STEP_or_label_PLY_not_opened=True,optimizer_updates=0,
        no_new_weld_accuracy_claim=True,full_contract_pass=False))
    records=[]
    try:
        check_resources();model=SurfaceFeatureModel().cuda()
        with checkpoint.open('rb') as stream:state=torch.load(stream,map_location='cpu')
        model.load_state_dict(state['model'],strict=True);del state;gc.collect();model.eval()
        for case in cohort['cases']:
            check_resources();xyz=load_xyz(case);key=case['source_key'];opens=[]
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            original_open,original_io_open=builtins.open,io.open
            def guarded(factory):
                def call(path,*a,**kw):
                    if isinstance(path,(str,bytes,Path)):
                        value=str(path).lower();opens.append(value)
                        if value.endswith(('.stp','.step','.xlsx','.ply')) or any(s in value for s in ('target','teacher','geometry.json','supervision')):
                            raise RuntimeError('Label/structure access during XYZ feature inference: '+str(path))
                    return factory(path,*a,**kw)
                return call
            try:
                builtins.open=guarded(original_open);io.open=guarded(original_io_open)
                features=infer_surface_features(model,xyz,
                    progress=lambda done,total,core:print('LOCAL_PTV3',key,'BLOCK',done,'/',total,'CORE',core,flush=True) if done==1 or done%20==0 or done==total else None,
                    resource_check=check_resources)
            finally:
                builtins.open=original_open;io.open=original_io_open
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            assert all(len(a)==len(xyz) for a in features.values())
            file=dest/f'source_{key}_features.npz';np.savez(file,**features)
            row=dict(source_key=key,role=case['role'],coordinate_frame='workpiece_local',unit='mm',
                original_points=len(xyz),input_file=case['input_file'],input_sha256=case['input_sha256'],
                feature_file=file.name,feature_sha256=digest(file),
                inference_seconds=seconds,gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                original_points_covered_once=int(np.count_nonzero(features['core_coverage']==1)),
                python_file_opens_during_inference=opens,python_label_or_STEP_reads=0,
                native_file_access_not_instrumented=True)
            records.append(row);write(dest/f'source_{key}_freeze.json',row)
            write(dest/'progress.json',dict(completed=len(records),total=len(cohort['cases']),sources=records))
            print('LOCAL_PTV3_FROZEN',key,len(xyz),'SECONDS',round(seconds,3),flush=True)
            del xyz,features;gc.collect();torch.cuda.empty_cache()
        write(dest/'all_predictions_frozen.json',dict(sources=records,label_reads_before_freeze=0,
            all_original_points_retained=True,coordinate_frame='workpiece_local',unit='mm',utc=utc()))
        write(dest/'receipt.json',dict(status='NATIVE_LOCAL_PTV3_FEATURE_BASELINE_COMPLETE',sources=records,
            original_eight_G=496,optimizer_updates=0,weld_accuracy_not_evaluated=True,
            full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_sources=records,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
