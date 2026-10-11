"""Freeze pure-XYZ surface features on eight cases before offline evaluation."""
import argparse
import builtins
import gc
import io
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_ptv3_runtime_807 import SurfaceFeatureModel,infer_surface_features


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    checkpoint=OUT/'run26/last.pt'
    write(dest/'registered_protocol.json',dict(stage='PURE_XYZ_PTV3_FEATURE_FREEZE_THEN_OFFLINE_SCORE',start_utc=utc(),
        checkpoint_sha256=digest(checkpoint),sources=TRAIN+VALID+ARC,contract_G=496,
        source_sha256=digest(Path(__file__)),runtime_sha256=digest(ROOT/'weld_pipeline/sjg_ptv3_runtime_807.py'),
        full_weld_model=False,actual_optimizer_updates=0,contract_pass=False))
    try:
        assert read(OUT/'run26/receipt.json')['status']=='PTV3_SIX_CASE_REGISTERED_BUDGET_COMPLETE'
        check_resources();model=SurfaceFeatureModel().cuda()
        with checkpoint.open('rb') as h:state=torch.load(h,map_location='cuda')
        model.load_state_dict(state['model'],strict=True);del state;model.eval()
        inventory=read(OUT/'inventory.json');frozen=[]
        for source in TRAIN+VALID+ARC:
            check_resources();raw=ROOT/inventory[source]['raw']['original_pcd_path']
            xyz=read_xyz_pcd(raw)*1000.;opened=[];original_open=builtins.open;original_io_open=io.open
            def guarded(factory):
                def call(path,*args,**kwargs):
                    if isinstance(path,(str,bytes,Path)):
                        value=str(path).lower();opened.append(value)
                        if value.endswith(('.stp','.step')) or 'teacher' in value or 'geometry.json' in value or 'supervision' in value:
                            raise RuntimeError('Forbidden structural/label input during pure XYZ inference: '+str(path))
                    return factory(path,*args,**kwargs)
                return call
            start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            try:
                builtins.open=guarded(original_open);io.open=guarded(original_io_open)
                result=infer_surface_features(model,xyz,
                    progress=lambda done,total,core:print('PURE_XYZ_PTV3',source,'BLOCK',done,'/',total,'CORE_POINTS',core,flush=True) if done==1 or done%20==0 or done==total else None,
                    resource_check=check_resources)
            finally:
                builtins.open=original_open;io.open=original_io_open
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            file=dest/f'source_{source:02d}_features.npz';np.savez_compressed(file,**result)
            record=dict(source=source,raw_sha256=digest(raw),points=len(xyz),feature_file=file.name,
                feature_sha256=digest(file),inference_seconds=seconds,gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                python_file_opens_during_numeric_inference=opened,python_STEP_or_label_reads=0,
                native_file_access_not_instrumented=True)
            frozen.append(record);write(dest/f'source_{source:02d}_freeze.json',record)
            print('PURE_XYZ_PTV3_FROZEN',source,len(xyz),'seconds',round(seconds,3),flush=True)
            del xyz,result;gc.collect();torch.cuda.empty_cache()
        write(dest/'all_predictions_frozen.json',dict(sources=frozen,utc=utc(),teacher_reads_before_all_frozen=0))
        # This section is evaluation only, after all eight outputs are frozen.
        summary=[]
        for record in frozen:
            source=record['source'];teacher=OUT/'run22'/f'source_{source:02d}_teacher.npz'
            if not teacher.exists():
                summary.append(dict(source=source,status='NO_STRUCTURE_REFERENCE',points=record['points']));continue
            assert digest(dest/record['feature_file'])==record['feature_sha256']
            with np.load(dest/record['feature_file']) as pred,np.load(teacher) as truth:
                known=truth['instance_known'];dot=np.abs((pred['normal'][known]*truth['normal'][known]).sum(1))
                angle=np.rad2deg(np.arccos(np.clip(dot,0,1)))
                by=truth['boundary_target'];mask=by>=0;positive=by==1;bp=pred['boundary']>=.5
                summary.append(dict(source=source,scope='PLANAR_POINT_TEACHER_DIAGNOSTICS_ONLY',
                    original_points=record['points'],normal_known_points=int(known.sum()),
                    normal_unknown_points=int((~known).sum()),unoriented_normal_p50_deg=float(np.median(angle)),
                    unoriented_normal_p95_deg=float(np.quantile(angle,.95)),unoriented_normal_within3deg=int((angle<=3).sum()),
                    boundary=dict(tp=int((bp&mask&positive).sum()),fp=int((bp&mask&~positive).sum()),
                        fn=int((~bp&mask&positive).sum()),tn=int((~bp&mask&~positive).sum()),unknown=int((~mask).sum()))))
        write(dest/'offline_point_metrics.json',summary)
        write(dest/'receipt.json',dict(status='EIGHT_PURE_XYZ_FEATURES_FROZEN_AND_POINT_DIAGNOSTICS_COMPLETE',end_utc=utc(),
            sources=frozen,point_metrics=summary,actual_optimizer_updates=0,weld_extraction_accuracy_not_evaluated=True,
            instance_clustering_not_evaluated=True,contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
