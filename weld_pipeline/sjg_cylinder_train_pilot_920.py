"""Observe cylinders from BM79A XYZ, freeze first, then score TRAIN geometry."""
import argparse
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_observed_cylinders_919 import extract
from weld_pipeline.sjg_local_surface_curves_912 import serial,save_faces
from weld_pipeline.sjg_local_curve_supervision_915 import supervise,controls

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='BM79A_native_local')
    write(dest/'registered_protocol.json',dict(stage='INPUT_ONLY_CYLINDER_PLANE_TRAIN_PILOT',source_sha256=digest(__file__),
        geometry_sha256=digest(ROOT/'weld_pipeline/sjg_observed_cylinders_919.py'),start_utc=utc(),source_key=case['source_key'],
        xyz_only_generation=True,STEP_reads=0,original_eight_G=496,pilot_TRAIN_G=4,
        grid_mm=1.,normal_neighbors=32,rank2_ratio_minimum=.05,plane_eigen_ratio_maximum=.02,
        projected_column_quantization_mm=.1,column_min_distinct_heights=3,radial_tolerance_mm=.2,
        local_circle_trials_per_axis=4096,maximum_axis_families=8,optimizer_updates=0,full_contract_pass=False))
    try:
        write(dest/'event_controls.json',controls());path=OUT/case['input_file'];assert digest(path)==case['input_sha256']
        xyz=np.load(path,mmap_mode='r',allow_pickle=False);start=time.perf_counter()
        def progress(*v):check_resources();print('OBSERVED_CYLINDER',*v,flush=True)
        faces,profiles,cylinders,stats=extract(xyz,progress)
        save_faces(dest,faces);write(dest/'profiles.json',serial(profiles));write(dest/'cylinders.json',cylinders)
        freeze=dict(input_sha256=case['input_sha256'],generation_seconds=time.perf_counter()-start,stats=stats,
            files={f:digest(dest/f) for f in ('surfaces.json','surface_support.npz','profiles.json','cylinders.json')},
            generation_target_reads=0,STEP_reads=0)
        write(dest/'input_freeze.json',freeze);print('CYLINDER_PLANE_INPUTS_FROZEN',len(faces),len(cylinders),len(profiles),flush=True)
        # All local cylinder-plane outputs are immutable before TRAIN labels.
        assert digest(OUT/case['target_file'])==case['target_sha256'];target=read(OUT/case['target_file'])
        score=supervise(profiles,target);write(dest/'TRAIN_only_geometric_compatibility.json',score)
        write(dest/'receipt.json',dict(status='OBSERVED_CYLINDER_TRAIN_PILOT_COMPLETE',original_G=4,
            geometrically_supported=score['covered_children'],missing=score['missing'],faces=len(faces),cylinders=len(cylinders),
            arc_carriers=len(profiles),stats=stats,optimizer_updates=0,not_weld_accuracy=True,full_contract_pass=False,end_utc=utc()))
        print('CYLINDER_TRAIN_GEOMETRY',score['covered_children'],'/',4,'NOT_WELD_ACCURACY',flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
