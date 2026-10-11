"""Recompute the unchanged pilot cohort with stable native local geometry."""
import argparse
import gc
import shutil
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_rank2_curve_pilot_926 import generate
from weld_pipeline.sjg_observed_boundary_lines_927 import controls as line_controls
from weld_pipeline.sjg_polygonal_arc_extension_923 import controls as arc_controls


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    cases=read(OUT/'run98/local_cohort.json')['cases'];rows=[]
    protocol=dict(stage='LOCAL_RANK2_LINE_ARC_COHORT',original_G=496,additional_TRAIN_candidate_G=4,
        sources=[c['source_key'] for c in cases],source_hashes={f:digest(ROOT/'weld_pipeline'/f) for f in
        ('sjg_rank2_cohort_929.py','sjg_rank2_curve_pilot_926.py','sjg_rank2_finite_planes_925.py',
         'sjg_polygonal_arcs_921.py','sjg_polygonal_arc_extension_923.py','sjg_observed_boundary_lines_927.py')},
        target_reads=0,STEP_reads=0,optimizer_updates=0,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        write(dest/'controls.json',dict(line=line_controls(),arc=arc_controls()))
        for case in sorted(cases,key=lambda c:c['source_key']!='BM79A_native_local'):
            check_resources();key=case['source_key'];folder=dest/('source_'+key);folder.mkdir()
            if key=='BM79A_native_local':
                # Reuse actual, completed input stages. The label diagnosis is
                # neither copied into the graph nor used to select carriers.
                before=read(OUT/'run113/input_freeze.json');after=read(OUT/'run114/input_freeze.json')
                assert before['input_sha256']==after['input_sha256']==case['input_sha256']
                for file in ('surfaces.json','surface_support.npz'):
                    assert digest(OUT/'run113'/file)==before['files'][file];shutil.copyfile(OUT/'run113'/file,folder/file)
                assert digest(OUT/'run114/profiles.json')==after['profile_sha256']
                shutil.copyfile(OUT/'run114/profiles.json',folder/'profiles.json')
                count=len(read(folder/'profiles.json'));new_lines=count-before['profiles']
                row=dict(before,raw_lines=before['raw_lines']+new_lines,profiles=count,
                    files={f:digest(folder/f) for f in ('surfaces.json','surface_support.npz','profiles.json')},
                    reused_input_stages=['run113','run114'],no_target_dependent_carrier_filter=True)
                write(folder/'input_freeze.json',row)
            else:
                def progress(*v):check_resources();print('LOCAL_RANK2',key,*v,flush=True)
                profiles,row=generate(case,folder,progress);del profiles
            rows.append(row);gc.collect();print('LOCAL_RANK2_FROZEN',key,'PROFILES',row['profiles'],flush=True)
        write(dest/'all_inputs_frozen.json',dict(sources=rows,target_reads=0,STEP_reads=0))
        write(dest/'receipt.json',dict(status='LOCAL_RANK2_TYPED_CURVES_FROZEN',sources=rows,original_G=496,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_sources=rows,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
