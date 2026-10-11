"""Check finite extent recovery without changing plane fits or support masks."""
import argparse
import gc
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_finite_envelope_carriers_941 import envelope_profiles,controls
from weld_pipeline.sjg_local_curve_supervision_915 import supervise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    rows=[];cases={c['source_key']:c for c in read(OUT/'run98/local_cohort.json')['cases']}
    write(dest/'registered_protocol.json',dict(stage='FINITE_ENVELOPE_TRAIN_PILOT',sources=['4','10'],source_sha256=digest(__file__),
        geometry_sha256=digest(ROOT/'weld_pipeline/sjg_finite_envelope_carriers_941.py'),input_planes_and_meshes='run125',
        all_previous_profiles_retained=True,observed_support_masks_preserved=True,STEP_reads=0,optimizer_updates=0,
        full_contract_pass=False,start_utc=utc()))
    try:
        write(dest/'controls.json',controls());print('FINITE_ENVELOPE_CONTROLS_PASSED',flush=True)
        for key in ('4','10'):
            check_resources();case=cases[key];source=OUT/'run125'/('source_'+key);folder=dest/('source_'+key);folder.mkdir()
            binding=read(source/'input_freeze.json')
            for file in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(source/file)==binding['files'][file]
            assert digest(OUT/case['input_file'])==case['input_sha256'];xyz=np.load(OUT/case['input_file'],mmap_mode='r')
            faces=read(source/'surfaces.json');profiles=read(source/'profiles.json')
            with np.load(source/'surface_support.npz',allow_pickle=False) as z:
                for face in faces:face['point_ids']=z[face['point_ids_key']]
            new=envelope_profiles(xyz,faces,profiles,lambda *v:print(key,*v,flush=True));profiles.extend(new)
            write(folder/'profiles.json',profiles)
            write(folder/'input_freeze.json',dict(input_sha256=case['input_sha256'],parent_freeze_sha256=digest(source/'input_freeze.json'),
                profiles_sha256=digest(folder/'profiles.json'),new_envelope_profiles=len(new),target_reads_before_freeze=0,STEP_reads=0))
            score=supervise(profiles,read(OUT/case['target_file']));write(folder/'TRAIN_only_geometric_compatibility.json',score)
            row=dict(source=key,G=case['original_G'],supported=score['covered_children'],missing=score['missing'],new_envelopes=len(new));rows.append(row)
            write(folder/'receipt.json',dict(status='FINITE_ENVELOPE_TRAIN_PILOT_COMPLETE',**row,optimizer_updates=0,not_weld_accuracy=True,end_utc=utc()))
            print('FINITE_ENVELOPE_TRAIN_GEOMETRY',key,score['covered_children'],'/',case['original_G'],flush=True)
            del xyz,faces,profiles,new;gc.collect()
        write(dest/'receipt.json',dict(status='FINITE_ENVELOPE_TRAIN_PILOT_COMPLETE',sources=rows,optimizer_updates=0,
            not_weld_accuracy=True,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_sources=rows,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
