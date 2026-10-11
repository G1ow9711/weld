"""Complete the line/arc boundary carrier interface on frozen TRAIN input."""
import argparse
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_observed_boundary_lines_927 import face_lines,controls
from weld_pipeline.sjg_local_curve_supervision_915 import supervise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    source=OUT/'run113';binding=read(source/'input_freeze.json')
    write(dest/'registered_protocol.json',dict(stage='LINE_AND_ARC_BOUNDARY_TRAIN_PILOT',source_sha256=digest(__file__),
        geometry_sha256=digest(ROOT/'weld_pipeline/sjg_observed_boundary_lines_927.py'),original_eight_G=496,pilot_TRAIN_G=4,
        optimizer_updates=0,full_contract_pass=False,start_utc=utc()))
    try:
        write(dest/'controls.json',controls());xyz=np.load(OUT/'run97/xyz_mm.npy',mmap_mode='r')
        assert digest(OUT/'run97/xyz_mm.npy')==binding['input_sha256']
        for file in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(source/file)==binding['files'][file]
        faces=read(source/'surfaces.json');profiles=read(source/'profiles.json');before=len(profiles)
        with np.load(source/'surface_support.npz',allow_pickle=False) as z:
            for i,face in enumerate(faces[:binding['raw_faces']]):
                check_resources();face['point_ids']=z[face['point_ids_key']];face['triangles']=z[face['triangles_key']]
                lines=face_lines(xyz,face)
                for line in lines:line.update(face_indices=[i],proposal_branch='observed_boundary_line');profiles.append(line)
                print('OBSERVED_BOUNDARY_LINES',i+1,binding['raw_faces'],'NEW',len(profiles)-before,flush=True)
        write(dest/'profiles.json',profiles)
        write(dest/'input_freeze.json',dict(input_sha256=binding['input_sha256'],parent_freeze_sha256=digest(source/'input_freeze.json'),
            profile_sha256=digest(dest/'profiles.json'),target_reads_before_freeze=0,STEP_reads=0))
        score=supervise(profiles,read(OUT/'run97/targets_local.json'));write(dest/'TRAIN_only_geometric_compatibility.json',score)
        write(dest/'receipt.json',dict(status='BOUNDARY_LINE_ARC_TRAIN_PILOT_COMPLETE',G=4,geometrically_supported=score['covered_children'],
            missing=score['missing'],new_lines=len(profiles)-before,optimizer_updates=0,not_weld_accuracy=True,full_contract_pass=False,end_utc=utc()))
        print('BOUNDARY_LINE_ARC_TRAIN_GEOMETRY',score['covered_children'],'/',4,'NOT_WELD_ACCURACY',flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
