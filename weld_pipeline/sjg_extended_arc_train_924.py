"""Input-only boundary extension pilot, frozen before TRAIN diagnosis."""
import argparse
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_polygonal_arc_extension_923 import face_arcs,controls
from weld_pipeline.sjg_local_curve_supervision_915 import supervise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    source=OUT/'run108/source_BM79A_native_local';binding=read(source/'input_freeze.json')
    write(dest/'registered_protocol.json',dict(stage='CONNECTED_BOUNDARY_EXTENSION_TRAIN_PILOT',
        source_sha256=digest(__file__),geometry_sha256=digest(ROOT/'weld_pipeline/sjg_polygonal_arc_extension_923.py'),
        original_eight_G=496,pilot_TRAIN_G=4,STEP_reads=0,optimizer_updates=0,full_contract_pass=False,start_utc=utc()))
    try:
        write(dest/'controls.json',controls());print('EXTENSION_CONTROLS_PASSED',flush=True)
        assert digest(OUT/'run97/xyz_mm.npy')==binding['input_sha256'];xyz=np.load(OUT/'run97/xyz_mm.npy',mmap_mode='r')
        for f in ('surfaces.json','surface_support.npz'):assert digest(source/f)==binding['files'][f]
        previous=read(OUT/'run110/input_freeze.json')
        assert digest(OUT/'run110/profiles.json')==previous['profile_sha256']
        profiles=read(OUT/'run110/profiles.json');before=len(profiles);faces=read(source/'surfaces.json');stats=[]
        with np.load(source/'surface_support.npz',allow_pickle=False) as z:
            for i,face in enumerate(faces):
                check_resources();face['point_ids']=z[face['point_ids_key']];face['triangles']=z[face['triangles_key']]
                arcs,row=face_arcs(xyz,face)
                for arc in arcs:arc.update(face_indices=[i],proposal_branch='connected_boundary_extension');profiles.append(arc)
                stats.append(dict(face=i,**row));del face['point_ids'],face['triangles']
                if i%20==0:print('CONNECTED_BOUNDARY',i+1,'/',len(faces),'NEW_ARCS',len(profiles)-before,flush=True)
        write(dest/'profiles.json',profiles);write(dest/'boundary_stats.json',stats)
        write(dest/'input_freeze.json',dict(input_sha256=binding['input_sha256'],parent_freeze_sha256=digest(source/'input_freeze.json'),
            profile_sha256=digest(dest/'profiles.json'),new_arcs=len(profiles)-before,target_reads_before_freeze=0,STEP_reads=0))
        score=supervise(profiles,read(OUT/'run97/targets_local.json'));write(dest/'TRAIN_only_geometric_compatibility.json',score)
        write(dest/'receipt.json',dict(status='CONNECTED_BOUNDARY_TRAIN_PILOT_COMPLETE',G=4,geometrically_supported=score['covered_children'],
            missing=score['missing'],new_arcs=len(profiles)-before,optimizer_updates=0,not_weld_accuracy=True,full_contract_pass=False,end_utc=utc()))
        print('CONNECTED_BOUNDARY_TRAIN_GEOMETRY',score['covered_children'],'/',4,'NOT_WELD_ACCURACY',flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
