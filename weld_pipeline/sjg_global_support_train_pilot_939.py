"""Single TRAIN pilot of whole-cloud support for learned plane hypotheses."""
import argparse
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_global_plane_support_938 import expand_planes
from weld_pipeline.sjg_local_surface_curves_912 import save_faces,line_profiles,serial
from weld_pipeline.sjg_observed_boundary_lines_927 import face_lines
from weld_pipeline.sjg_local_curve_supervision_915 import supervise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--source-key',default='21');ap.add_argument('--maximum-planes',type=int,default=32)
    args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    key=args.source_key;source=OUT/'run115'/('source_'+key);binding=read(source/'input_freeze.json')
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert case['role']=='TRAIN'
    write(dest/'registered_protocol.json',dict(stage='GLOBAL_OBSERVED_PLANE_SUPPORT_TRAIN_PILOT',source_key=key,G=case['original_G'],
        source_sha256=digest(__file__),geometry_sha256=digest(ROOT/'weld_pipeline/sjg_global_plane_support_938.py'),
        max_planes=args.maximum_planes,selection='canonical_inlier_count_then_extent',plane_tolerance_mm=.2,
        mesher_sha256=digest(ROOT/'weld_pipeline/sjg_rank_adaptive_mesh_936.py'),
        input_plane_equations_unchanged=True,GT_not_used_to_select_planes=True,STEP_reads=0,optimizer_updates=0,
        full_contract_pass=False,start_utc=utc()))
    try:
        assert digest(source/'surfaces.json')==binding['files']['surfaces.json']
        assert digest(OUT/case['input_file'])==case['input_sha256'];xyz=np.load(OUT/case['input_file'],mmap_mode='r')
        def progress(*v):check_resources();print(*v,flush=True)
        faces,stats=expand_planes(xyz,read(source/'surfaces.json'),maximum=args.maximum_planes,progress=progress)
        profiles=line_profiles(faces,xyz,progress)
        for i,face in enumerate(faces):
            lines=face_lines(xyz,face)
            for line in lines:line['face_indices']=[i];line['proposal_branch']='global_observed_boundary';profiles.append(line)
        save_faces(dest,faces);write(dest/'profiles.json',serial(profiles));write(dest/'support_stats.json',stats)
        write(dest/'input_freeze.json',dict(input_sha256=case['input_sha256'],files={f:digest(dest/f) for f in ('surfaces.json','surface_support.npz','profiles.json')},
                                           target_reads_before_freeze=0,STEP_reads=0))
        score=supervise(profiles,read(OUT/case['target_file']));write(dest/'TRAIN_only_geometric_compatibility.json',score)
        write(dest/'receipt.json',dict(status='GLOBAL_OBSERVED_SUPPORT_TRAIN_PILOT_COMPLETE',G=case['original_G'],geometrically_supported=score['covered_children'],
            missing=score['missing'],faces=len(faces),profiles=len(profiles),optimizer_updates=0,not_weld_accuracy=True,
            full_contract_pass=False,end_utc=utc()))
        print('GLOBAL_SUPPORT_TRAIN_GEOMETRY',score['covered_children'],'/',case['original_G'],'NOT_WELD_ACCURACY',flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
