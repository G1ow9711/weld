"""Small TRAIN experiment: repair finite-mesh sampling gaps, freeze, diagnose."""
import argparse
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_rank_adaptive_mesh_936 import remesh,controls
from weld_pipeline.sjg_observed_boundary_lines_927 import face_lines
from weld_pipeline.sjg_local_curve_supervision_915 import supervise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    key='21';source=OUT/'run115'/('source_'+key);binding=read(source/'input_freeze.json');case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    write(dest/'registered_protocol.json',dict(stage='FINITE_MESH_RANK_TRAIN_PILOT',source=key,pilot_G=1,original_eight_G=496,
        source_sha256=digest(__file__),mesher_sha256=digest(ROOT/'weld_pipeline/sjg_rank_adaptive_mesh_936.py'),
        largest_raw_faces=8,selection_uses_observed_point_count_only=True,grid_mm=1.,minimum_planar_rank_ratio=.0001,
        plane_parameters_unchanged=True,all_previous_profiles_retained=True,optimizer_updates=0,full_contract_pass=False,start_utc=utc()))
    try:
        write(dest/'controls.json',controls());print('RANK_ADAPTIVE_MESH_CONTROLS_PASSED',flush=True)
        for f in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(source/f)==binding['files'][f]
        assert digest(OUT/case['input_file'])==case['input_sha256'];xyz=np.load(OUT/case['input_file'],mmap_mode='r')
        faces=read(source/'surfaces.json');profiles=read(source/'profiles.json');before=len(profiles);details=[]
        with np.load(source/'surface_support.npz',allow_pickle=False) as z:
            chosen=sorted(range(binding['raw_faces']),key=lambda i:-len(z[faces[i]['point_ids_key']]))[:8]
            for number,i in enumerate(chosen):
                check_resources();face=dict(faces[i]);face['point_ids']=z[face['point_ids_key']]
                uv=(xyz[face['point_ids']]-np.asarray(face['center']))@np.asarray(face['basis'])
                print('RANK_MESH_FACE_START',number+1,len(chosen),'FACE',i,'POINTS',len(uv),flush=True)
                def progress(*v):check_resources();print(*v,flush=True)
                triangles,stats=remesh(uv,progress);face['triangles']=triangles
                lines=face_lines(xyz,face)
                for line in lines:line.update(face_indices=[i],proposal_branch='rank_adaptive_mesh_boundary');profiles.append(line)
                np.save(dest/('face_'+str(i)+'_triangles.npy'),triangles)
                details.append(dict(face=i,original_triangles=len(z[face['triangles_key']]),new_lines=len(lines),**stats))
                print('RANK_MESH_FACE_COMPLETE',i,'TRIANGLES',len(triangles),'NEW_LINES',len(lines),flush=True)
        write(dest/'profiles.json',profiles);write(dest/'mesh_stats.json',details)
        write(dest/'input_freeze.json',dict(input_sha256=case['input_sha256'],profiles_sha256=digest(dest/'profiles.json'),
            previous_profiles=before,new_profiles=len(profiles)-before,STEP_reads=0,target_reads_before_freeze=0))
        score=supervise(profiles,read(OUT/case['target_file']));write(dest/'TRAIN_only_geometric_compatibility.json',score)
        write(dest/'receipt.json',dict(status='RANK_ADAPTIVE_MESH_TRAIN_PILOT_COMPLETE',G=1,geometrically_supported=score['covered_children'],
            missing=score['missing'],optimizer_updates=0,not_weld_accuracy=True,full_contract_pass=False,end_utc=utc()))
        print('RANK_MESH_TRAIN_GEOMETRY',score['covered_children'],'/',1,'NOT_WELD_ACCURACY',flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
