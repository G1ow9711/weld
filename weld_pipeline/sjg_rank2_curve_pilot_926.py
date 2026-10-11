"""Freeze stable local plane/curve input before a small TRAIN-only diagnosis."""
import argparse
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_rank2_finite_planes_925 import extract_planes
from weld_pipeline.sjg_local_surface_curves_912 import serial,line_profiles,save_faces
from weld_pipeline.sjg_polygonal_arcs_921 import face_arcs as base_arcs
from weld_pipeline.sjg_polygonal_arc_extension_923 import face_arcs as extended_arcs,controls
from weld_pipeline.sjg_observed_boundary_lines_927 import face_lines
from weld_pipeline.sjg_local_curve_supervision_915 import supervise


def generate(case,folder,progress=print):
    path=OUT/case['input_file'];assert digest(path)==case['input_sha256']
    xyz=np.load(path,mmap_mode='r');start=time.perf_counter()
    faces,stats=extract_planes(xyz,progress=progress)
    profiles=line_profiles(faces,xyz,progress);lines=len(profiles)
    for i,face in enumerate(faces):
        for branch,extractor in (('polygonal_boundary',base_arcs),('connected_boundary_extension',extended_arcs)):
            arcs,_=extractor(xyz,face)
            for arc in arcs:arc.update(face_indices=[i],proposal_branch=branch);profiles.append(arc)
        progress('RANK2_BOUNDARY',i+1,len(faces),'ARCS',len(profiles)-lines)
    raw_faces=len(faces);raw_profiles=len(profiles)
    previous=OUT/'run107'/('source_'+case['source_key']);binding=read(previous/'input_freeze.json')
    assert binding['input_sha256']==case['input_sha256']
    for file in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(previous/file)==binding['files'][file]
    old_faces=read(previous/'surfaces.json')
    with np.load(previous/'surface_support.npz',allow_pickle=False) as z:
        for face in old_faces:
            face['point_ids']=z[face.pop('point_ids_key')];face['triangles']=z[face.pop('triangles_key')]
            face['proposal_branch']='learned';faces.append(face)
    for profile in read(previous/'profiles.json'):
        profile['face_indices']=[i+raw_faces for i in profile['face_indices']];profile['proposal_branch']='learned';profiles.append(profile)
    boundary_lines=0
    for i,face in enumerate(faces[:raw_faces]):
        for profile in face_lines(xyz,face):
            profile.update(face_indices=[i],proposal_branch='observed_boundary_line');profiles.append(profile);boundary_lines+=1
        progress('RANK2_BOUNDARY_LINES',i+1,raw_faces,'LINES',boundary_lines)
    for profile in profiles:profile.setdefault('proposal_branch','raw_geometry')
    save_faces(folder,faces);write(folder/'profiles.json',serial(profiles))
    row=dict(source_key=case['source_key'],role=case['role'],input_sha256=case['input_sha256'],
        points=len(xyz),raw_faces=raw_faces,learned_faces=len(old_faces),raw_lines=lines+boundary_lines,raw_arcs=raw_profiles-lines,
        profiles=len(profiles),stats=stats,elapsed_seconds=time.perf_counter()-start,
        files={f:digest(folder/f) for f in ('surfaces.json','surface_support.npz','profiles.json')},
        target_reads_before_freeze=0,STEP_reads=0)
    write(folder/'input_freeze.json',row)
    return profiles,row


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='BM79A_native_local')
    write(dest/'registered_protocol.json',dict(stage='RANK2_FINITE_CURVE_TRAIN_PILOT',source_sha256=digest(__file__),
        geometry_sha256=digest(ROOT/'weld_pipeline/sjg_rank2_finite_planes_925.py'),original_eight_G=496,pilot_TRAIN_G=4,
        grid_mm=1.,normal_neighbors=32,maximum_axis_families=64,plane_tolerance_mm=.2,
        optimizer_updates=0,full_contract_pass=False,start_utc=utc()))
    try:
        write(dest/'controls.json',controls())
        def progress(*v):check_resources();print(*v,flush=True)
        profiles,row=generate(case,dest,progress)
        score=supervise(profiles,read(OUT/case['target_file']));write(dest/'TRAIN_only_geometric_compatibility.json',score)
        write(dest/'receipt.json',dict(status='RANK2_FINITE_CURVE_TRAIN_PILOT_COMPLETE',G=4,geometrically_supported=score['covered_children'],
            missing=score['missing'],inputs=row,optimizer_updates=0,not_weld_accuracy=True,full_contract_pass=False,end_utc=utc()))
        print('RANK2_TRAIN_GEOMETRY',score['covered_children'],'/',4,'NOT_WELD_ACCURACY',flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
