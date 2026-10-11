"""Rebuild original run115 candidates from raw XYZ and original run26 surfaces."""
import argparse,time,traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
ADAPTER=install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest,check_resources
from weld_pipeline.sjg_rank2_finite_planes_925 import extract_planes
from weld_pipeline.sjg_local_surface_curves_912 import serial,line_profiles,save_faces
from weld_pipeline.sjg_polygonal_arcs_921 import face_arcs as base_arcs
from weld_pipeline.sjg_polygonal_arc_extension_923 import face_arcs as extended_arcs,controls
from weld_pipeline.sjg_observed_boundary_lines_927 import face_lines
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/best_baseline_restore'
def generate(case,folder,progress=print):
    path=ROOT/case['input_file'];assert digest(path)==case['input_sha256']
    xyz=np.load(path,mmap_mode='r');start=time.perf_counter()
    faces,stats=extract_planes(xyz,progress=progress)
    profiles=line_profiles(faces,xyz,progress);lines=len(profiles)
    for i,face in enumerate(faces):
        for branch,extractor in (('polygonal_boundary',base_arcs),('connected_boundary_extension',extended_arcs)):
            arcs,_=extractor(xyz,face)
            for arc in arcs:arc.update(face_indices=[i],proposal_branch=branch);profiles.append(arc)
        progress('RANK2_BOUNDARY',i+1,len(faces),'ARCS',len(profiles)-lines)
    raw_faces=len(faces);raw_profiles=len(profiles)
    previous=BASE/'run26_learned_surfaces'/('source_'+case['source_key']);binding=read(previous/'receipt.json')
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


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);args=ap.parse_args()
    case=read(BASE/'run26_merged_point_features'/('source_'+args.key)/'receipt.json');case['role']='TRAIN'
    folder=BASE/'rank2_candidates'/('source_'+args.key);folder.mkdir(parents=True,exist_ok=False)
    write(folder/'protocol.json',dict(source_sha256=digest(__file__),original_source_sha256=digest(ROOT/'weld_pipeline/sjg_rank2_curve_pilot_926.py'),adapter=ADAPTER,GT_reads=0,STEP_reads=0,historical_parity_proven=False))
    def progress(*v):
        check_resources();write(folder/'progress.json',dict(event=v));print(args.key,*v,flush=True)
    try:
        with threadpool_limits(limits=2):
            write(folder/'controls.json',controls());profiles,row=generate(case,folder,progress)
        write(folder/'receipt.json',dict(status='ORIGINAL_RANK2_CANDIDATES_RECONSTRUCTED',historical_parity_proven=False,weld_accuracy_measured=False,**row))
    except BaseException:
        write(folder/'failure.json',dict(error=traceback.format_exc()));raise
