"""Uniform original 950 candidate reconstruction without historical cache shortcuts."""
import argparse,copy,time,traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
ADAPTER=install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest,check_resources
from weld_pipeline.sjg_unified_local_candidates_950 import expand_planes,recover,canonical_normals,line_profiles,face_lines,envelope_profiles,save_graph_faces
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/best_baseline_restore'
def run(key):
    case=read(BASE/'run26_merged_point_features'/('source_'+key)/'receipt.json')
    file=ROOT/case['input_file'];assert digest(file)==case['input_sha256'];xyz=np.load(file,mmap_mode='r')
    previous=BASE/'rank2_candidates'/('source_'+key);binding=read(previous/'input_freeze.json')
    assert binding['input_sha256']==case['input_sha256']
    for name in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(previous/name)==binding['files'][name]
    folder=BASE/'unified_xyz_candidates'/('source_'+key);folder.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    write(folder/'protocol.json',dict(source_sha256=digest(__file__),original_source_sha256=digest(ROOT/'weld_pipeline/sjg_unified_local_candidates_950.py'),adapter=ADAPTER,GT_reads=0,STEP_reads=0,cache_shortcuts_disabled=True))
    def progress(*v):
        check_resources();write(folder/'progress.json',dict(event=v,seconds=time.perf_counter()-start));print(key,*v,flush=True)
    faces,stats=expand_planes(xyz,read(previous/'surfaces.json'),128,progress)
    profiles=line_profiles(faces,xyz,progress)
    for i,face in enumerate(faces):
        for p in face_lines(xyz,face):p.update(face_indices=[i],proposal_branch='global_observed_boundary');profiles.append(p)
    normal_data=canonical_normals(xyz,progress)
    new,offset_stats=recover(xyz,faces,normal_data,progress,selection='direction_balanced');faces.extend(new)
    stats['direction_offset_recovery']=offset_stats;del normal_data
    profiles.extend(envelope_profiles(xyz,faces,profiles,progress))
    # Keep previously generated 3D arc paths and their own native face
    # support. New straight-plane branches never replace circular paths.
    old_profiles=read(previous/'profiles.json');arcs=[p for p in old_profiles if p['representation']['kind']=='arc'];del old_profiles
    referenced=sorted({i for p in arcs for i in p['face_indices']});old_faces=read(previous/'surfaces.json');mapping={}
    with np.load(previous/'surface_support.npz',allow_pickle=False) as z:
        for i in referenced:
            face=copy.deepcopy(old_faces[i]);face['point_ids']=z[face['point_ids_key']];mapping[i]=len(faces);faces.append(face)
    for p in arcs:p['face_indices']=[mapping[i] for i in p['face_indices']]
    profiles.extend(arcs)
    for p in profiles:
        p.setdefault('proposal_branch','global_observed_plane_pair')
        assert p['length_mm']>0 and all(0<=i<len(faces) for i in p['face_indices'])
    save_graph_faces(folder,faces);write(folder/'profiles.json',profiles);write(folder/'support_stats.json',stats)
    freeze=dict(input_sha256=case['input_sha256'],files={n:digest(folder/n) for n in ('surfaces.json','surface_support.npz','profiles.json')},
        STEP_reads=0,target_reads_before_freeze=0,graph_support_requires_point_ids_only=True)
    write(folder/'input_freeze.json',freeze)
    row=dict(source_key=key,faces=len(faces),profiles=len(profiles),arc_profiles=len(arcs),elapsed_seconds=time.perf_counter()-start)
    write(folder/'receipt.json',dict(status='ORIGINAL_UNIFIED_XYZ_CANDIDATES_RECONSTRUCTED',historical_parity_proven=False,weld_accuracy_measured=False,**row))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);args=ap.parse_args()
    with threadpool_limits(limits=2):run(args.key)
