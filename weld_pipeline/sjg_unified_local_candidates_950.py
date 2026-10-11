"""Uniform native XYZ candidate repair for the unchanged nine-case pilot."""
import argparse
import copy
import gc
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_global_plane_support_938 import expand_planes
from weld_pipeline.sjg_rank2_offset_recovery_947 import recover
from weld_pipeline.sjg_observed_cylinders_919 import canonical_normals
from weld_pipeline.sjg_local_surface_curves_912 import line_profiles,serial
from weld_pipeline.sjg_observed_boundary_lines_927 import face_lines
from weld_pipeline.sjg_finite_envelope_carriers_941 import envelope_profiles


def load_faces(folder):
    faces=read(folder/'surfaces.json')
    with np.load(folder/'surface_support.npz',allow_pickle=False) as z:
        for f in faces:
            f['point_ids']=z[f['point_ids_key']]
            if f.get('triangles_key') in z.files:f['triangles']=z[f['triangles_key']]
    return faces


def save_graph_faces(folder,faces):
    rows=[];support={}
    for i,f in enumerate(faces):
        row={k:serial(v) for k,v in f.items() if k not in ('point_ids','triangles','point_ids_key','triangles_key')}
        row['point_ids_key']='ids_'+str(i);support[row['point_ids_key']]=f['point_ids'];rows.append(row)
    write(folder/'surfaces.json',rows);np.savez_compressed(folder/'surface_support.npz',**support)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    cases=read(OUT/'run98/local_cohort.json')['cases'];rows=[]
    order=['10','4','21','25','36','42','BM79A_native_local','64','24'];cases=sorted(cases,key=lambda c:order.index(c['source_key']))
    write(dest/'registered_protocol.json',dict(stage='UNIFIED_LOCAL_XYZ_GEOMETRY_REPAIR',sources=order,
        initial_planes=128,additional_planes=128,additional_selection='direction_balanced',
        original_arc_carriers_retained=True,boundary_line_carriers_retained=True,
        STEP_reads=0,weld_target_reads=0,original_eight_G=496,additional_BM_G=4,optimizer_updates=0,
        source_sha256=digest(__file__),plane_module_sha256=digest(ROOT/'weld_pipeline/sjg_rank2_offset_recovery_947.py'),start_utc=utc()))
    try:
        for case in cases:
            check_resources();key=case['source_key'];folder=dest/('source_'+key);folder.mkdir();start=time.perf_counter()
            assert digest(OUT/case['input_file'])==case['input_sha256'];xyz=np.load(OUT/case['input_file'],mmap_mode='r')
            previous=OUT/'run115'/('source_'+key);binding=read(previous/'input_freeze.json')
            for name in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(previous/name)==binding['files'][name]
            def progress(*v):print(key,*v,flush=True)
            # Exact earlier native-input results may be reused. Every case uses
            # the same 128 + 128 plane and direction-balanced protocol.
            if key=='10':
                faces=load_faces(OUT/'run131');profiles=read(OUT/'run131/profiles.json');stats=dict(reused_verified_run='run131')
                frozen=read(OUT/'run131/input_freeze.json')
                assert frozen['input_sha256']==case['input_sha256'] and digest(OUT/'run131/profiles.json')==frozen['profiles_sha256']
            else:
                if key=='4':
                    base=OUT/'run125/source_4';faces=load_faces(base);profiles=read(base/'profiles.json');stats=dict(reused_base='run125/source_4')
                    freeze=read(base/'input_freeze.json')
                    for name in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(base/name)==freeze['files'][name]
                else:
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
            rows.append(row);write(folder/'receipt.json',dict(status='UNIFIED_LOCAL_CASE_FROZEN',**row,optimizer_updates=0,end_utc=utc()))
            print('UNIFIED_LOCAL_CASE_FROZEN',key,'FACES',len(faces),'PROFILES',len(profiles),'ARCS',len(arcs),flush=True)
            del faces,profiles,arcs,old_faces,xyz;gc.collect();reclaim_own_idle_pages()
        write(dest/'receipt.json',dict(status='UNIFIED_LOCAL_CANDIDATES_FROZEN',sources=rows,STEP_reads=0,weld_target_reads=0,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_sources=rows,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
