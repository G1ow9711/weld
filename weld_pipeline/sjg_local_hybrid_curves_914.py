"""Add geometry-only local surface carriers to the frozen learned branch."""
import argparse
import gc
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_local_xyz_planes_913 import extract_planes
from weld_pipeline.sjg_local_surface_curves_912 import serial,line_profiles,save_faces
from weld_pipeline.sjg_planar_boundary_curves_894 import face_boundary_arcs
from weld_pipeline.sjg_local_curve_supervision_915 import controls

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);cohort=read(OUT/'run98/local_cohort.json');records=[]
    assert read(OUT/'run107/receipt.json')['status']=='LOCAL_FINITE_CURVE_INPUTS_FROZEN'
    cases=sorted(cohort['cases'],key=lambda c:c['source_key']!='BM79A_native_local')
    write(dest/'registered_protocol.json',dict(stage='LOCAL_RAW_AND_LEARNED_SURFACE_CURVES',source_sha256=digest(__file__),
        raw_extractor_sha256=digest(ROOT/'weld_pipeline/sjg_local_xyz_planes_913.py'),
        original_G=496,new_BM79A_candidate_G=4,input_frame='workpiece_local',unit='mm',
        old_794_rules_recomputed_on_native_input=True,raw_plus_learned_recall_not_learned_replacement=True,
        max_candidate_iterations=512,plane_tolerance_mm=.2,STEP_reads=0,target_reads=0,
        all_existing_learned_profiles_retained=True,optimizer_updates=0,full_contract_pass=False,start_utc=utc()))
    try:
        write(dest/'typed_event_controls.json',controls())
        print('TYPED_LINE_ARC_EVENT_CONTROLS_PASSED',flush=True)
        for case in cases:
            check_resources();key=case['source_key'];folder=dest/f'source_{key}';folder.mkdir();start=time.perf_counter()
            path=OUT/case['input_file'];assert digest(path)==case['input_sha256'];xyz=np.load(path,mmap_mode='r',allow_pickle=False)
            faces,stats=extract_planes(xyz,progress=lambda *v:print('LOCAL_RAW',key,*v,flush=True))
            profiles=line_profiles(faces,xyz,lambda *v:print('LOCAL_RAW',key,*v,flush=True));raw_lines=len(profiles)
            for i,face in enumerate(faces):
                arcs,_=face_boundary_arcs(xyz,face,face['point_ids'],face['triangles'])
                for arc in arcs:
                    profiles.append(dict(representation=dict(kind='arc',arc_parameters=arc['arc_parameters']),
                        length_mm=arc['length_mm'],face_indices=[i],support_normal=face['normal'].tolist(),
                        support_intervals_mm=[[0.,arc['length_mm']]],unobserved_gaps_mm=[],material_pair=None,weld_type=None,
                        second_contact_face_unknown=True,status=arc['status'],proposal_branch='raw_geometry',
                        boundary_arc_evidence={k:v for k,v in arc.items() if k not in ('arc_parameters','material_pair','weld_type','status')}))
                if i%20==0:print('LOCAL_RAW',key,'BOUNDARY_ARCS',i+1,'/',len(faces),'ARCS',len(profiles)-raw_lines,flush=True)
            raw_faces=len(faces);raw_curves=len(profiles);previous=OUT/'run107'/f'source_{key}';binding=read(previous/'input_freeze.json')
            assert binding['input_sha256']==case['input_sha256']
            for file in ('surfaces.json','surface_support.npz','profiles.json'):assert digest(previous/file)==binding['files'][file]
            old_faces=read(previous/'surfaces.json')
            with np.load(previous/'surface_support.npz',allow_pickle=False) as z:
                for face in old_faces:
                    face['point_ids']=z[face.pop('point_ids_key')];face['triangles']=z[face.pop('triangles_key')]
                    face['proposal_branch']='learned'
                    faces.append(face)
            old_profiles=read(previous/'profiles.json')
            for profile in old_profiles:
                profile['face_indices']=[i+raw_faces for i in profile['face_indices']];profile['proposal_branch']='learned';profiles.append(profile)
            for p in profiles:p.setdefault('proposal_branch','raw_geometry')
            save_faces(folder,faces);write(folder/'profiles.json',serial(profiles))
            row=dict(source_key=key,role=case['role'],input_file=case['input_file'],input_sha256=case['input_sha256'],
                points=len(xyz),raw_faces=raw_faces,learned_faces=len(old_faces),raw_lines=raw_lines,raw_arcs=raw_curves-raw_lines,
                learned_profiles=len(old_profiles),profiles=len(profiles),stats=stats,elapsed_seconds=time.perf_counter()-start,
                files={file:digest(folder/file) for file in ('surfaces.json','surface_support.npz','profiles.json')})
            write(folder/'input_freeze.json',row);records.append(row)
            print('LOCAL_HYBRID_FROZEN',key,'RAW_LINES',raw_lines,'RAW_ARCS',raw_curves-raw_lines,'ALL',len(profiles),flush=True)
            del xyz,faces,profiles,old_faces,old_profiles;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=records,STEP_reads=0,target_reads=0))
        write(dest/'receipt.json',dict(status='LOCAL_HYBRID_TYPED_CURVES_FROZEN',sources=records,original_G=496,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_sources=records,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
