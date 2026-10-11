"""Reconstruct original 912 learned surfaces/curves using authenticated run26 features."""
import argparse
import time
import traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.server_legacy_runtime import install
ADAPTER=install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest,check_resources
from weld_pipeline.sjg_local_surface_curves_912 import extract,line_profiles,face_boundary_arcs,save_faces,serial
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/best_baseline_restore'

def run(key):
    data=BASE/'run26_merged_point_features'/('source_'+key)
    case=read(data/'receipt.json');case['role']='TRAIN'
    xyz_file=ROOT/case['input_file'];feature_file=data/case['feature_file']
    assert digest(xyz_file)==case['input_sha256'] and digest(feature_file)==case['feature_sha256']
    folder=BASE/'run26_learned_surfaces'/('source_'+key);folder.mkdir(parents=True,exist_ok=False)
    write(folder/'protocol.json',dict(source_sha256=digest(__file__),original_source_sha256=digest(ROOT/'weld_pipeline/sjg_local_surface_curves_912.py'),adapter=ADAPTER,GT_reads=0,STEP_reads=0))
    start=time.perf_counter()
    xyz=np.load(xyz_file,mmap_mode='r',allow_pickle=False)
    with np.load(feature_file,allow_pickle=False) as z:pred={k:z[k] for k in ('embedding','normal','boundary')}
    def progress(*v):
        check_resources();write(folder/'progress.json',dict(event=v,seconds=time.perf_counter()-start));print('ORIGINAL_RUN107',key,*v,flush=True)
    faces,labels,stats=extract(xyz,pred,progress)
    print('LOCAL_FACE_FIT_COMPLETE',key,len(faces),flush=True)
    profiles=line_profiles(faces,xyz,progress);line_count=len(profiles);arc_diagnostics=[]
    for i,face in enumerate(faces):
        arcs,counts=face_boundary_arcs(xyz,face,face['point_ids'],face['triangles']);arc_diagnostics.append(dict(face=i,**counts))
        for arc in arcs:
            profiles.append(dict(representation=dict(kind='arc',arc_parameters=arc['arc_parameters']),
                length_mm=arc['length_mm'],face_indices=[i],support_normal=np.asarray(face['normal']).tolist(),
                support_intervals_mm=[[0.,arc['length_mm']]],unobserved_gaps_mm=[],
                material_pair=None,weld_type=None,status=arc['status'],second_contact_face_unknown=True,
                boundary_arc_evidence={k:v for k,v in arc.items() if k not in ('arc_parameters','material_pair','weld_type','status')}))
        if i%20==0:progress('FINITE_BOUNDARY_ARCS',i+1,len(faces),'ARCS',len(profiles)-line_count)
    save_faces(folder,faces);np.save(folder/'affinity_components.npy',labels)
    write(folder/'profiles.json',serial(profiles));write(folder/'boundary_diagnostics.json',arc_diagnostics)
    row=dict(source_key=key,role=case['role'],points=len(xyz),input_sha256=case['input_sha256'],
        feature_sha256=case['feature_sha256'],surfaces=len(faces),line_carriers=line_count,
        arc_carriers=len(profiles)-line_count,stats=serial(stats),elapsed_seconds=time.perf_counter()-start,
        files={f:digest(folder/f) for f in ('surfaces.json','surface_support.npz','profiles.json','affinity_components.npy','boundary_diagnostics.json')})
    write(folder/'receipt.json',dict(status='ORIGINAL_RUN107_BRANCH_RECONSTRUCTED', historical_parity_proven=False, weld_accuracy_measured=False, GT_reads=0, STEP_reads=0, **row))
    print('LOCAL_CURVE_INPUTS_FROZEN',key,'LINES',line_count,'ARCS',len(profiles)-line_count,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);args=ap.parse_args()
    try:
        with threadpool_limits(limits=2):run(args.key)
    except BaseException:
        folder=BASE/'run26_learned_surfaces'/('source_'+args.key)
        if folder.exists():write(folder/'failure.json',dict(error=traceback.format_exc()))
        raise
