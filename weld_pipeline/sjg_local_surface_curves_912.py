"""Finite learned surfaces and typed line/circle carriers from native local XYZ."""
import argparse
import gc
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_learned_surfaces_810 import extract
from weld_pipeline.sjg_planar_boundary_curves_894 import face_boundary_arcs

def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [serial(x) for x in v]
    return v

def intervals(face,xyz,origin,direction):
    # Same finite-triangle cross-section rule as 795, without training imports.
    tri=xyz[face['point_ids']][face['triangles']];side=np.cross(face['normal'],direction)
    signed=(tri-origin)@side;q=(tri-origin)@direction;tolerance=.2
    mask=(signed.min(1)<=tolerance)&(signed.max(1)>=-tolerance);signed=signed[mask];q=q[mask]
    lower=np.full(len(q),np.inf);upper=np.full(len(q),-np.inf)
    for i,j in ((0,1),(1,2),(2,0)):
        at=np.abs(signed[:,i])<=tolerance
        lower[at]=np.minimum(lower[at],q[at,i]);upper[at]=np.maximum(upper[at],q[at,i])
        for boundary in (-tolerance,tolerance):
            aa=signed[:,i]-boundary;bb=signed[:,j]-boundary;cross=aa*bb<0
            cut=q[cross,i]+(q[cross,j]-q[cross,i])*aa[cross]/(aa[cross]-bb[cross])
            lower[cross]=np.minimum(lower[cross],cut);upper[cross]=np.maximum(upper[cross],cut)
    valid=(upper-lower)>1e-6;order=np.argsort(lower[valid]);merged=[]
    for lo,hi in zip(lower[valid][order],upper[valid][order]):
        if merged and lo<=merged[-1][1]+1e-5:merged[-1][1]=max(merged[-1][1],float(hi))
        else:merged.append([float(lo),float(hi)])
    return merged

def line_profiles(faces,xyz,progress=None):
    bounds=[(xyz[f['point_ids']].min(0),xyz[f['point_ids']].max(0)) for f in faces];profiles=[]
    for i,a in enumerate(faces):
        for j in range(i+1,len(faces)):
            if np.any(np.maximum(bounds[i][0],bounds[j][0])>np.minimum(bounds[i][1],bounds[j][1])+.4):continue
            b=faces[j];direction=np.cross(a['normal'],b['normal']);norm=float(np.linalg.norm(direction))
            if norm<.15:continue
            direction/=norm
            if direction[np.argmax(np.abs(direction))]<0:direction=-direction
            normals=np.stack([a['normal'],b['normal']]);offset=np.array([a['offset'],b['offset']])
            origin=normals.T@np.linalg.solve(normals@normals.T,offset)
            left=intervals(a,xyz,origin,direction);right=intervals(b,xyz,origin,direction)
            pieces=sorted([max(alo,blo),min(ahi,bhi)] for alo,ahi in left for blo,bhi in right if min(ahi,bhi)-max(alo,blo)>=1.)
            if not pieces:continue
            merged=[]
            for lo,hi in pieces:
                if merged and lo<=merged[-1][1]+1e-5:merged[-1][1]=max(merged[-1][1],hi)
                else:merged.append([lo,hi])
            start,end=merged[0][0],merged[-1][1];support=[[lo-start,hi-start] for lo,hi in merged]
            profiles.append(dict(representation=dict(kind='line',endpoints_mm=np.stack([origin+start*direction,origin+end*direction]).tolist()),
                length_mm=end-start,face_indices=[i,j],support_normal=np.asarray(a['normal']).tolist(),
                support_intervals_mm=support,unobserved_gaps_mm=[[x[1],y[0]] for x,y in zip(support[:-1],support[1:])],
                material_pair=None,weld_type=None,status='UNCLASSIFIED_FINITE_PLANE_INTERSECTION',
                sampling_void_not_certified_hole=True))
        if progress and i%20==0:progress('FINITE_LINE_PAIRS',i+1,len(faces),'PROFILES',len(profiles))
    return profiles

def save_faces(folder,faces):
    arrays={};rows=[]
    for i,face in enumerate(faces):
        arrays[f'ids_{i}']=face['point_ids'];arrays[f'tri_{i}']=face['triangles']
        rows.append(dict(**{k:serial(v) for k,v in face.items() if k not in ('point_ids','triangles')},
                         point_ids_key=f'ids_{i}',triangles_key=f'tri_{i}'))
    np.savez(folder/'surface_support.npz',**arrays);write(folder/'surfaces.json',rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);data=OUT/'run106';records=[]
    assert read(data/'receipt.json')['status']=='LOCAL_FEATURES_ASSEMBLED_AND_VERIFIED'
    manifest=read(data/'all_predictions_frozen.json');cases=manifest['sources']
    # Start with the newly supplied circular TRAIN candidate to test the new
    # representation early; all original eight cases remain in the schedule.
    cases=sorted(cases,key=lambda c:c['source_key']!='BM79A_native_local')
    write(dest/'registered_protocol.json',dict(stage='LOCAL_PTV3_FINITE_SURFACES_AND_TYPED_CURVES',start_utc=utc(),
        source_sha256=digest(__file__),feature_manifest_sha256=digest(data/'all_predictions_frozen.json'),
        sources=[c['source_key'] for c in cases],original_G=496,new_candidate_G=4,
        feature_checkpoint='run26',native_XYZ_only=True,STEP_reads=0,weld_target_reads=0,
        curved_surface_instance_fitting_not_yet_implemented=True,
        arc_carriers_from_observed_finite_plane_boundaries=True,arc_second_material_owner_unknown=True,
        optimizer_updates=0,full_contract_pass=False))
    try:
        # Exact analytic cross-section sanity check, without a target reader.
        square=np.array([[-10,-10,0],[10,-10,0],[10,10,0],[-10,10,0]],float)
        cross=square[:,[0,2,1]];cloud=np.r_[square,cross]
        tri=np.array([[0,1,2],[0,2,3]],np.int64)
        faces=[dict(point_ids=np.arange(4),triangles=tri,normal=np.array([0.,0.,1.]),offset=0.),
               dict(point_ids=np.arange(4,8),triangles=tri,normal=np.array([0.,1.,0.]),offset=0.)]
        test=line_profiles(faces,cloud);assert len(test)==1 and abs(test[0]['length_mm']-20)<1e-8
        for case in cases:
            check_resources();key=case['source_key'];folder=dest/f'source_{key}';folder.mkdir();start=time.perf_counter()
            xyz_file=OUT/case['input_file'];feature_file=data/case['feature_file']
            assert digest(xyz_file)==case['input_sha256'] and digest(feature_file)==case['feature_sha256']
            xyz=np.load(xyz_file,mmap_mode='r',allow_pickle=False)
            with np.load(feature_file,allow_pickle=False) as z:pred={k:z[k] for k in ('embedding','normal','boundary')}
            def progress(*v):
                check_resources();print('LOCAL_FINITE_SURFACE',key,*v,flush=True)
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
            write(folder/'input_freeze.json',row);records.append(row)
            print('LOCAL_CURVE_INPUTS_FROZEN',key,'LINES',line_count,'ARCS',len(profiles)-line_count,flush=True)
            del xyz,pred,faces,labels,profiles;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=records,label_reads=0,STEP_reads=0))
        write(dest/'receipt.json',dict(status='LOCAL_FINITE_CURVE_INPUTS_FROZEN',sources=records,
            optimizer_updates=0,no_weld_accuracy_claim=True,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',completed_sources=records,error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
