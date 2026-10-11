"""Conditional TRAIN-native planar face, boundary and material-owner supervision."""
import argparse
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_point_teacher_803 import point_teachers


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    pose=read(OUT/'run158/receipt.json');assert pose['source']=='21' and pose['role']=='TRAIN'
    # Do not change the failed nearest-sample gate. This separate diagnostic
    # tests continuous finite-face consistency and keeps pose provenance explicit.
    assert len(pose['candidates'])==1, 'Ambiguous exporter orientation; no teacher assignment permitted'
    candidate=pose['candidates'][0];r=np.array(candidate['world_to_native_rotation']);t=np.array(candidate['world_to_native_translation_mm'])
    write(dest/'registered_protocol.json',dict(stage='CONDITIONAL_NATIVE_PLANAR_TRAIN_TEACHER',source='21',
        input_run='run96',pose_run='run158',pose_uses_TRAIN_export_correspondence=True,
        previous_nearest_sample_gate_not_overridden=True,continuous_finite_face_check_separate=True,
        reference_poses_not_independent_acceptance_evidence=True,STEP_offline_only=True,
        original_points_all_retained=True,ambiguous_face_and_material_unknown=True,
        curved_primitive_labels_not_implemented=True,source_sha256=digest(__file__),start_utc=utc()))
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21');assert case['role']=='TRAIN'
    assert digest(OUT/case['input_file'])==pose['native_input_sha256']==case['input_sha256']
    folder=OUT/'source_21';freeze=read(folder/'input_freeze.json');assert digest(folder/'geometry.json')==freeze['geometry_sha256']
    faces=read(folder/'geometry.json')['faces']
    for face in faces:
        face['normal']=(r@np.array(face['normal'])).tolist();face['center']=(r@np.array(face['center'])+t).tolist()
        face['lines']=[(np.array(v)@r.T+t).tolist() for v in face['lines']]
        vertices=np.concatenate(face['lines']);face['lo']=vertices.min(0).tolist();face['hi']=vertices.max(0).tolist()
    write(dest/'conditional_native_faces.json',dict(faces=faces,runtime_input_allowed=False,source='21',role='TRAIN',
        input_sha256=case['input_sha256'],pose_receipt_sha256=digest(OUT/'run158/receipt.json'),
        reference_geometry_sha256=freeze['geometry_sha256'],independently_certified_pose=False))
    xyz=np.load(OUT/case['input_file'],mmap_mode='r');check_resources();labels=point_teachers(xyz,faces,'21_native')
    count=labels['membership_count'];known=labels['instance_known'];volumes=labels['reference_volume']
    assert len(labels['instance'])==len(xyz)
    residual=np.full(len(xyz),np.nan)
    per_face=[]
    for fi,face in enumerate(faces):
        ids=np.flatnonzero(labels['instance']==fi)
        values=np.abs((xyz[ids]-np.array(face['center']))@np.array(face['normal']))
        residual[ids]=values
        per_face.append(dict(face=fi,reference_volume=face['volume'],unique_points=len(ids),
            plane_residual_max_mm=float(values.max()) if len(values) else None))
    file=dest/'source_21_native_teacher.npz';np.savez_compressed(file,**labels)
    assert int((count==0).sum())+int((count==1).sum())+int((count>1).sum())==len(xyz)
    record=dict(status='CONDITIONAL_NATIVE_PLANAR_TEACHER_DIAGNOSED',source='21',role='TRAIN',points=len(xyz),
        unique_face_points=int(known.sum()),multiple_face_points=int((count>1).sum()),uncovered_points=int((count==0).sum()),
        planar_covered_fraction=float((count>0).mean()),unique_material_points=int((volumes>=0).sum()),
        multiple_material_points=int(labels['multiple_material'].sum()),
        material_counts={str(v):int((volumes==v).sum()) for v in np.unique(volumes) if v>=0},
        boundary_positive=int((labels['boundary_target']==1).sum()),boundary_negative=int((labels['boundary_target']==0).sum()),
        unique_face_residual_quantiles_mm=np.quantile(residual[known],[.5,.95,.99,1]).tolist() if known.any() else None,
        per_face=per_face,input_sha256=case['input_sha256'],teacher_sha256=digest(file),
        reference_pose_independence_not_established=True,full_geometry_pose_certification_not_claimed=True,
        unknowns_not_relabelled_background=True,original_point_order_preserved=True,
        customer_approved_E16_truth=False,optimizer_updates=0,not_extraction_accuracy=True,full_contract_pass=False,end_utc=utc())
    write(dest/'receipt.json',record);print('NATIVE_PLANAR_TEACHER',record,flush=True)

if __name__=='__main__':main()
