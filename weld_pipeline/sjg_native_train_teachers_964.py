"""Original-index native TRAIN teachers; reference structures remain offline."""
import argparse
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_point_teacher_803 import point_teachers


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    cohort=read(OUT/'run98/local_cohort.json')['cases'];inventory=read(OUT/'inventory.json');rows=[]
    cases=[c for c in cohort if c['role']=='TRAIN'];assert {c['source_key'] for c in cases}=={'21','4','10','25','36','42'}
    write(dest/'registered_protocol.json',dict(stage='NATIVE_ALL_POINT_TRAIN_TEACHERS',train=[c['source_key'] for c in cases],
        BM79A_structural_teacher='UNKNOWN_NO_BOUND_REFERENCE',BM79A_weld_TRAIN_unchanged=True,
        dev64_and_diagnostic24_not_read=True,original_eight_G=496,supplemental_BM_G=4,
        STEP_offline_only=True,point_order_and_original_denominators_preserved=True,
        customer_approved_attribute_truth_not_claimed=True,source_sha256=digest(__file__),start_utc=utc()))
    for case in cases:
        check_resources();key=case['source_key'];path=OUT/case['input_file'];assert digest(path)==case['input_sha256']
        xyz=np.load(path,mmap_mode='r');provenance={}
        if key=='21':
            previous=read(OUT/'run159/receipt.json');assert previous['input_sha256']==case['input_sha256']
            teacher=OUT/'run159/source_21_native_teacher.npz';assert digest(teacher)==previous['teacher_sha256']
            row=dict(source_key=key,input_file=case['input_file'],input_sha256=case['input_sha256'],points=len(xyz),
                teacher_file=str(teacher.relative_to(OUT)),teacher_sha256=digest(teacher),
                known_owner_points=previous['unique_material_points'],single_face_points=previous['unique_face_points'],
                boundary_positive=previous['boundary_positive'],boundary_negative=previous['boundary_negative'],
                uncovered_points=previous['uncovered_points'],material_counts=previous['material_counts'],
                conditional_TRAIN_export_pose=True,independent_pose_certification=False)
        else:
            inv=inventory[int(key)];folder=(ROOT/inv['raw']['original_pcd_path']).parent;exports=list(folder.glob('*_weld_welds.json'));assert len(exports)==1
            native=read(exports[0]);assert native['coordinateFrame']=='reference_workpiece' and native['unit']=='millimeter'
            transform=np.array(native['exportToWorld']);r=transform[:3,:3];t=transform[:3,3]
            assert np.allclose(r.T@r,np.eye(3),atol=1e-7) and np.isclose(np.linalg.det(r),1,atol=1e-7)
            reference=OUT/f'source_{int(key):02d}';binding=read(reference/'input_freeze.json');assert digest(reference/'geometry.json')==binding['geometry_sha256']
            faces=read(reference/'geometry.json')['faces']
            for f in faces:
                f['normal']=(np.array(f['normal'])@r).tolist();f['center']=((np.array(f['center'])-t)@r).tolist()
                f['lines']=[((np.array(v)-t)@r).tolist() for v in f['lines']]
                vertices=np.concatenate(f['lines']);f['lo']=vertices.min(0).tolist();f['hi']=vertices.max(0).tolist()
            write(dest/f'source_{key}_offline_faces.json',dict(faces=faces,source_key=key,role='TRAIN',runtime_input_allowed=False,
                native_export_sha256=digest(exports[0]),reference_geometry_sha256=binding['geometry_sha256']))
            labels=point_teachers(xyz,faces,key);teacher=dest/f'source_{key}_native_teacher.npz';np.savez_compressed(teacher,**labels)
            counts=labels['membership_count'];owner=labels['reference_volume'];known=owner>=0
            assert int((counts==0).sum())+int((counts==1).sum())+int((counts>1).sum())==len(xyz)
            row=dict(source_key=key,input_file=case['input_file'],input_sha256=case['input_sha256'],points=len(xyz),
                teacher_file=str(teacher.relative_to(OUT)),teacher_sha256=digest(teacher),
                known_owner_points=int(known.sum()),single_face_points=int(labels['instance_known'].sum()),
                boundary_positive=int((labels['boundary_target']==1).sum()),boundary_negative=int((labels['boundary_target']==0).sum()),
                uncovered_points=int((counts==0).sum()),material_counts={str(v):int((owner==v).sum()) for v in np.unique(owner) if v>=0},
                native_export_sha256=digest(exports[0]),reference_geometry_sha256=binding['geometry_sha256'],
                original_STEP_registration_provenance=inv['step']['verified_offline_frames'],independent_pose_certification=False)
            del labels,faces
        rows.append(row);write(dest/'manifest.json',dict(rows=rows,role='TRAIN',BM79A_structural_teacher_unknown=True))
        print('NATIVE_TRAIN_TEACHER_COMPLETE',key,'POINTS',row['points'],'KNOWN_FACE',row['single_face_points'],'KNOWN_OWNER',row['known_owner_points'],flush=True)
        del xyz;gc.collect()
    write(dest/'receipt.json',dict(status='NATIVE_TRAIN_PLANAR_TEACHERS_READY',rows=rows,
        points=sum(r['points'] for r in rows),known_owner_points=sum(r['known_owner_points'] for r in rows),
        all_known_core_points_required_for_material_loss=True,unknowns_not_background=True,
        curved_surface_teachers_not_implemented=True,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))

if __name__=='__main__':main()
