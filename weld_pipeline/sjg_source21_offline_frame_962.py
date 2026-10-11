"""TRAIN-only export-pose binding, checked against entire legacy/native XYZ clouds."""
import argparse
import itertools
import numpy as np
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_local_coordinate_audit_896 import workbook_labels


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    inv=read(OUT/'inventory.json')[21];case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21')
    assert case['role']=='TRAIN';original=ROOT/inv['raw']['original_pcd_path'];book=next(original.parent.glob('*.xlsx'))
    assert digest(original)==inv['raw']['original_pcd_sha256'] and digest(OUT/case['input_file'])==case['input_sha256']
    labels=workbook_labels(book);assert len(labels)==1;label=labels[0]
    world_start=np.array(label['source_start']);world_end=np.array(label['source_end'])
    local_start=np.array(label['local_start']);local_end=np.array(label['local_end'])
    native=np.load(OUT/case['input_file'],mmap_mode='r');tree=cKDTree(native);world=read_xyz_pcd(original)*1000
    records=[]
    for perm in itertools.permutations(range(3)):
        for sign in itertools.product((-1,1),repeat=3):
            r=np.eye(3)[list(perm)]*np.array(sign)[:,None]
            if np.linalg.det(r)<.5:continue
            shift=local_start-r@world_start
            end_error=float(np.linalg.norm(r@world_end+shift-local_end))
            if end_error>.01:continue
            converted=world@r.T+shift;distance=tree.query(converted,workers=2)[0]
            record=dict(world_to_native_rotation=r.tolist(),world_to_native_translation_mm=shift.tolist(),
                export_end_consistency_mm=end_error,legacy_points_checked=len(world),
                whole_legacy_to_native_nearest_quantiles_mm=np.quantile(distance,[0,.5,.95,.99,1]).tolist(),
                legacy_to_native_fraction_within_3mm=float(np.mean(distance<=3)))
            records.append(record);print('OFFLINE_FRAME_CANDIDATE',record,flush=True)
    assert records
    records.sort(key=lambda r:(r['whole_legacy_to_native_nearest_quantiles_mm'][2],r['whole_legacy_to_native_nearest_quantiles_mm'][1]))
    best=records[0];runner=records[1] if len(records)>1 else None
    distinct=runner is None or runner['whole_legacy_to_native_nearest_quantiles_mm'][2]>best['whole_legacy_to_native_nearest_quantiles_mm'][2]+3
    accepted=best['whole_legacy_to_native_nearest_quantiles_mm'][2]<=3 and distinct
    result=dict(status='TRAIN_EXPORT_FRAME_DIAGNOSED',source='21',role='TRAIN',
        candidates=records,selected=best if accepted else None,geometry_support_gate_passed=accepted,
        pose_hypothesis_family='24 proper signed-axis permutations; exporter paired translation anchors',
        search_not_claimed_general_registration=True,translation_uses_TRAIN_weld_export_metadata=True,
        independent_of_weld_GT=False,for_offline_TRAIN_supervision_only=True,runtime_input_allowed=False,
        native_input_sha256=case['input_sha256'],legacy_input_sha256=digest(original),workbook_sha256=digest(book),
        original_STEP_registration_GT_use_unknown=True,full_contract_pass=False,end_utc=utc())
    write(dest/'receipt.json',result)
    if accepted:
        folder=OUT/'source_21';freeze=read(folder/'input_freeze.json');assert digest(folder/'geometry.json')==freeze['geometry_sha256']
        geometry=read(folder/'geometry.json');r=np.array(best['world_to_native_rotation']);t=np.array(best['world_to_native_translation_mm'])
        for f in geometry['faces']:
            f['normal']=(r@np.array(f['normal'])).tolist();f['center']=(r@np.array(f['center'])+t).tolist()
            f['lines']=[(np.array(p)@r.T+t).tolist() for p in f['lines']]
            vertices=np.concatenate(f['lines']);f['lo']=vertices.min(0).tolist();f['hi']=vertices.max(0).tolist()
        # Export only face teacher geometry. No runtime carrier contact list is produced.
        write(dest/'offline_native_faces.json',dict(faces=geometry['faces'],coordinate_frame='workpiece_local',unit='mm',
            source='21',role='TRAIN',reference_geometry_sha256=digest(folder/'geometry.json'),
            pose_receipt_sha256=digest(dest/'receipt.json'),runtime_input_allowed=False,
            customer_reviewed_E16_truth=False,original_STEP_registration_GT_use_unknown=True))
        print('OFFLINE_TRAIN_FACE_TEACHER_READY',len(geometry['faces']),flush=True)

if __name__=='__main__':main()
