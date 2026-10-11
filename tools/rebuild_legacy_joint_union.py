"""Original 999 union of reconstructed XYZ and learned finite-support branches."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from weld_pipeline.server_legacy_runtime import install
ADAPTER=install()
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_unified_local_candidates_950 import load_faces,save_graph_faces
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(__import__('os').environ.get('WELD_RESTORE_ROOT',str(ROOT/'server_rebuild/best_baseline_restore'))).resolve()
BASE.relative_to(ROOT)

def run(key):
    prior=BASE/'unified_xyz_candidates'/('source_'+key)
    finite=BASE/'learned_finite_support'/('source_'+key)
    old=read(prior/'input_freeze.json');new=read(finite/'receipt.json')
    assert new['status']=='LEARNED_FINITE_BRANCH_RECONSTRUCTED'
    assert old['input_sha256']==new['input_sha256']
    for folder,binding in ((prior,old),(finite,new)):
        for name in ('surfaces.json','surface_support.npz','profiles.json'):
            assert digest(folder/name)==binding['files'][name]
    old_faces=load_faces(prior);new_faces=load_faces(finite)
    profiles=read(prior/'profiles.json');old_count=len(profiles)
    added=read(finite/'profiles.json');offset=len(old_faces)
    for p in added:p['face_indices']=[i+offset for i in p['face_indices']]
    faces=old_faces+new_faces;profiles.extend(added)
    for p in profiles:
        assert p['length_mm']>0 and all(0<=i<len(faces) for i in p['face_indices'])
    assert all(f['graph_membership_uses_full_original_support'] for f in new_faces)
    with np.load(finite/'finite_mesh/original_surface_support.npz') as full:
        for f in new_faces:assert np.array_equal(f['point_ids'],full[f['full_support_key']])
    root=BASE/'legacy_runtime'/('source_'+key);folder=root/'run185'/('source_'+key)
    folder.mkdir(parents=True,exist_ok=False)
    save_graph_faces(folder,faces);write(folder/'profiles.json',profiles)
    write(folder/'input_freeze.json',dict(input_sha256=old['input_sha256'],files={n:digest(folder/n) for n in ('surfaces.json','surface_support.npz','profiles.json')},original_point_support_preserved=True,GT_reads=0,STEP_reads=0))
    case=read(BASE/'run26_merged_point_features'/('source_'+key)/'receipt.json')
    (root/'run132').mkdir()
    (root/'run132'/('source_'+key)).symlink_to(prior,target_is_directory=True)
    (root/'run98').mkdir()
    write(root/'run98/local_cohort.json',dict(cases=[dict(source_key=key,role='TRAIN',input_file=str(ROOT/case['input_file']),input_sha256=case['input_sha256'],original_points=case['original_points'])]))
    write(folder/'receipt.json',dict(status='ORIGINAL_JOINT_UNION_RECONSTRUCTED',source_key=key,faces=len(faces),retained_profiles=old_count,native_profiles=len(added),profiles=len(profiles),historical_parity_proven=False,weld_accuracy_measured=False,source_sha256=digest(__file__),original_source_sha256=digest(ROOT/'weld_pipeline/sjg_native_joint_candidates_999.py')))
    print('JOINT_UNION_COMPLETE',key,len(faces),len(profiles),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True,choices=['21','BM79A_native_local','10','25','36']);run(ap.parse_args().key)
