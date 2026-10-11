"""TRAIN-only rigid pose augmentation feasibility; not an independent workpiece."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.fast_mixed_sequences import make_fast_mixed_sequences
from weld_pipeline.mixed_sequences import mixed_training_targets,FEATURE_VERSION
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.evaluation import score_welds
from weld_pipeline.curved_cache import write_piece


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='223-FR83A')
    ap.add_argument('--outdir',default='weld_pipeline/reports/rigid_vertical_augmentation');a=ap.parse_args()
    root=Path('weld_pipeline/data/mixed_surface_v1');split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    if a.group not in split['train']:raise ValueError('Only original TRAIN pieces may be augmented')
    folder=root/a.group;manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));truth=copy.deepcopy(manifest['truth'])
    for w in truth:
        if w['type']!='horizontal' or len(w['segments'])!=1 or w['segments'][0]['kind']!='line':raise ValueError('Requires single straight horizontal welds')
        if any(w['metadata'].get(k,False) for k in ['IsWorkRangeSplit (Any)','IsSplitedForDualRobot (Any)','HasEave (Any)','WorkRangeFrontSplit','WorkRangeBackSplit']):
            raise ValueError('Process-dependent cuts/eaves must be separately reviewed before pose augmentation')
    source=Path(manifest['record']['input'])
    if sha256(source)!=manifest['record']['pcd_sha256']:raise ValueError('PCD differs')
    xyz=read_xyz_pcd(source)*1000;first=np.array(truth[0]['segments'][0]['points'])
    u=first[-1]-first[0];u=u/np.linalg.norm(u)
    if abs(u[2])>1e-8:raise ValueError('Source seam not horizontal')
    z=np.array([0.,0.,1.]);v=np.cross(z,u);rotation=np.stack([v,z,u])
    if not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-10) or not np.isclose(np.linalg.det(rotation),1.):raise ValueError('Improper rotation')
    rotated=xyz@rotation.T;translation=np.array([200.,200.,200.])-rotated.min(0);rotated+=translation
    inverse_error=float(np.abs((rotated-translation)@rotation-xyz).max())
    length_errors=[]
    for w in truth:
        old=np.array(w['segments'][0]['points']);new=old@rotation.T+translation
        direction=new[-1]-new[0];direction/=np.linalg.norm(direction)
        if abs(direction[2])<.999999:raise ValueError('Not all source welds transform to vertical')
        length_errors.append(abs(float(np.linalg.norm(old[-1]-old[0])-np.linalg.norm(new[-1]-new[0]))))
        w['segments'][0]['points']=new.tolist();w['type']='vertical'
        w['metadata']['augmentation_original_type']='horizontal'
    out=Path(a.outdir)/a.group;out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    # Features consume rotated XYZ only; reference geometry is attached afterward.
    sequences,geometry=make_fast_mixed_sequences(rotated)
    targets,mapping=mixed_training_targets(sequences,truth)
    oracle=decode_curved(sequences,targets);score=score_welds(oracle,truth)
    ready=mapping['mapped_to_sequences']==len(truth) and score['valid_welds']==len(truth) and score['prediction_count']==len(truth)
    report={'status':'REPRESENTATION_READY' if ready else 'REPRESENTATION_GAP','source_group':a.group,'role':'training_augmentation',
        'scope':'Rigidly transformed geometry supervision only. Not a new independent assembly or model accuracy.',
        'constraints':'Source has no work-range/side split/eave flags and only single straight seams; all original segments preserved.',
        'source_pcd':str(source),'source_pcd_sha256':sha256(source),'source_manifest_sha256':sha256(folder/'manifest.json'),
        'rotation':rotation.tolist(),'translation_mm':translation.tolist(),'inverse_coordinate_max_error_mm':inverse_error,
        'max_weld_length_change_mm':max(length_errors),'mapping':mapping,'oracle_score':score,'geometry':geometry,
        'elapsed_seconds':time.perf_counter()-start,'source_hashes':{str(Path(__file__)):sha256(Path(__file__))}}
    (out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if ready:
        name=a.group+'__rigid_vertical'
        record={**manifest['record'],'name':name,'source_group':a.group,'augmentation':'rigid_pose_horizontal_to_vertical',
                'feature_version':FEATURE_VERSION,'mapping':mapping}
        piece={'record':record,'sequences':sequences,'targets':targets,'truth':truth}
        write_piece(out/'candidate_cache',piece,{'source_manifest_sha256':sha256(folder/'manifest.json'),'audit_sha256':sha256(out/'receipt.json'),
            'rotation':rotation.tolist(),'translation_mm':translation.tolist(),'scope':'TRAIN augmentation candidate; not in active split.'})
    print(json.dumps({k:report[k] for k in ['status','source_group','inverse_coordinate_max_error_mm','max_weld_length_change_mm','elapsed_seconds']}),flush=True)
    print('MAPPING',mapping['mapped_to_sequences'],len(truth),'ORACLE',score['valid_welds'],score['prediction_count'],flush=True)


if __name__=='__main__':main()
