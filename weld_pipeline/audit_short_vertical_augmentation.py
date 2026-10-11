"""Rotate TRAIN geometry, retaining only fully labeled process-free sequences."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from training.labels import sha256
from training.dataio import read_xyz_pcd
from weld_pipeline.fast_mixed_sequences import make_fast_sequences,append_fast_surface_features
from weld_pipeline.typed_sequences import training_targets
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.evaluation import score_welds
from weld_pipeline.curved_cache import write_piece
from weld_pipeline.mixed_sequences import FEATURE_VERSION

FLAGS=['IsWorkRangeSplit (Any)','IsSplitedForDualRobot (Any)','HasEave (Any)','WorkRangeFrontSplit','WorkRangeBackSplit']


def process_free(weld):
    return len(weld['segments'])==1 and weld['segments'][0]['kind']=='line' and not any(weld['metadata'].get(k,False) for k in FLAGS)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='206-TT1A');ap.add_argument('--number',default='f135')
    ap.add_argument('--outdir',default='weld_pipeline/reports/short_vertical_augmentation');a=ap.parse_args()
    root=Path('weld_pipeline/data/mixed_surface_v1');split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    if a.group not in split['train']:raise ValueError('TRAIN parents only')
    folder=root/a.group;manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    source_truth=manifest['truth'];chosen=[w for w in source_truth if w['metadata']['WeldNumber (f/v + Name number)']==a.number]
    if len(chosen)!=1 or not process_free(chosen[0]) or chosen[0]['type']!='horizontal':raise ValueError('Target must be process-free single horizontal seam')
    original_endpoints=np.array(chosen[0]['segments'][0]['points'])[[0,-1]]
    u=original_endpoints[1]-original_endpoints[0];length=np.linalg.norm(u);u/=length
    if not 60<=length<=250 or abs(u[2])>1e-8:raise ValueError('Expected short horizontal seam')
    rotation=np.stack([np.cross([0.,0.,1.],u),np.array([0.,0.,1.]),u])
    if not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-10) or not np.isclose(np.linalg.det(rotation),1.):raise ValueError('Invalid rotation')
    source=Path(manifest['record']['input'])
    if sha256(source)!=manifest['record']['pcd_sha256']:raise ValueError('Source cloud changed')
    xyz=read_xyz_pcd(source)*1000;rotated=xyz@rotation.T
    translation=np.array([200.,200.,200.])-rotated.min(0);rotated+=translation
    transformed=copy.deepcopy(source_truth)
    for w in transformed:
        for segment in w['segments']:segment['points']=(np.array(segment['points'])@rotation.T+translation).tolist()
        direction=np.array(w['segments'][-1]['points'][-1])-w['segments'][0]['points'][0]
        w['type']='vertical' if abs(direction[2])/np.linalg.norm(direction)>.999999 else 'horizontal'
    start=time.perf_counter();sequences,geometry=make_fast_sequences(rotated)
    sequences,surface=append_fast_surface_features(rotated,sequences)
    selected=[];included=set();rejected=[]
    # Every reference touching a retained sequence must be fully represented and
    # process-free. A missing annotation on a retained sequence is never a negative.
    for si,seq in enumerate(sequences):
        if seq['type']!='vertical':continue
        q=seq['xyz'];direction=seq['direction'];limit=(len(q)-1)*seq['sample_step_mm'];touching=[];unsafe=[]
        for wi,w in enumerate(transformed):
            points=np.array([p for s in w['segments'] for p in s['points']])
            position=(points-q[0])@direction
            if position.max()<0 or position.min()>limit:continue
            distance=np.linalg.norm(np.cross(points-q[0],direction),axis=1)
            if distance.min()>.5:continue
            if distance.max()>.5:
                # A crossing seam touches only at its corner, not along this path.
                if (distance<=.5).sum()<=2:continue
                unsafe.append(wi);continue
            if not process_free(w) or position.min()<-1 or position.max()>limit+1:unsafe.append(wi)
            else:touching.append(wi)
        if not any(transformed[i]['metadata']['WeldNumber (f/v + Name number)']==a.number for i in touching):continue
        if unsafe:rejected.append({'sequence':si,'unsafe_truth_indices':unsafe});continue
        selected.append(seq);included.update(touching)
    truth=[transformed[i] for i in sorted(included)]
    out=Path(a.outdir)/(a.group+'_'+a.number);out.mkdir(parents=True,exist_ok=True)
    if truth:
        targets,mapping=training_targets(selected,truth);pred=decode_curved(selected,targets);score=score_welds(pred,truth)
        ready=score['valid_welds']==len(truth) and score['prediction_count']==len(truth)
    else:targets=[];mapping={};score={};ready=False
    report={'status':'REPRESENTATION_READY' if ready else 'NO_SAFE_COMPLETE_SEQUENCE','source_group':a.group,'source_weld':a.number,
        'source_manifest_sha256':sha256(folder/'manifest.json'),'source_pcd_sha256':sha256(source),
        'rotation':rotation.tolist(),'translation_mm':translation.tolist(),'retained_sequences':len(selected),'retained_truth':len(truth),
        'retained_numbers':[w['metadata']['WeldNumber (f/v + Name number)'] for w in truth],
        'rejected':rejected,'mapping':mapping,'oracle_score':score,'elapsed_seconds':time.perf_counter()-start,
        'scope':'TRAIN sequence augmentation only. The complete original cloud was transformed; loss is allowed only on complete process-free reference sequences. Not an independently annotated new assembly.',
        'source_hashes':{str(Path(__file__)):sha256(Path(__file__))}}
    (out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if ready:
        name=a.group+'_'+a.number+'__short_vertical'
        record={'name':name,'source_group':a.group,'augmentation':'rigid_pose_selected_complete_sequences',
                'input':str(source),'pcd_sha256':sha256(source),'feature_version':FEATURE_VERSION,'mapping':mapping}
        write_piece(out/'candidate_cache',{'record':record,'sequences':selected,'targets':targets,'truth':truth},
            {'audit_sha256':sha256(out/'receipt.json'),'source_manifest_sha256':sha256(folder/'manifest.json'),
             'scope':'TRAIN-only partial sequence view; excluded sequences carry no negative supervision.'})
    print(json.dumps({k:report[k] for k in ['status','source_group','source_weld','retained_sequences','retained_truth','retained_numbers','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
