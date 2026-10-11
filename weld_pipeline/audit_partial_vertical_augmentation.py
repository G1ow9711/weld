"""Audit a short rigid TRAIN variant with explicit unknown-context labels."""
import argparse,copy,json,time,shutil
from pathlib import Path
import numpy as np
from training.labels import sha256
from training.dataio import read_xyz_pcd
from weld_pipeline.fast_mixed_sequences import make_fast_sequences,append_fast_surface_features
from weld_pipeline.typed_sequences import training_targets
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.evaluation import score_welds
from weld_pipeline.curved_cache import write_piece
from weld_pipeline.partial_supervision import clean_targets,restrict_supervision
from weld_pipeline.audit_short_vertical_augmentation import process_free
from weld_pipeline.mixed_sequences import FEATURE_VERSION


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='206-TT1A');ap.add_argument('--number',default='f135')
    ap.add_argument('--outdir',default='weld_pipeline/reports/partial_vertical_augmentation');a=ap.parse_args()
    root=Path('weld_pipeline/data/mixed_surface_v1');split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    if a.group not in split['train']:raise ValueError('TRAIN parents only')
    folder=root/a.group;manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    truth=copy.deepcopy(manifest['truth']);indices=[i for i,w in enumerate(truth) if w['metadata']['WeldNumber (f/v + Name number)']==a.number]
    if len(indices)!=1:raise ValueError('Unknown focus weld')
    wi=indices[0];focus=truth[wi]
    if not process_free(focus) or focus['type']!='horizontal':raise ValueError('Process-free horizontal seam required')
    endpoints=np.array(focus['segments'][0]['points'])[[0,-1]];u=endpoints[1]-endpoints[0];length=np.linalg.norm(u);u/=length
    if not 60<=length<=250 or abs(u[2])>1e-8:raise ValueError('Expected short horizontal seam')
    rotation=np.stack([np.cross([0.,0.,1.],u),np.array([0.,0.,1.]),u])
    if not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-10) or not np.isclose(np.linalg.det(rotation),1.):raise ValueError('Not a rigid rotation')
    source=Path(manifest['record']['input'])
    if sha256(source)!=manifest['record']['pcd_sha256']:raise ValueError('Source changed')
    xyz=read_xyz_pcd(source)*1000;rotated=xyz@rotation.T;translation=np.array([200.,200.,200.])-rotated.min(0);rotated+=translation
    for w in truth:
        for segment in w['segments']:segment['points']=(np.array(segment['points'])@rotation.T+translation).tolist()
        direction=np.array(w['segments'][-1]['points'][-1])-w['segments'][0]['points'][0]
        w['type']='vertical' if abs(direction[2])/np.linalg.norm(direction)>.999999 else 'horizontal'
    start=time.perf_counter();sequences,_=make_fast_sequences(rotated);sequences,_=append_fast_surface_features(rotated,sequences)
    _,mapping=training_targets(sequences,[focus])
    out=Path(a.outdir)/(a.group+'_'+a.number);out.mkdir(parents=True,exist_ok=True)
    report={'source_group':a.group,'source_weld':a.number,'role':'training_partial_augmentation',
            'source_manifest_sha256':sha256(folder/'manifest.json'),'source_pcd_sha256':sha256(source),
            'rotation':rotation.tolist(),'translation_mm':translation.tolist(),
            'scope':'Unknown-context targets are -1 in every channel and MUST be masked in sampling/mining/loss. Not in any active training split.',
            'source_hashes':{str(Path(__file__)):sha256(Path(__file__)),'weld_pipeline/partial_supervision.py':sha256('weld_pipeline/partial_supervision.py')}}
    if not mapping['mapped_to_sequences']:report['status']='NO_GEOMETRY_CANDIDATE'
    else:
        seq=sequences[mapping['weld_to_sequence'][0]];targets,_=training_targets([seq],[focus]);other=[]
        q=seq['xyz'];direction=seq['direction'];position=np.arange(len(q))*seq['sample_step_mm']
        bounds=(np.array(focus['segments'][0]['points'])[[0,-1]]-q[0])@direction
        for i,w in enumerate(truth):
            if i==wi:continue
            points=np.array([p for s in w['segments'] for p in s['points']]);distance=np.linalg.norm(np.cross(points-q[0],direction),axis=1)
            if distance.max()<=.5:
                p=(points-q[0])@direction;other.append((float(p.min()),float(p.max())))
        try:target=restrict_supervision(targets[0],position,bounds,other)
        except ValueError as error:report.update(status='ENDPOINT_SUPERVISION_OVERLAP',reason=str(error))
        else:
            clean,valid=clean_targets(target);pred=decode_curved([seq],[clean]);score=score_welds(pred,[focus])
            ready=score['valid_welds']==score['prediction_count']==1
            report.update(status='MASKED_REPRESENTATION_READY' if ready else 'REPRESENTATION_GAP',
                supervised_points=int(valid.sum()),unknown_points=int((valid==0).sum()),oracle_score=score,
                length_mm=float(length),supervised_endpoints_mm=bounds.tolist())
            if ready:
                name=a.group+'_'+a.number+'__partial_vertical'
                record={'name':name,'source_group':a.group,'input':str(source),'pcd_sha256':sha256(source),
                        'feature_version':FEATURE_VERSION,'target_policy':'all_channel_minus_one_is_unknown',
                        'source_geometry_mapping':mapping,
                        'mapping':{'total_reference_welds':1,'mapped_to_sequences':1,'weld_to_sequence':{0:0},'unmapped':[]}}
                report['elapsed_seconds']=time.perf_counter()-start
                (out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                write_piece(out/'candidate_cache',{'record':record,'sequences':[seq],'targets':[target],'truth':[focus]},
                    {'audit_sha256':sha256(out/'receipt.json'),'source_manifest_sha256':sha256(folder/'manifest.json'),
                     'target_policy':'all_channel_minus_one_is_unknown','scope':'No active trainer may consume this until mask support is implemented and tested.'})
    report.setdefault('elapsed_seconds',time.perf_counter()-start)
    (out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    snapshots=out/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for path in report['source_hashes']:shutil.copy2(path,snapshots/Path(path).name)
    print(json.dumps({k:v for k,v in report.items() if k in ['status','length_mm','supervised_points','unknown_points','reason','elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
