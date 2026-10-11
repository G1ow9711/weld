"""Fit both observed V process topologies. All reported fit scores are labeled."""
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from training.labels import sha256
from weld_pipeline.candidate_frame_tree_model import frames
from weld_pipeline.contract_child_evaluation import aggregate,score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.plate_candidate_model import predict_vertical,VERSION
from weld_pipeline.point_plate_context import contact_descriptor,endpoint_descriptor
from weld_pipeline.train_candidate_tree import supervision,mapped_references


def main():
    root=Path('weld_pipeline/data/full_available_fit_v1')
    split=json.loads((root/'split.json').read_text('utf8'))
    out=Path('weld_pipeline/runs/contract_full_fit_plate_heads_seed129')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'heads.joblib').exists():
        raise ValueError('Existing FIT heads')
    train={n:load_piece(split['piece_roots'][n],verify=True) for n in split['train']}
    arrays={}
    for n in split['train']+split['validation']:
        a=np.load(Path(split['structure_roots'][n])/(n+'.npz'),allow_pickle=False)
        arrays[n]=(a['structural'],a['process'])
    original=[n for n in split['train'] if '__' not in n]
    def source(n):
        return next((r for r in original if n==r or n.startswith(r+'__') or n.startswith(r+'_f')),n)
    rows=[{**r,'name':n} for n,p in train.items() for r in supervision(p)]
    counts=Counter((source(r['name']),r['foreground']) for r in rows)
    groups=Counter(y for n,y in counts)
    xf,yf,weights,xe,ye,ew=[],[],[],[],[],[]
    for r in rows:
        seq=train[r['name']]['sequences'][r['sequence']]
        structural=arrays[r['name']][0][r['sequence']]
        weight=1./(16*counts[source(r['name']),r['foreground']]*groups[r['foreground']])
        for transformed,reverse in frames(seq):
            xf.append(np.r_[structural,contact_descriptor(transformed)]);yf.append(r['foreground']);weights.append(weight)
            if r['foreground']:
                xe.append(endpoint_descriptor(transformed));ye.append(r['margins'][::-1] if reverse else r['margins']);ew.append(weight)
    side_rows=[]
    for n,p in train.items():
        structural,process=arrays[n]
        for i,refs in mapped_references(p).items():
            if len(refs)!=1:
                continue
            seq=p['sequences'][i]
            y=refs[0]['metadata'].get('IsSplitedForDualRobot (Any)')
            if y is None:
                continue
            length=seq['sample_step_mm']*(len(seq['xyz'])-1)
            side_rows.append((n,int(bool(y)),np.r_[structural[i],process[i],length/1000.,seq['type']=='vertical']))
    sc=Counter((source(n),y) for n,y,x in side_rows)
    sg=Counter(y for n,y in sc)
    sw=[1./(sc[source(n),y]*sg[y]) for n,y,x in side_rows]
    fractions={}
    for c in sorted({r['segment_count'] for r in rows if r['foreground']}):
        fractions[str(c)]=np.median(np.array([r['fractions'] for r in rows if r['foreground'] and r['segment_count']==c]),axis=0).tolist()
    if set(fractions)!={'1','3'}:
        raise ValueError('Both observed topologies must be supervised')
    if any(r['segment_count']!=(3 if r['split_flag'] else 1) for r in rows if r['foreground']):
        raise ValueError('Observed flag/topology relationship differs')
    paths=[__file__,'weld_pipeline/plate_candidate_model.py','weld_pipeline/point_plate_context.py',
           'weld_pipeline/candidate_frame_tree_model.py','weld_pipeline/candidate_tree_model.py',
           'weld_pipeline/sequence_augmentation.py','weld_pipeline/vertical_frame_augmentation.py',
           'weld_pipeline/train_candidate_tree.py','weld_pipeline/contract_child_evaluation.py']
    config={'version':VERSION,'seed':129,'trees':512,'role':'FULL_AVAILABLE_FIT_NOT_INDEPENDENT',
            'train':split['train'],'unseen_check':split['validation'],'split_sha256':sha256(root/'split.json'),
            'sources':{p:sha256(p) for p in paths},'threshold':.5,'split_fractions':fractions,
            'input_policy':'XYZ-derived structure, local distance profiles and physical process context; no names, identifiers, STEP or runtime labels.',
            'data_hashes':{n:sha256(Path(split['piece_roots'][n])/'manifest.json') for n in split['train']+split['validation']},
            'structural_hashes':{n:sha256(Path(split['structure_roots'][n])/(n+'.npz')) for n in split['train']+split['validation']},
            'process_scope':'TRAIN observed 1/3 topology relation only; cannot prove a universal policy.',
            'full_contract_pass':False}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for p in paths:
        shutil.copy2(p,archive/Path(p).name)
    started=time.perf_counter()
    foreground=ExtraTreesClassifier(n_estimators=512,max_features=.8,n_jobs=4,random_state=129)
    endpoints=ExtraTreesRegressor(n_estimators=512,max_features=.5,n_jobs=4,random_state=129)
    side=ExtraTreesClassifier(n_estimators=512,max_features=1.,n_jobs=4,random_state=129)
    foreground.fit(np.stack(xf),np.array(yf),sample_weight=np.array(weights))
    endpoints.fit(np.stack(xe),np.array(ye),sample_weight=np.array(ew))
    side.fit(np.stack([x for n,y,x in side_rows]),np.array([y for n,y,x in side_rows]),sample_weight=np.array(sw))
    bundle={'foreground':foreground,'endpoints':endpoints,'side_split':side,'split_fractions':fractions}
    joblib.dump(bundle,out/'heads.joblib')
    for role,names in [('FIT',original),('UNSEEN_CHECK',split['validation'])]:
        scores={}
        for n in names:
            p=train[n] if n in train else load_piece(split['piece_roots'][n],verify=True)
            pred,diagnostic=predict_vertical(bundle,p['sequences'],*arrays[n])
            score=score_children(pred,[w for w in p['truth'] if w['type']=='vertical'])
            scores[n]=score
            (out/(role+'_'+n+'.json')).write_text(json.dumps({'predictions':pred,'diagnostic':diagnostic,'score':score},indent=2),encoding='utf8')
            if score['truth_count'] or pred:
                print(role,n,'V',score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)
        summary=aggregate(scores);summary.update(role=role,independent_vertical_acceptance=False,full_contract_pass=False)
        (out/(role+'_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')
    (out/'complete.json').write_text(json.dumps({'elapsed_seconds':time.perf_counter()-started,
        'head_sha256':sha256(out/'heads.joblib'),'role':config['role'],'full_contract_pass':False},indent=2),encoding='utf8')
    print('HEADS_129_COMPLETE',round(time.perf_counter()-started,2),flush=True)


if __name__=='__main__':
    main()
