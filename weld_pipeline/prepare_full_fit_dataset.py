"""Separate full available-label fit cohort and untouched two-piece check cohort."""
import copy
import json
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece,write_piece
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.evaluation import score_welds
from weld_pipeline.mixed_sequences import make_mixed_sequences,mixed_training_targets
from weld_pipeline.point_plate_context import PlateContext
from weld_pipeline.truth import load_truth


def main():
    parent=Path('weld_pipeline/data/point_plate_context_v1')
    old=json.loads((parent/'split.json').read_text('utf8'))
    out=Path('weld_pipeline/data/full_available_fit_v1')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise ValueError('Frozen FIT dataset exists')
    split=copy.deepcopy(old)
    split.update(train=old['train']+old['validation'],validation=['232-TT1A','224-FR71A'],
        status='FULL_AVAILABLE_LABEL_FIT_NOT_ORIGINAL_DEV',
        original_development_now_in_train=old['validation'],original_split_sha256=sha256(parent/'split.json'),
        role_policy='25 original plus 16 masked augmented TRAIN views. Old DEV is FIT only. Two reserved pieces check unseen H; no unseen vertical/arc positive acceptance.',
        structure_roots={n:str(parent) for n in old['train']+old['validation']})
    audits=[]
    for name in split['validation']:
        started=time.perf_counter()
        folder=Path('DataSet/9月20日')/name
        pcd=next(folder.glob('*AllFaces.pcd'))
        xyz=read_xyz_pcd(pcd)*1000.
        sequences,geometry=make_mixed_sequences(xyz)
        truth,excluded=load_truth(folder)
        if excluded:
            raise ValueError('Check cohort reference incomplete: '+name)
        targets,mapping=mixed_training_targets(sequences,truth)
        oracle=score_welds(decode_curved(sequences,targets),truth)
        record={'name':name,'input':str(pcd),'pcd_sha256':sha256(pcd),
            'reference_hashes':{str(p):sha256(p) for p in [*folder.glob('*.xlsx'),*folder.glob('*welds.json')]},
            'feature_version':split['feature_version'],'feature_channels':84,'mapping':mapping,
            'representation_oracle':oracle,'geometry':geometry}
        destination=out/name
        write_piece(destination,{'sequences':sequences,'targets':targets,'truth':truth,'record':record},
                    {'sources':{p:sha256(p) for p in [__file__,'weld_pipeline/mixed_sequences.py']},'pcd_sha256':sha256(pcd)})
        context=PlateContext(xyz)
        a,b=[],[]
        for seq in sequences:
            structural,process,diagnostic=context.describe(seq)
            a.append(structural);b.append(process)
        np.savez_compressed(out/(name+'.npz'),structural=np.stack(a),process=np.stack(b))
        split['piece_roots'][name]=str(destination)
        split['structure_roots'][name]=str(out)
        audits.append({'name':name,'parents':len(truth),'segments':sum(len(w['segments']) for w in truth),
            'mapped':mapping['mapped_to_sequences'],'oracle_valid':oracle['valid_welds'],
            'oracle_false':oracle['false_or_duplicate_predictions'],'seconds':time.perf_counter()-started,
            'manifest_sha256':sha256(destination/'manifest.json'),'structural_sha256':sha256(out/(name+'.npz'))})
        print('UNSEEN_PREPARED',json.dumps(audits[-1]),flush=True)
        del xyz,context,sequences,targets
    (out/'split.json').write_text(json.dumps(split,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'complete.json').write_text(json.dumps({'status':'READY_FULL_FIT_WITH_UNSEEN_CHECK',
        'scope':split['role_policy'],'check_representation':audits,'split_sha256':sha256(out/'split.json'),
        'full_contract_pass':False},ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
