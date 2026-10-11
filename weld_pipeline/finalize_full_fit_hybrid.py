"""Package completed refits and report FIT separately from unseen checks."""
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.contract_child_evaluation import aggregate,score_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.full_fit_hybrid import MODEL_NAME,infer_from_sequences


def main():
    hroot=Path('weld_pipeline/runs/contract_full_fit_horizontal_seed128')
    vroot=Path('weld_pipeline/runs/contract_full_fit_plate_heads_seed129')
    if not (hroot/'complete.json').exists() or not (vroot/'complete.json').exists():
        raise ValueError('Training not complete')
    root=Path('weld_pipeline/data/full_available_fit_v1')
    split=json.loads((root/'split.json').read_text('utf8'))
    h=torch.load(hroot/'best.pt',map_location='cpu')
    verify_cached_inference_sources(h,hroot/'best.pt')
    heads=joblib.load(vroot/'heads.joblib')
    config=json.loads((vroot/'config.json').read_text('utf8'))
    if sha256(root/'split.json')!=config['split_sha256']:
        raise ValueError('FIT split changed')
    paths=['weld_pipeline/full_fit_hybrid.py','weld_pipeline/predict_full_fit_hybrid.py',
           'weld_pipeline/finalize_full_fit_hybrid.py','weld_pipeline/contract_export.py',
           'weld_pipeline/export.py','weld_pipeline/point_plate_attributes.py']
    sources={**config['sources'],**{p:sha256(p) for p in paths}}
    out=Path('weld_pipeline/runs/contract_full_fit_hybrid_v1')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():
        raise ValueError('Frozen combined checkpoint exists')
    state={'model_name':MODEL_NAME,'horizontal':h,'plate_heads':heads,'config':config,
           'source_hashes':sources,'fit_role':split['role_policy'],'full_contract_pass':False}
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for name,wanted in {**h['source_hashes'],**sources}.items():
        path=Path(name)
        if not path.exists() or sha256(path)!=wanted:
            path=hroot/'source_snapshot'/path.name
        if sha256(path)!=wanted:
            raise ValueError('Incomplete source archive: '+name)
        shutil.copy2(path,archive/path.name)
    torch.save(state,out/'best.pt')
    torch.set_num_threads(4)
    model=from_checkpoint(h).cuda().eval()
    roles={}
    for role,names in [('FIT',[n for n in split['train'] if '__' not in n]),('UNSEEN_CHECK',split['validation'])]:
        scores={}
        for n in names:
            if sha256(Path(split['piece_roots'][n])/'manifest.json')!=config['data_hashes'][n]:
                raise ValueError('Sequence source changed')
            a=np.load(Path(split['structure_roots'][n])/(n+'.npz'),allow_pickle=False)
            if sha256(Path(split['structure_roots'][n])/(n+'.npz'))!=config['structural_hashes'][n]:
                raise ValueError('Structural source changed')
            p=load_piece(split['piece_roots'][n],verify=True)
            pred,diagnostic=infer_from_sequences(state,model,p['sequences'],a['structural'],a['process'])
            score=score_children(pred,p['truth']);scores[n]=score
            (out/(role+'_'+n+'.json')).write_text(json.dumps({'predictions':pred,'diagnostic':diagnostic,'score':score},ensure_ascii=False,indent=2),encoding='utf8')
            print(role,n,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)
        summary=aggregate(scores)
        summary.update(role=role,formal_independent_acceptance=False,full_contract_pass=False)
        (out/(role+'_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')
        roles[role]=summary
    report={'status':'MODEL_TRAINED_CONTRACT_NOT_ACHIEVED','checkpoint':str(out/'best.pt'),
            'checkpoint_sha256':sha256(out/'best.pt'),'single_checkpoint':True,'horizontal_epoch':h['epoch'],
            'roles':roles,'runtime_step_forbidden':True,'raw_inference_verified':False,
            'missing_acceptance':['Independent 200-piece sealed test with vertical/curve positives',
                                  'Reviewed weld-side directed normals/thickness and remaining child attributes',
                                  'All required interface/export acceptance'],
            'speed_gate':'DEFERRED_BY_USER','baseline_115_replaced':False,'full_contract_pass':False}
    (out/'training_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('MODEL_PACKAGED',json.dumps({role:{k:v[k] for k in ['valid_children','truth_count','false_predictions','geometry_core_gates_pass']} for role,v in roles.items()}),flush=True)


if __name__=='__main__':
    main()
