"""Fixed-model input ablation; output geometry and training remain untouched."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from training.dataio import read_xyz_pcd
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.model_registry_branch import model_from_checkpoint
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.axial_context_features import append_axial_context,infer_context_probabilities
from weld_pipeline.interval_decoder import decode_intervals
from weld_pipeline.evaluation import score_welds


def main():
    ap=argparse.ArgumentParser()
    for key in ['checkpoint','data','outdir']:ap.add_argument('--'+key,required=True)
    ap.add_argument('--groups',nargs='+',required=True);args=ap.parse_args()
    checkpoint=Path(args.checkpoint)
    with checkpoint.open('rb') as f:state=torch.load(f,map_location='cpu')
    verify_cached_inference_sources(state,checkpoint)
    model=model_from_checkpoint(state).cuda().eval();torch.set_num_threads(4)
    root=Path(args.data);split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    rows=[];out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    for name in args.groups:
        if name not in split['augmentation_policy']['original_train']+split['validation']:
            raise ValueError('This audit handles original-coordinate clouds only')
        folder=(root/split.get('piece_roots',{}).get(name,name)).resolve()
        if sha256(folder/'manifest.json')!=state['data_hashes'][name]:raise ValueError('Manifest changed')
        p=load_piece(folder);source=Path(p['record']['input'])
        if sha256(source)!=p['record']['pcd_sha256']:raise ValueError('Raw point cloud changed')
        sequences=[s for s in p['sequences'] if s['type']=='vertical']
        context,stats=append_axial_context(read_xyz_pcd(source)*1000,sequences)
        original=infer_probabilities(model,sequences);changed=infer_context_probabilities(model,context)
        truth=[w for w in p['truth'] if w['type']=='vertical'];scores=[]
        for label,prob in [('original',original),('physical_context',changed)]:
            pred=decode_intervals(sequences,prob,threshold=.5,weld_penalty=2.,mask_weight=.25)
            if truth:
                score=score_welds(pred,truth)
                compact={k:score[k] for k in ['valid_welds','truth_count','prediction_count','false_or_duplicate_predictions']}
            else:
                score={'truth_count':0,'prediction_count':len(pred),'valid_welds':None,
                       'false_or_duplicate_predictions':None,
                       'scope':'No labeled vertical welds. Report prediction count only; no recall or acceptance rate.'}
                compact=score
            scores.append({'condition':label,**compact})
            (out/(name+'_'+label+'_score.json')).write_text(json.dumps(score,indent=2),encoding='utf-8')
        delta=np.concatenate([abs(a-b) for a,b in zip(original,changed)])
        row={'group':name,'role':'development' if name in split['validation'] else 'training',
             'scores':scores,'mean_probability_change_per_head':delta.mean(0).tolist(),'context':stats,
             'manifest_sha256':sha256(folder/'manifest.json'),'raw_input_sha256':sha256(source)}
        rows.append(row);print(json.dumps(row),flush=True)
    files=[Path(__file__),*[Path('weld_pipeline')/n for n in ['axial_context_features.py','indexed_surface_features.py','interval_decoder.py','evaluation.py']]]
    report={'scope':'Frozen-model input ablation only, not a trained model or acceptance result. Original output extent, geometry and labels unchanged.',
            'checkpoint_sha256':sha256(checkpoint),'sources':{str(p):sha256(p) for p in files},'groups':rows}
    (out/'receipt.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
