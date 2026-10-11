"""Score selected mixed-model groups without confusing training fit with testing."""
import argparse,json
from pathlib import Path
import torch
from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.model_registry_context import model_from_checkpoint
from weld_pipeline.evaluation import score_welds
from weld_pipeline.curve_geometry_metrics import check_matched_curves


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--groups',nargs='+',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args()
    root,out=Path(a.data),Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/'split.json').read_text(encoding='utf-8'));torch.set_num_threads(4)
    with Path(a.checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    for source,wanted in state['source_hashes'].items():
        if sha256(source)!=wanted:raise ValueError('Training source has changed: '+source)
    model=model_from_checkpoint(state).cuda().eval();records=[]
    for name in a.groups:
        if name not in split['train']+split['validation']:raise ValueError('Group is outside the frozen split')
        dest=root/name
        if sha256(dest/'manifest.json')!=state['data_hashes'][name]:raise ValueError('Dataset manifest differs')
        piece=load_piece(dest);pred=decode_curved(piece['sequences'],infer_probabilities(model,piece['sequences']))
        truth=piece['truth'];score=score_welds(pred,truth)
        score['curve_geometry']=check_matched_curves(pred,truth,score)
        score['by_geometry']={}
        for kind in ['line','arc']:
            gt=[w for w in truth if all(s['kind']==kind for s in w['segments'])]
            if gt:score['by_geometry'][kind]=score_welds([w for w in pred if all(s['kind']==kind for s in w['segments'])],gt)
        role='TRAINING_FIT_ONLY' if name in split['train'] else 'DEVELOPMENT_VALIDATION'
        score['dataset_role']=role
        (out/(name+'_score.json')).write_text(json.dumps(score,indent=2),encoding='utf-8')
        (out/(name+'_predictions.json')).write_text(json.dumps(pred,indent=2),encoding='utf-8')
        brief={'group':name,'role':role,'valid':score['valid_welds'],'truth':len(truth),'predictions':len(pred),
               'arc_valid':score['by_geometry'].get('arc',{}).get('valid_welds'),
               'curve_checked_valid':score['curve_geometry']['valid_welds_after_curve_check']}
        records.append(brief);print(json.dumps(brief),flush=True)
    (out/'receipt.json').write_text(json.dumps({'checkpoint_sha256':sha256(a.checkpoint),'groups':records,
        'scope':'Cached representation scoring; raw point-cloud inference is a separate check.',
        'diagnostic_sources':{str(p):sha256(p) for p in [Path(__file__),Path('weld_pipeline/curve_geometry_metrics.py')]}},indent=2),encoding='utf-8')


if __name__=='__main__':main()
