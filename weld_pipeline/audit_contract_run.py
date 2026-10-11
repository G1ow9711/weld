"""Full cached development/training-fit and separate raw-PCD evaluation."""
import argparse
import json
import time
from pathlib import Path
import torch
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.predict_contract import factory,predict
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.contract_child_evaluation import score_children,aggregate
from weld_pipeline.evaluation import score_welds


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--raw',action='store_true');ap.add_argument('--all-originals',action='store_true')
    args=ap.parse_args();run=Path(args.run);checkpoint=run/'best.pt'
    split=json.loads(Path('weld_pipeline/data/contract_child_v1/split.json').read_text('utf8'))
    with checkpoint.open('rb') as f:state=torch.load(f,map_location='cpu')
    verify_cached_inference_sources(state,checkpoint);torch.set_num_threads(4)
    model=factory(state).cuda().eval()
    out=run/'contract_final_evaluation';out.mkdir(exist_ok=True)
    names=split['validation']+([n for n in split['train'] if '__' not in n] if args.all_originals else [])
    groups=[];dev_scores={};train_scores={};raw_scores={}
    for name in names:
        piece=load_piece(split['piece_roots'][name],verify=True)
        role='development' if name in split['validation'] else 'training_fit'
        pred=decode_intervals(piece['sequences'],infer_probabilities(model,piece['sequences']),
            threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.)
        score=score_children(pred,piece['truth']);legacy=score_welds(pred,piece['truth'])
        (out/(name+'_predictions.json')).write_text(json.dumps(pred,indent=2),encoding='utf8')
        (out/(name+'_score.json')).write_text(json.dumps(score,indent=2),encoding='utf8')
        (dev_scores if role=='development' else train_scores)[name]=score
        if args.raw and (role=='development' or name=='TD2-LB8A'):
            raw_pred,receipt=predict(piece['record']['input'],checkpoint,run/('raw_'+name))
            raw_score=score_children(raw_pred,piece['truth'])
            if raw_pred!=pred:raise ValueError('Raw versus cached prediction mismatch: '+name)
            raw_scores[name]=raw_score
            (run/('raw_'+name)/'geometry_score.json').write_text(json.dumps(raw_score,indent=2),encoding='utf8')
        row={'name':name,'role':role,'valid_children':score['valid_children'],'truth_children':score['truth_count'],
            'false_children':score['false_predictions'],'parent_valid':legacy['valid_welds'],'parent_truth':legacy['truth_count']}
        groups.append(row);print('GROUP',json.dumps(row),flush=True)
    audit={'status':'NOT_ACHIEVED','checkpoint_sha256':sha256(checkpoint),'selected_epoch':state['epoch'],
        'split_sha256':sha256('weld_pipeline/data/contract_child_v1/split.json'),
        'development':aggregate(dev_scores),'training_fit':aggregate(train_scores) if train_scores else None,
        'raw_validated':list(raw_scores),'raw_cache_predictions_equal':bool(raw_scores),
        'groups':groups,'attributes_status':'NOT_RUN_NO_COMPLETE_REVIEWED_TRUTH_OR_ATTRIBUTE_ESTIMATORS',
        'side_truth_status':'NOT_AVAILABLE','formal_independent_test':'NOT_AVAILABLE',
        'independent_curve_validation':'NOT_AVAILABLE','timing_gate':'DEFERRED_BY_USER_NOT_PASSED'}
    (run/'contract_goal_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print('AUDIT',json.dumps({k:v for k,v in audit.items() if k in ['status','selected_epoch','development','raw_validated']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
