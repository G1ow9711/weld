"""Evaluate the frozen TRAIN-calibrated decoder on development assemblies."""
import argparse,json
from pathlib import Path
import torch
from training.labels import sha256
from weld_pipeline.axial_context_cache import load_piece as load_curved_piece,STORAGE_FORMAT
from weld_pipeline.mapped_cache import load_piece as load_mapped_piece
from weld_pipeline.model_registry_axial_branch import model_from_checkpoint
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.axial_context_features import infer_context_probabilities as infer_probabilities
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.event_gap_config import load_interval_config
from weld_pipeline.evaluation import score_welds
from weld_pipeline.curve_geometry_metrics import check_matched_curves
from weld_pipeline.partial_evaluation import score_partial_welds


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--config',required=True);ap.add_argument('--outdir',required=True)
    ap.add_argument('--data',default='weld_pipeline/data/mixed_surface_v1')
    ap.add_argument('--extra-training-groups',nargs='*',default=[])
    a=ap.parse_args();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);root=Path(a.data)
    parameters,config=load_interval_config(a.config,a.checkpoint);torch.set_num_threads(4)
    with Path(a.checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    historical_sources=verify_cached_inference_sources(state,a.checkpoint)
    model=model_from_checkpoint(state).cuda().eval();split=json.loads((root/'split.json').read_text(encoding='utf-8'));records=[]
    load_piece=load_curved_piece if split.get('storage_format')==STORAGE_FORMAT else load_mapped_piece
    augmented=[v['name'] for v in split.get('augmentation_policy',{}).get('variants',[])]
    original_train=split.get('augmentation_policy',{}).get('original_train',split['train'])
    if not set(a.extra_training_groups)<=set(original_train):raise ValueError('Extra diagnostics must be original TRAIN groups')
    groups=list(dict.fromkeys(split['validation']+(['TD2-LB8A'] if 'TD2-LB8A' in split['train'] else [])+augmented+a.extra_training_groups))
    for name in groups:
        folder=(root/split.get('piece_roots',{}).get(name,name)).resolve()
        if sha256(folder/'manifest.json')!=state['data_hashes'][name]:raise ValueError('Cache manifest differs')
        piece=load_piece(folder);prob=infer_probabilities(model,piece['sequences'],halo=getattr(model,'inference_halo',256))
        pred=decode_intervals(piece['sequences'],prob,**parameters)
        partial=piece['record'].get('target_policy')=='all_channel_minus_one_is_unknown'
        if partial:
            if name in split['validation']:raise ValueError('Partial annotation cannot be used for whole-piece development acceptance')
            score=score_partial_welds(pred,piece['truth'])
        else:
            score=score_welds(pred,piece['truth'])
            score['curve_geometry']=check_matched_curves(pred,piece['truth'],score)
            score['role']='development' if name in split['validation'] else ('training_augmentation' if name in augmented else 'training_fit')
        score['by_type']={}
        for typ in ['horizontal','vertical']:
            true_type=[w for w in piece['truth'] if w['type']==typ]
            if true_type:
                typed=(score_partial_welds if partial else score_welds)([w for w in pred if w['type']==typ],true_type)
                score['by_type'][typ]={k:typed[k] for k in ['valid_welds','truth_count','prediction_count','false_or_duplicate_predictions']}
        (out/(name+'_score.json')).write_text(json.dumps(score,indent=2),encoding='utf-8')
        (out/(name+'_predictions.json')).write_text(json.dumps(pred,indent=2),encoding='utf-8')
        row={'group':name,'role':score['role'],'valid':score['valid_welds'],'truth':score['truth_count'],
             'predictions':len(pred),'false':score['false_or_duplicate_predictions'],'gates_pass':score['all_gates_pass'],
             'curve_checked_valid':score['curve_geometry']['valid_welds_after_curve_check']}
        records.append(row);print(json.dumps(row),flush=True)
    report={'checkpoint_sha256':sha256(a.checkpoint),'decoder_config_sha256':sha256(a.config),'parameters':parameters,'groups':records,
            'scope':'One shared model and TRAIN-calibrated decoder; existing development pieces are not blind acceptance.',
            'data_root':str(root),'historical_entrypoints':historical_sources,
            'sources':{str(p):sha256(p) for p in [Path(__file__),Path('weld_pipeline/event_gap_config.py'),Path('weld_pipeline/event_gap_decoder.py'),
                Path('weld_pipeline/evaluation.py'),Path('weld_pipeline/curve_geometry_metrics.py'),Path('weld_pipeline/checkpoint_sources.py'),Path('weld_pipeline/partial_evaluation.py')]}}
    (out/'receipt.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
