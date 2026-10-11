"""Fit one shared decoder on training pieces, freeze, then score development pieces."""
import argparse,gc,itertools,json,shutil
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.mapped_cache import load_piece
from weld_pipeline.model_registry_research import model_from_checkpoint
from weld_pipeline.typed_contact_model import infer_probabilities,decode_typed
from weld_pipeline.constrained_decoder import decode_constrained
from weld_pipeline.fast_counts import score_counts
from weld_pipeline.evaluation import score_welds


def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--outdir',required=True)
    args=ap.parse_args();root=Path(args.data);out=Path(args.outdir)
    if (out/'calibration.json').exists():raise ValueError('Use a new output folder')
    out.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    def folder(name):return root/split.get('piece_roots',{}).get(name,name)
    lengths=[]
    for name in split['train']:
        m=json.loads((folder(name)/'manifest.json').read_text(encoding='utf-8'))
        lengths.extend(float(np.linalg.norm(np.array(s['points'][-1])-s['points'][0]))
                       for w in m['truth'] if len(w['segments'])>1 for s in w['segments'])
    minimum=float(min(lengths))
    baseline={'endpoint_threshold':.5,'internal_threshold':.5,'minimum_segment_mm':1.5,'internal_distance_mm':4.}
    configs=[baseline]
    for end,cut,margin in itertools.product([.3,.5,.7],[.3,.5,.7],[1.5,minimum/2,max(1.5,minimum-3.)]):
        cfg={'endpoint_threshold':end,'internal_threshold':cut,'minimum_segment_mm':margin,'internal_distance_mm':max(4.,margin)}
        if cfg not in configs:configs.append(cfg)
    with Path(args.checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    torch.set_num_threads(4);model=model_from_checkpoint(state).cuda().eval()
    trials=[{'config':c,'training':{}} for c in configs]
    sources=[Path(__file__),Path('weld_pipeline/constrained_decoder.py'),Path('weld_pipeline/fast_counts.py'),Path('weld_pipeline/evaluation.py')]
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in sources:shutil.copyfile(p,snapshot/p.name)
    source_hashes={str(p):sha256(p) for p in sources}
    for name in split['train']:
        p=load_piece(folder(name));probs=infer_probabilities(model,p['sequences'])
        for trial in trials:
            pred=decode_constrained(p['sequences'],probs,**trial['config'])
            trial['training'][name]=score_counts(pred,p['truth'])
        # Verify the accelerated metric and the original decoder on real data.
        original=decode_typed(p['sequences'],probs);full=score_welds(original,p['truth'])
        assert all(full[k]==v for k,v in trials[0]['training'][name].items()),name
        print('TRAIN_CALIBRATION',name,'baseline',full['valid_welds'],'/',full['truth_count'],flush=True)
        write(out/'training_trials.partial.json',trials)
        del p,probs;gc.collect()
    for trial in trials:
        scores=list(trial['training'].values())
        trial['macro_f1']=float(np.mean([2*s['valid_welds']/(s['truth_count']+s['prediction_count']) for s in scores]))
        trial['valid']=sum(s['valid_welds'] for s in scores)
        trial['false']=sum(s['false_or_duplicate_predictions'] for s in scores)
    selected=max(range(len(trials)),key=lambda i:(trials[i]['macro_f1'],trials[i]['valid'],-trials[i]['false']))
    report={'scope':'Training-only decoder calibration followed by fixed development validation. Not blind acceptance.',
        'checkpoint_sha256':sha256(args.checkpoint),'source_hashes':source_hashes,
        'minimum_training_multisegment_length_mm':minimum,'selected_trial':selected,
        'selected_config':configs[selected],'selection_rule':'training macro strict weld F1, then valid count, then fewer false; earliest on exact tie',
        'trials':trials,'validation':{}}
    # The selected global configuration is committed BEFORE reading validation arrays/truth.
    write(out/'calibration.json',report)
    print('SELECTED',selected,configs[selected],flush=True)
    for name in split['validation']:
        p=load_piece(folder(name));probs=infer_probabilities(model,p['sequences'])
        pred=decode_constrained(p['sequences'],probs,**configs[selected])
        score=score_welds(pred,p['truth']);fast=score_counts(pred,p['truth'])
        assert all(score[k]==v for k,v in fast.items())
        report['validation'][name]={k:v for k,v in score.items() if k not in ['per_truth','unmatched_prediction_indices']}
        write(out/(name+'_predictions.json'),pred);write(out/(name+'_score.json'),score)
        print('VALIDATION',name,fast,flush=True)
        del p,probs;gc.collect()
    report['complete']=True
    write(out/'calibration.json',report)


if __name__=='__main__':main()
