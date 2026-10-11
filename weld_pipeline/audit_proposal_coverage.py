"""Post-training diagnosis of proposal recall; never used as inference input."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from weld_pipeline.mapped_cache import load_piece
from weld_pipeline.model_registry_research import model_from_checkpoint
from weld_pipeline.typed_contact_model import infer_probabilities,decode_typed
from weld_pipeline.evaluation import score_welds


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--outdir',required=True)
    args=ap.parse_args();root=Path(args.data);out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    with Path(args.checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    model=model_from_checkpoint(state).cuda().eval();thresholds=[.1,.2,.3,.5,.7]
    report={'scope':'Development diagnostic using reference locations AFTER model inference; not extraction accuracy or a new blind test.',
            'checkpoint_sha256':sha256(args.checkpoint),'epoch':state['epoch'],'groups':{}}
    torch.set_num_threads(4)
    for name in split['validation']:
        p=load_piece(root/split.get('piece_roots',{}).get(name,name))
        probabilities=infer_probabilities(model,p['sequences']);rows=[]
        for wi,w in enumerate(p['truth']):
            si=p['record']['mapping']['weld_to_sequence'].get(str(wi))
            if si is None:raise ValueError('Reference cannot be represented')
            seq=p['sequences'][si];prob=probabilities[si]
            points=np.array([w['segments'][0]['points'][0],w['segments'][-1]['points'][-1]])
            positions=(points-seq['xyz'][0])@seq['direction'];lo,hi=sorted(positions)
            t=np.arange(len(prob))*seq['sample_step_mm'];region=(t>=lo)&(t<=hi)
            near_start=np.abs(t-lo)<=3.;near_end=np.abs(t-hi)<=3.
            internals=[float((np.array(s['points'][-1])-seq['xyz'][0])@seq['direction']) for s in w['segments'][:-1]]
            row={'id':w['id'],'type':w['type'],'segments':len(w['segments']),
                 'start_peak_within_3mm':float(prob[near_start,1].max()),
                 'end_peak_within_3mm':float(prob[near_end,2].max()),
                 'foreground_mean_on_reference':float(prob[region,0].mean()),
                 'internal_peaks_within_3mm':[float(prob[np.abs(t-v)<=3.,3].max()) for v in internals]}
            rows.append(row)
        diagnostics={}
        for threshold in thresholds:
            predicted=decode_typed(p['sequences'],probabilities,threshold=threshold)
            strict=score_welds(predicted,p['truth'])
            potential={typ:sum(1 for r in rows if r['type']==typ and min(r['start_peak_within_3mm'],r['end_peak_within_3mm'],r['foreground_mean_on_reference'])>=threshold)
                       for typ in ['horizontal','vertical']}
            diagnostics[str(threshold)]={'strict_valid':strict['valid_welds'],'predictions':strict['prediction_count'],
                'false_or_duplicate':strict['false_or_duplicate_predictions'],'truth':strict['truth_count'],
                'reference_assisted_endpoint_signal_count':potential}
        report['groups'][name]={'thresholds':diagnostics,'per_reference_signal':rows}
        print(name,json.dumps(diagnostics),flush=True)
        (out/'coverage.partial.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        del p,probabilities;gc.collect()
    report['global_threshold_totals']={str(t):{
        key:sum(g['thresholds'][str(t)][key] for g in report['groups'].values())
        for key in ['strict_valid','predictions','false_or_duplicate','truth']} for t in thresholds}
    (out/'coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
