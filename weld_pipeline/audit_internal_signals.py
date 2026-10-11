"""Diagnose internal-boundary learning after inference; never a predictor."""
import argparse, gc, json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import find_peaks
from training.labels import sha256
from weld_pipeline.mapped_cache import load_piece
from weld_pipeline.model_registry_research import model_from_checkpoint
from weld_pipeline.typed_contact_model import infer_probabilities


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--data',required=True)
    ap.add_argument('--outdir',required=True)
    args=ap.parse_args(); root=Path(args.data); out=Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    with Path(args.checkpoint).open('rb') as f: state=torch.load(f,map_location='cpu')
    torch.set_num_threads(4);model=model_from_checkpoint(state).cuda().eval()
    report={'checkpoint_sha256':sha256(args.checkpoint),
            'scope':'Reference-assisted diagnostic after inference; not extraction accuracy.', 'groups':{}}
    for part in ['train','validation']:
        for name in split[part]:
            folder=root/split.get('piece_roots',{}).get(name,name)
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            if part=='train' and not any(len(w['segments'])>1 for w in manifest['truth']): continue
            piece=load_piece(folder)
            probs=infer_probabilities(model,piece['sequences'])
            rows=[]
            for wi,w in enumerate(piece['truth']):
                if len(w['segments'])<2:continue
                si=piece['record']['mapping']['weld_to_sequence'][str(wi)]
                seq=piece['sequences'][si];prob=probs[si];step=seq['sample_step_mm']
                endpoints=np.array([w['segments'][0]['points'][0],w['segments'][-1]['points'][-1]])
                lo,hi=sorted((endpoints-seq['xyz'][0])@seq['direction'])
                t=np.arange(len(prob))*step
                peaks,_=find_peaks(prob[:,3],height=.5,distance=max(1,int(np.ceil(4/step))),prominence=.05)
                peaks=peaks[(t[peaks]>lo+1.5)&(t[peaks]<hi-1.5)]
                bounds=[float((np.array(s['points'][-1])-seq['xyz'][0])@seq['direction']) for s in w['segments'][:-1]]
                rows.append({'id':w['id'],'type':w['type'],'segments':len(w['segments']),
                    'start_probability':float(prob[np.abs(t-lo)<=3,1].max()),
                    'end_probability':float(prob[np.abs(t-hi)<=3,2].max()),
                    'mask_mean':float(prob[(t>=lo)&(t<=hi),0].mean()),
                    'internal_peak_probabilities':[float(prob[np.abs(t-b)<=3,3].max()) for b in bounds],
                    'predicted_internal_count':len(peaks),
                    'nearest_internal_errors_mm':[float(np.min(np.abs(t[peaks]-b))) if len(peaks) else None for b in bounds]})
            truth_boundaries=sum(len(r['internal_peak_probabilities']) for r in rows)
            high=sum(p>=.5 for r in rows for p in r['internal_peak_probabilities'])
            report['groups'][name]={'partition':part,'multi_welds':len(rows),'truth_boundaries':truth_boundaries,
                                    'boundary_signal_at_05':high,'rows':rows}
            print(name,part,'multi',len(rows),'boundaries',high,'/',truth_boundaries,flush=True)
            (out/'internal_signals.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            del piece,probs;gc.collect()


if __name__=='__main__':main()
