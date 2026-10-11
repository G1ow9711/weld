"""Sampled FIT head errors, kept separate from full-child acceptance scores."""
import gc
import json
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.neural_physical_event_sets import PhysicalEventNet

RUN=Path('weld_pipeline/runs/deep_physical_events_170_v3')
DATA=Path('weld_pipeline/data/neural_physical_event_pilot170_v2')
OUT=Path('weld_pipeline/reports/deep_sample_capacity_174_20261002')


@torch.no_grad()
def main():
    if OUT.exists():raise ValueError('Preserve the fixed sample-capacity diagnosis')
    complete=json.loads((RUN/'complete.json').read_text('utf8'))
    variant=sorted(complete['variants'],key=lambda v:(-complete['variants'][v]['fit']['valid_children'],complete['variants'][v]['fit']['false_predictions'],v))[0]
    config=json.loads((RUN/'config.json').read_text('utf8'))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    checkpoint=RUN/variant/'final.pt';frozen=sha256(checkpoint)
    if frozen!=complete['variants'][variant]['checkpoint_sha256']:raise ValueError('Frozen model changed')
    saved=torch.load(checkpoint,map_location=device);model=PhysicalEventNet().to(device);model.load_state_dict(saved['state_dict']);model.eval()
    OUT.mkdir(parents=True)
    names=('foreground','start_support','end_support','cut_support')
    totals={k:dict(known=0,tp=0,fn=0,fp=0,tn=0) for k in names}
    errors={k:[] for k in ('start','end','cut')};receipts={}
    for name in config['train']:
        root=DATA/name;manifest=json.loads((root/'manifest.json').read_text('utf8'))
        for file,digest in manifest['files'].items():
            if sha256(root/file)!=digest:raise ValueError('Frozen sampled FIT data changed')
        x=np.load(root/'features.npy',allow_pickle=False);y=np.load(root/'targets.npy',allow_pickle=False);known=np.load(root/'known.npy',allow_pickle=False)
        if not saved['actual_xyz']:x[:,-3:]=0.
        raw=np.zeros((len(x),7),np.float32)
        for a in range(0,len(x),8192):
            b=min(len(x),a+8192)
            raw[a:b]=model(torch.from_numpy((x[a:b]-saved['mean'])/saved['std']).to(device)).cpu().numpy()
        local={}
        for channel,key in enumerate(names):
            valid=known[:,channel]>0;truth=y[:,channel]>.5;pred=raw[:,channel]>=0.
            row=dict(known=int(valid.sum()),tp=int((valid&truth&pred).sum()),fn=int((valid&truth&~pred).sum()),
                     fp=int((valid&~truth&pred).sum()),tn=int((valid&~truth&~pred).sum()))
            local[key]=row
            for k,v in row.items():totals[key][k]+=v
        local_errors={}
        for role,key in enumerate(errors):
            valid=known[:,role+4]>0;error=np.abs(raw[:,role+4]-y[:,role+4])[valid]*16.
            errors[key].append(error)
            local_errors[key]=dict(known=len(error),median_mm=float(np.median(error)) if len(error) else None,
                p95_mm=float(np.quantile(error,.95)) if len(error) else None,within_3mm=float(np.mean(error<=3.)) if len(error) else None)
        receipt=dict(binary=local,offsets=local_errors,sample_metrics_not_child_acceptance=True,full_contract_pass=False)
        p=OUT/(name+'.json');p.write_text(json.dumps(receipt,indent=2),encoding='utf8');receipts[name]=sha256(p)
        del x,y,known,raw;gc.collect()
    for row in totals.values():
        row['positive_recall']=row['tp']/(row['tp']+row['fn']) if row['tp']+row['fn'] else None
        row['negative_false_positive_rate']=row['fp']/(row['fp']+row['tn']) if row['fp']+row['tn'] else None
    summary={}
    for key,arrays in errors.items():
        values=np.concatenate(arrays)
        summary[key]=dict(known=len(values),median_mm=float(np.median(values)),p95_mm=float(np.quantile(values,.95)),within_3mm=float(np.mean(values<=3.)))
    result=dict(variant=variant,checkpoint_sha256=frozen,sampled_binary=totals,sampled_offsets=summary,
        piece_receipts=receipts,actual_trained_models=0,full_grid_or_full_child_acceptance_not_replaced=True,
        held_or_observed_read=False,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('DEEP_SAMPLE_CAPACITY_COMPLETE',json.dumps({'binary':totals,'offsets':summary}),flush=True)


if __name__=='__main__':main()
