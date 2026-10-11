"""Measure actual learned offset precision and separate support from localization."""
import json
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.joint_event_votes import JointEventVotes,infer_votes
from weld_pipeline.train_joint_relation_crossfit import DATA,load_data

OUT=Path('weld_pipeline/reports/joint_event_vote_precision_151_20261001')
RUN=Path('weld_pipeline/runs/joint_event_vote_capacity_seed151')
TARGETS=Path('weld_pipeline/data/joint_event_votes_pilot151')
NAMES=('161-LB18A','203-TT1A','207-FR80A')


def main():
    if OUT.exists():raise ValueError('Preserve learned event-vote precision audit')
    protocol=json.loads((RUN/'config.json').read_text('utf8'))
    if any(n not in protocol['train'] for n in NAMES):raise ValueError('Only actual training sources allowed')
    OUT.mkdir(parents=True);torch.set_num_threads(2)
    config=json.loads((DATA/'config.json').read_text('utf8'));config['pieces']=list(NAMES)
    data=load_data(config);checkpoint=RUN/'final.pt';digest=sha256(checkpoint)
    state=torch.load(checkpoint,map_location='cpu');model=JointEventVotes().cuda();model.load_state_dict(state['model']);model.eval()
    results={}
    for name in NAMES:
        item=data[name];piece=load_piece(item['manifest']['source_cache_root'],verify=True)
        blocks=[]
        for entry in item['horizontal']:
            length=entry['stop']-entry['start']
            for start in range(0,length,1024):blocks.append((entry,start,min(length,start+1024)))
        captured=[]
        handle=model.output.register_forward_hook(lambda module,args,result:captured.append(result.detach().cpu().numpy()))
        try:infer_votes(model,item,piece['sequences'])
        finally:handle.remove()
        raw=np.full((len(item['targets']),7),np.nan,np.float32)
        at=0
        for batch in captured:
            for bi in range(len(batch)):
                entry,a,b=blocks[at];at+=1
                raw[entry['start']+a:entry['start']+b]=batch[bi,:,64:64+b-a].T
        if at!=len(blocks):raise ValueError('Actual output hook/block alignment differs')
        target=np.load(TARGETS/(name+'_targets.npy'),mmap_mode='r')
        known=np.load(TARGETS/(name+'_known.npy'),mmap_mode='r')
        foreground=(known[:,0]>0)&np.isfinite(raw[:,0]);fg_probability=1./(1.+np.exp(-np.clip(raw[foreground,0],-60,60)))
        fg_truth=target[foreground,0]>.5;fg_prediction=fg_probability>.5
        heads={}
        for c,label in enumerate(('start','end','cut')):
            mask=(known[:,c+4]>0)&np.isfinite(raw[:,c+4])
            errors=np.abs(np.clip(raw[mask,c+4],-1.,1.)-target[mask,c+4])*64.
            support=1./(1.+np.exp(-np.clip(raw[mask,c+1],-60,60)))
            if not len(errors):heads[label]=dict(supported_points=0);continue
            heads[label]=dict(supported_points=len(errors),support_recall=float((support>.5).mean()),
                offset_mean_mm=float(errors.mean()),offset_median_mm=float(np.median(errors)),
                offset_p95_mm=float(np.quantile(errors,.95)),offset_within_3mm_rate=float((errors<=3.).mean()),
                mean_normalized_SmoothL1=float(np.where(errors/64.<.05,.5*(errors/64.)**2/.05,errors/64.-.025).mean()),
                support_and_offset_within_3mm_rate=float(((support>.5)&(errors<=3.)).mean()))
        result=dict(piece=name,foreground_true=int(fg_truth.sum()),foreground_recall=float((fg_prediction&fg_truth).sum()/fg_truth.sum()),
            foreground_false=int((fg_prediction&~fg_truth).sum()),heads=heads,checkpoint_sha256=digest,
            analysis_not_threshold_tuning=True,original_labels_changed=False,full_contract_pass=False)
        results[name]=result
        print('EVENT_VOTE_OFFSET_PRECISION',name,json.dumps(heads),flush=True)
    if sha256(checkpoint)!=digest:raise ValueError('Frozen vote checkpoint changed')
    complete=dict(pieces=results,checkpoint_sha256=digest,verifier_sha256=sha256(__file__),
        source_sha256=sha256('weld_pipeline/joint_event_votes.py'),held_sources_read=False,observed_check_read=False,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete,indent=2,ensure_ascii=False),encoding='utf8')


if __name__=='__main__':main()
