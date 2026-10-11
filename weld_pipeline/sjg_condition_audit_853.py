"""TRAIN-only sensitivity to replacing global condition blocks by TRAIN means."""
import argparse
import numpy as np
import torch
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_train_adjacency_841 import DATA,load_core,full_schedule
from weld_pipeline.sjg_adjacency_model_840 import AdjacencySequenceModel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    records=read(DATA/'manifest.json')['sequences'];conditions=[]
    for r in records:
        assert r['source'] in TRAIN and digest(DATA/r['file'])==r['sha256']
        with np.load(DATA/r['file']) as z:conditions.append(z['condition'])
    mean=torch.from_numpy(np.mean(conditions,axis=0))[None]
    schedule=full_schedule(records);rng=np.random.default_rng(853);chosen=[]
    for source in TRAIN:
        ids=[i for i,(ri,s,e) in enumerate(schedule) if records[ri]['source']==source]
        chosen.extend(rng.choice(ids,min(48,len(ids)),replace=False).tolist())
    protocol=dict(stage='TRAIN_ONLY_GLOBAL_CONDITION_SENSITIVITY',sources=TRAIN,sampled_cores=len(chosen),seed=853,
        sample='up to48 random full-schedule cores per TRAIN workpiece; no endpoint-based selection',
        checkpoint='run52',checkpoint_sha256=digest(OUT/'run52/last.pt'),source_sha256=digest(__file__),
        condition_blocks={'joint':[0,96],'global_statistics':[96,202]},
        intervention='replace block by TRAIN profile mean, no retraining; diagnostic distribution shift',
        development_not_read=True,optimizer_updates=0,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);np.save(dest/'TRAIN_condition_mean.npy',mean.numpy()[0])
    model=AdjacencySequenceModel()
    with (OUT/'run52/last.pt').open('rb') as h:state=torch.load(h,map_location='cpu')
    model.load_state_dict(state['model']);model.eval();rows=[]
    with torch.inference_mode():
        for count,index in enumerate(chosen,1):
            ri,s,e=schedule[index];r=records[ri];x,c,truth,core,valid=load_core(r,s,e)
            for name,blocks in [('baseline',[]),('joint_mean',[(0,96)]),('all_mean',[(0,202)])]:
                changed=c.clone()
                for a,b in blocks:changed[:,a:b]=mean[:,a:b]
                logits=model(x,changed,valid)[0][:,core,1:];p=logits.sigmoid();target=truth['targets'][...,1:]
                positive=target>=.88;negative=target<=.01
                rows.append(dict(source=r['source'],profile=r['profile'],start=s,end=e,arm=name,
                    positive_cells=int(positive.sum()),negative_cells=int(negative.sum()),
                    positive_probability_sum=float(p[positive].sum()),negative_probability_sum=float(p[negative].sum()),
                    positive_over_half=int((p[positive]>=.5).sum()),negative_over_half=int((p[negative]>=.5).sum())))
            if count%24==0 or count==len(chosen):check_resources();print('TRAIN_CONDITION_AUDIT',count,'/',len(chosen),flush=True)
    summary=[]
    for source in TRAIN:
        for name in ['baseline','joint_mean','all_mean']:
            subset=[r for r in rows if r['source']==source and r['arm']==name]
            sums={key:sum(r[key] for r in subset) for key in ['positive_cells','negative_cells','positive_probability_sum','negative_probability_sum','positive_over_half','negative_over_half']}
            sums.update(source=source,arm=name,positive_mean=sums['positive_probability_sum']/max(1,sums['positive_cells']),
                negative_mean=sums['negative_probability_sum']/max(1,sums['negative_cells']))
            summary.append(sums)
    write(dest/'receipt.json',dict(status='TRAIN_CONDITION_SENSITIVITY_AUDITED',summary=summary,rows=rows,
        no_generalization_causality_claim=True,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    print('CONDITION_SUMMARY',summary,flush=True)


if __name__=='__main__':main()
