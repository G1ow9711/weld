"""Matched-budget sequence training with input-only third-surface context."""
import argparse
import json
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_sequence_model_820 import configure_precision
from weld_pipeline.sjg_train_full_sequence_827 import DATA,CORE,ACCUMULATE,full_schedule
from weld_pipeline.sjg_train_sequence_822 import objective
from weld_pipeline.sjg_adjacency_model_840 import AdjacencySequenceModel

EXTRA=OUT/'run51';EPOCHS=4;SEED=836


def load_core(record,start,end):
    with np.load(DATA/record['file']) as z:data={k:z[k] for k in z.files}
    with np.load(EXTRA/f"source_{record['source']:02d}/profile_{record['profile']:06d}.npz") as z:
        assert np.array_equal(data['q_mm'],z['q_mm']);extra=z['extra']
    left=max(0,start-126);right=min(len(data['x']),end+126)
    x=torch.from_numpy(np.c_[data['x'][left:right],extra[left:right]])[None]
    valid=torch.zeros((1,CORE+252));valid[:,:x.shape[1]]=1
    x=F.pad(x,(0,0,0,valid.shape[1]-x.shape[1]));condition=torch.from_numpy(data['condition'])[None]
    truth={k:torch.from_numpy(v[start:end])[None] for k,v in data.items() if k not in ('x','condition','q_mm')}
    return x,condition,truth,slice(start-left,end-left),valid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(SEED);configure_precision()
    records=read(DATA/'manifest.json')['sequences'];schedule=full_schedule(records);updates=completed=0
    protocol=dict(stage='FINITE_THIRD_SURFACE_SEQUENCE_TRAINING',train=TRAIN,development_not_read=VALID,arc_not_read=ARC,
        original_G=496,epochs=EPOCHS,core_points=CORE,gradient_accumulation=ACCUMULATE,learning_rate=.0001,
        grid_points_per_epoch=sum(r['points'] for r in records),cores_per_epoch=len(schedule),updates_per_epoch=602,
        comparison='run49/control: same initial checkpoint, seed, order, full coverage, optimizer, budget and original loss',
        seed=SEED,device='cpu',input_features=65,condition_features_unchanged=202,
        initialization_sha256=digest(OUT/'run44/last.pt'),zero_initialized_extra_projection=True,
        adjacency_input_manifest_sha256=digest(EXTRA/'all_inputs_frozen.json'),
        source_sha256=digest(__file__),model_sha256=digest(ROOT/'weld_pipeline/sjg_adjacency_model_840.py'),
        feature_sha256=digest(ROOT/'weld_pipeline/sjg_third_surface_838.py'),loss_sha256=digest(ROOT/'weld_pipeline/sjg_train_sequence_822.py'),
        no_GT_cutback_rule=True,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        assert read(EXTRA/'receipt.json')['status']=='FINITE_THIRD_SURFACE_FEATURES_COMPLETE'
        for source in TRAIN:
            folder=EXTRA/f'source_{source:02d}'
            for record in read(folder/'manifest.json')['profiles']:assert digest(folder/record['file'])==record['sha256']
        assert all(r['source'] in TRAIN and digest(DATA/r['file'])==r['sha256'] for r in records)
        model=AdjacencySequenceModel()
        with (OUT/'run44/last.pt').open('rb') as h:state=torch.load(h,map_location='cpu')
        model.load_baseline(state['model']);del state
        initial={k:v.clone() for k,v in model.state_dict().items()};optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001)
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,EPOCHS+1):
                model.train();check_resources();optimizer.zero_grad(set_to_none=True)
                order=np.random.default_rng(SEED+epoch).permutation(len(schedule));points=0;losses=[];started=time.perf_counter()
                for number,index in enumerate(order,1):
                    if number%200==1:check_resources()
                    ri,start,end=schedule[int(index)];x,condition,truth,core,valid=load_core(records[ri],start,end)
                    loss,values=objective(tuple(v[:,core] for v in model(x,condition,valid)),truth);assert torch.isfinite(loss)
                    group_start=((number-1)//ACCUMULATE)*ACCUMULATE
                    (loss/min(ACCUMULATE,len(schedule)-group_start)).backward();losses.append(values);points+=end-start
                    if number%ACCUMULATE==0 or number==len(schedule):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                    if number%400==0 or number==len(schedule):print('ADJACENCY_EPOCH',epoch,'/',EPOCHS,'CORE',number,'/',len(schedule),'ACTUAL_UPDATES',updates,flush=True)
                assert points==protocol['grid_points_per_epoch'];completed=epoch
                row=dict(epoch=epoch,epochs=EPOCHS,actual_updates=updates,grid_points=points,
                    mean_losses=np.mean(losses,0).tolist(),seconds=time.perf_counter()-started)
                log.write(json.dumps(row)+'\n');log.flush();print('ADJACENCY_EPOCH_COMPLETE',row,flush=True)
                with (dest/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,protocol=protocol),h)
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])];assert 'adjacency.weight' in changed
        write(dest/'receipt.json',dict(status='CONTINUOUS_SEQUENCE_TRAINING_COMPLETE',actual_optimizer_updates=updates,
            completed_full_grid_epochs=completed,grid_points_per_epoch=points,changed_parameter_tensors=len(changed),
            adjacency_projection_trained=True,checkpoint_sha256=digest(dest/'last.pt'),end_utc=utc(),full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,completed_full_grid_epochs=completed,
            error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
