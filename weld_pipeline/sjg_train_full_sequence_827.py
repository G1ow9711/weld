"""Cover every TRAIN sequence grid point once per epoch, including endpoints."""
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
from weld_pipeline.sjg_sequence_model_820 import JointSequenceModel,configure_precision
from weld_pipeline.sjg_train_sequence_822 import objective

DATA=OUT/'run38';CORE=1024;ACCUMULATE=4;EPOCHS=4;SEED=827


def full_schedule(records):
    schedule=[]
    for index,record in enumerate(records):
        n=record['points'];seen=np.zeros(n,np.uint8)
        for start in range(0,n,CORE):
            end=min(n,start+CORE);seen[start:end]+=1;schedule.append((index,start,end))
        assert np.all(seen==1)
    return schedule


def load_core(record,start,end):
    with np.load(DATA/record['file']) as z:data={k:z[k] for k in z.files if k!='q_mm'}
    left=max(0,start-126);right=min(len(data['x']),end+126)
    x=torch.from_numpy(data['x'][left:right])[None]
    valid=torch.zeros((1,CORE+252));valid[:,:x.shape[1]]=1
    x=F.pad(x,(0,0,0,valid.shape[1]-x.shape[1]))
    condition=torch.from_numpy(data['condition'])[None]
    truth={k:torch.from_numpy(v[start:end])[None] for k,v in data.items() if k not in ('x','condition')}
    return x,condition,truth,slice(start-left,end-left),valid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(SEED);configure_precision()
    records=read(DATA/'manifest.json')['sequences'];schedule=full_schedule(records);updates=completed=0
    initial_path=OUT/'run41/last.pt'
    protocol=dict(stage='FULL_COVERAGE_SEQUENCE_FINETUNE',train=TRAIN,development_not_read=VALID,arc_not_read=ARC,
        epochs=EPOCHS,core_points=CORE,gradient_accumulation=ACCUMULATE,
        full_grid_points_per_epoch=sum(r['points'] for r in records),cores_per_epoch=len(schedule),
        updates_per_epoch=int(np.ceil(len(schedule)/ACCUMULATE)),
        every_original_TRAIN_grid_point_exactly_once_per_epoch=True,all_target_event_peaks_seen_each_epoch=True,
        learning_rate=.0001,device='cpu',TF32_disabled=True,seed=SEED,
        initialization_checkpoint_sha256=digest(initial_path),new_optimizer=True,
        source_sha256=digest(__file__),model_source_sha256=digest(ROOT/'weld_pipeline/sjg_sequence_model_820.py'),
        loss_source_sha256=digest(ROOT/'weld_pipeline/sjg_train_sequence_822.py'),
        supervision_sampling_audit_sha256=digest(OUT/'run41/sampling_audit_826.json'),
        source_context_unchanged=True,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);write(dest/'coverage_schedule.json',dict(schedule=schedule,all_points_once=True))
    try:
        assert all(r['source'] in TRAIN and digest(DATA/r['file'])==r['sha256'] for r in records)
        model=JointSequenceModel()
        with initial_path.open('rb') as h:state=torch.load(h,map_location='cpu')
        model.load_state_dict(state['model']);del state
        initial={k:v.clone() for k,v in model.state_dict().items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001)
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,EPOCHS+1):
                check_resources();model.train();optimizer.zero_grad(set_to_none=True)
                order=np.random.default_rng(SEED+epoch).permutation(len(schedule));losses=[];points=0;started=time.perf_counter()
                for number,index in enumerate(order,1):
                    if number%200==1:check_resources()
                    ri,start,end=schedule[int(index)];x,condition,truth,core,valid=load_core(records[ri],start,end)
                    loss,values=objective(tuple(v[:,core] for v in model(x,condition,valid)),truth)
                    assert torch.isfinite(loss)
                    group_start=((number-1)//ACCUMULATE)*ACCUMULATE
                    (loss/min(ACCUMULATE,len(schedule)-group_start)).backward()
                    losses.append(values);points+=end-start
                    if number%ACCUMULATE==0 or number==len(schedule):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                    if number%400==0 or number==len(schedule):
                        print('FULL_EPOCH',epoch,'/',EPOCHS,'CORE',number,'/',len(schedule),'ACTUAL_UPDATES',updates,flush=True)
                assert points==protocol['full_grid_points_per_epoch'];completed=epoch
                row=dict(epoch=epoch,epochs=EPOCHS,complete_epoch=True,actual_updates=updates,
                    original_grid_points_seen=points,cores=len(schedule),mean_losses=np.mean(losses,0).tolist(),
                    elapsed_seconds=time.perf_counter()-started)
                log.write(json.dumps(row)+'\n');log.flush();print('FULL_COVERAGE_EPOCH_COMPLETE',row,flush=True)
                with (dest/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    epoch=epoch,updates=updates,protocol=protocol),h)
        changed=sum(not torch.equal(value,initial[k]) for k,value in model.state_dict().items())
        assert changed>0
        write(dest/'receipt.json',dict(status='CONTINUOUS_SEQUENCE_TRAINING_COMPLETE',actual_optimizer_updates=updates,
            completed_full_grid_epochs=completed,grid_points_per_epoch=protocol['full_grid_points_per_epoch'],
            changed_parameter_tensors=changed,checkpoint_sha256=digest(dest/'last.pt'),device='cpu',end_utc=utc(),
            free_output_evaluation_pending=True,full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,completed_full_grid_epochs=completed,
            end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
