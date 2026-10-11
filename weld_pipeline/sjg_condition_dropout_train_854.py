"""Matched two-arm continuation with TRAIN-mean global-condition dropout."""
import argparse
import gc
import json
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_train_adjacency_841 import DATA,EXTRA,CORE,ACCUMULATE,full_schedule,load_core
from weld_pipeline.sjg_adjacency_model_840 import AdjacencySequenceModel
from weld_pipeline.sjg_sequence_model_820 import configure_precision
from weld_pipeline.sjg_train_sequence_822 import objective

EPOCHS=2;SEED=854


def mask_condition(condition,mean,masked):
    return mean.expand_as(condition) if masked else condition


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);configure_precision()
    records=read(DATA/'manifest.json')['sequences'];schedule=full_schedule(records);conditions=[]
    protocol=dict(stage='MATCHED_GLOBAL_CONDITION_DROPOUT',arms=['control','condition_dropout'],
        train=TRAIN,development_not_read=VALID,arc_not_read=ARC,original_G=496,
        epochs=EPOCHS,grid_points_per_epoch=sum(r['points'] for r in records),cores_per_epoch=len(schedule),
        updates_per_arm=EPOCHS*int(np.ceil(len(schedule)/ACCUMULATE)),seed=SEED,device='cpu',learning_rate=.0001,
        initial_checkpoint='run52',initial_checkpoint_sha256=digest(OUT/'run52/last.pt'),
        dropout='independent Bernoulli0.5 per TRAIN profile per epoch, whole202-dimensional condition replaced by TRAIN profile mean',
        same_profile_same_mask_for_all_cores=True,input_features_and_loss_unchanged=True,inference_condition_unchanged=True,
        same_architecture_parameters_optimizer_order_and_budget=True,all_arms_evaluated=True,
        fixed_event_threshold=.5,source_sha256=digest(__file__),data_manifest_sha256=digest(DATA/'manifest.json'),
        model_sha256=digest(ROOT/'weld_pipeline/sjg_adjacency_model_840.py'),full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);current=None;updates=completed=0
    try:
        for r in records:
            assert r['source'] in TRAIN and digest(DATA/r['file'])==r['sha256']
            with np.load(DATA/r['file']) as z:conditions.append(z['condition'])
        for source in TRAIN:
            folder=EXTRA/f'source_{source:02d}'
            for r in read(folder/'manifest.json')['profiles']:assert digest(folder/r['file'])==r['sha256']
        mean=torch.from_numpy(np.mean(conditions,axis=0))[None];np.save(dest/'TRAIN_condition_mean.npy',mean.numpy()[0])
        for arm in protocol['arms']:
            current=dest/arm;current.mkdir();torch.manual_seed(SEED);model=AdjacencySequenceModel()
            with (OUT/'run52/last.pt').open('rb') as h:state=torch.load(h,map_location='cpu')
            model.load_state_dict(state['model']);del state
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001);updates=completed=0
            write(current/'registered_protocol.json',dict(protocol,arm=arm))
            with (current/'epochs.jsonl').open('w',encoding='utf-8') as log:
                for epoch in range(1,EPOCHS+1):
                    model.train();check_resources();optimizer.zero_grad(set_to_none=True);points=masked_points=0;losses=[];started=time.perf_counter()
                    order=np.random.default_rng(SEED+epoch).permutation(len(schedule))
                    mask=np.random.default_rng(SEED+10000+epoch).random(len(records))<.5
                    if arm=='control':mask[:]=False
                    for number,index in enumerate(order,1):
                        if number%200==1:check_resources()
                        ri,start,end=schedule[int(index)];x,condition,truth,core,valid=load_core(records[ri],start,end)
                        condition=mask_condition(condition,mean,bool(mask[ri]))
                        loss,values=objective(tuple(v[:,core] for v in model(x,condition,valid)),truth);assert torch.isfinite(loss)
                        group_start=((number-1)//ACCUMULATE)*ACCUMULATE
                        (loss/min(ACCUMULATE,len(schedule)-group_start)).backward();points+=end-start;masked_points+=(end-start)*int(mask[ri]);losses.append(values)
                        if number%ACCUMULATE==0 or number==len(schedule):
                            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                        if number%400==0 or number==len(schedule):print('CONDITION_EPOCH',arm,epoch,'/',EPOCHS,'CORE',number,'/',len(schedule),'ACTUAL_UPDATES',updates,flush=True)
                    assert points==protocol['grid_points_per_epoch'];completed=epoch
                    row=dict(arm=arm,epoch=epoch,epochs=EPOCHS,actual_updates=updates,grid_points=points,
                        masked_profiles=int(mask.sum()),masked_points=masked_points,mean_losses=np.mean(losses,0).tolist(),seconds=time.perf_counter()-started)
                    log.write(json.dumps(row)+'\n');log.flush();print('CONDITION_EPOCH_COMPLETE',row,flush=True)
                    with (current/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,protocol=protocol,arm=arm),h)
            write(current/'receipt.json',dict(status='CONTINUOUS_SEQUENCE_TRAINING_COMPLETE',actual_optimizer_updates=updates,
                completed_full_grid_epochs=completed,checkpoint_sha256=digest(current/'last.pt'),full_contract_pass=False,end_utc=utc()))
            del model,optimizer;gc.collect()
        write(dest/'receipt.json',dict(status='TWO_CONDITION_ARMS_TRAINED',actual_optimizer_updates=2*protocol['updates_per_arm'],
            full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',arm=current.name if current else None,actual_optimizer_updates=updates,
            completed_full_grid_epochs=completed,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
