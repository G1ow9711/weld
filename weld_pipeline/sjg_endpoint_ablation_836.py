"""Two controlled, complete-coverage CPU training arms on the fixed TRAIN six."""
import argparse
import gc
import json
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_sequence_model_820 import JointSequenceModel,configure_precision
from weld_pipeline.sjg_train_full_sequence_827 import DATA,CORE,ACCUMULATE,full_schedule,load_core
from weld_pipeline.sjg_balanced_endpoint_835 import objective,baseline_objective

EPOCHS=4;SEED=836


def train_arm(dest,records,schedule,loss_function,protocol):
    dest.mkdir(exist_ok=False);torch.manual_seed(SEED)
    write(dest/'registered_protocol.json',dict(protocol,arm=dest.name))
    with (OUT/'run44/last.pt').open('rb') as handle:state=torch.load(handle,map_location='cpu')
    model=JointSequenceModel();model.load_state_dict(state['model']);del state
    initial={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001)
    updates=completed=0
    try:
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,EPOCHS+1):
                model.train();check_resources();optimizer.zero_grad(set_to_none=True)
                order=np.random.default_rng(SEED+epoch).permutation(len(schedule))
                losses=[];points=0;started=time.perf_counter()
                for number,index in enumerate(order,1):
                    if number%200==1:check_resources()
                    ri,start,end=schedule[int(index)];x,condition,truth,core,valid=load_core(records[ri],start,end)
                    loss,values=loss_function(tuple(v[:,core] for v in model(x,condition,valid)),truth)
                    assert torch.isfinite(loss)
                    group_start=((number-1)//ACCUMULATE)*ACCUMULATE
                    (loss/min(ACCUMULATE,len(schedule)-group_start)).backward()
                    losses.append(values);points+=end-start
                    if number%ACCUMULATE==0 or number==len(schedule):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                    if number%400==0 or number==len(schedule):
                        print('ARM',dest.name,'EPOCH',epoch,'/',EPOCHS,'CORE',number,'/',len(schedule),'ACTUAL_UPDATES',updates,flush=True)
                assert points==protocol['grid_points_per_epoch'];completed=epoch
                row=dict(arm=dest.name,epoch=epoch,epochs=EPOCHS,actual_updates=updates,
                    grid_points=points,mean_losses=np.mean(losses,0).tolist(),seconds=time.perf_counter()-started)
                log.write(json.dumps(row)+'\n');log.flush();print('ABLATION_EPOCH_COMPLETE',row,flush=True)
                with (dest/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    epoch=epoch,updates=updates,protocol=protocol,arm=dest.name),h)
        changed=sum(not torch.equal(v,initial[k]) for k,v in model.state_dict().items());assert changed
        write(dest/'receipt.json',dict(status='CONTINUOUS_SEQUENCE_TRAINING_COMPLETE',arm=dest.name,
            actual_optimizer_updates=updates,completed_full_grid_epochs=completed,grid_points_per_epoch=points,
            checkpoint_sha256=digest(dest/'last.pt'),changed_parameter_tensors=changed,end_utc=utc(),full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,completed_full_grid_epochs=completed,
            error=traceback.format_exc(),end_utc=utc()));raise
    del model,optimizer,initial;gc.collect()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);configure_precision()
    records=read(DATA/'manifest.json')['sequences'];schedule=full_schedule(records)
    protocol=dict(stage='CONTROLLED_ENDPOINT_LOSS_ABLATION',train=TRAIN,development_not_read=VALID,arc_not_read=ARC,
        arms=['control','balanced'],only_changed_factor='endpoint_loss_mass_normalization',epochs_per_arm=EPOCHS,
        original_G=496,core_points=CORE,gradient_accumulation=ACCUMULATE,learning_rate=.0001,
        grid_points_per_epoch=sum(r['points'] for r in records),cores_per_epoch=len(schedule),
        updates_per_epoch=int(np.ceil(len(schedule)/ACCUMULATE)),same_initialization_and_core_order=True,
        initialization_sha256=digest(OUT/'run44/last.pt'),sequence_manifest_sha256=digest(DATA/'manifest.json'),
        sources={name:digest(ROOT/'weld_pipeline'/name) for name in ('sjg_endpoint_ablation_836.py','sjg_balanced_endpoint_835.py',
            'sjg_train_full_sequence_827.py','sjg_train_sequence_822.py','sjg_sequence_model_820.py')},
        seed=SEED,device='cpu',local_geometry_change_deferred_for_isolation=True,
        fixed_parent_checkpoint_sha256=digest(OUT/'run47/last.pt'),
        evaluation='Both fixed-budget checkpoints; identical free decoder and frozen parent model; no validation checkpoint selection',
        no_external_consult=True,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        assert all(r['source'] in TRAIN and digest(DATA/r['file'])==r['sha256'] for r in records)
        for name,loss in (('control',baseline_objective),('balanced',objective)):
            train_arm(dest/name,records,schedule,loss,protocol)
        write(dest/'receipt.json',dict(status='TWO_ARM_ENDPOINT_TRAINING_COMPLETE',actual_optimizer_updates=EPOCHS*len(range(0,len(schedule),ACCUMULATE))*2,
            arms={name:read(dest/name/'receipt.json') for name in ('control','balanced')},end_utc=utc(),full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
