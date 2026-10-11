"""Matched complete-grid continuation with uncertain material scores."""
import argparse
import gc
import json
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_train_adjacency_841 import DATA,EXTRA,CORE,ACCUMULATE,full_schedule,load_core
from weld_pipeline.sjg_adjacency_model_840 import AdjacencySequenceModel
from weld_pipeline.sjg_weld_context_model_888 import MaterialSequenceModel
from weld_pipeline.sjg_sequence_model_820 import configure_precision
from weld_pipeline.sjg_train_sequence_822 import objective

EPOCHS=2;SEED=889;CONTEXT=OUT/'run88';INITIAL=OUT/'run63/control/last.pt'


def load_material_core(record,start,end,control=False):
    x,condition,truth,core,valid=load_core(record,start,end)
    with np.load(CONTEXT/f"source_{record['source']:02d}/profile_{record['profile']:06d}.npz") as z:
        extra=z['extra'];q=z['q_mm']
    with np.load(DATA/record['file']) as z:assert np.array_equal(q,z['q_mm'])
    left=max(0,start-126);right=min(len(extra),end+126)
    value=torch.from_numpy(extra[left:right])[None]
    value=F.pad(value,(0,0,0,x.shape[1]-value.shape[1]))
    if control:value=torch.zeros_like(value)
    return torch.cat([x,value],-1),condition,truth,core,valid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);configure_precision()
    records=read(DATA/'manifest.json')['sequences'];schedule=full_schedule(records)
    protocol=dict(stage='MATCHED_WELD_MATERIAL_CONTEXT',arms=['control','material_context'],
        train=TRAIN,development_not_read=VALID,arc_not_read=ARC,original_G=496,
        epochs=EPOCHS,grid_points_per_epoch=sum(r['points'] for r in records),cores_per_epoch=len(schedule),
        updates_per_arm=EPOCHS*int(np.ceil(len(schedule)/ACCUMULATE)),seed=SEED,device='cpu',learning_rate=.0001,
        initial_checkpoint=str(INITIAL.relative_to(OUT)),initial_checkpoint_sha256=digest(INITIAL),
        input_features=73,condition_features_unchanged=202,control_material_inputs_zero=True,
        same_architecture_seed_optimizer_order_loss_and_budget=True,point_backbone_updates=0,
        fixed_event_threshold=.5,source_sha256=digest(__file__),data_manifest_sha256=digest(DATA/'manifest.json'),
        context_manifest_sha256=digest(CONTEXT/'all_inputs_frozen.json'),
        model_sha256=digest(ROOT/'weld_pipeline/sjg_weld_context_model_888.py'),full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);current=None;updates=completed=0
    try:
        assert read(CONTEXT/'receipt.json')['status']=='RAW_MATERIAL_WELD_CONTEXT_COMPLETE'
        for r in records:assert r['source'] in TRAIN and digest(DATA/r['file'])==r['sha256']
        for source in TRAIN:
            for base in (EXTRA,CONTEXT):
                folder=base/f'source_{source:02d}'
                for r in read(folder/'manifest.json')['profiles']:assert digest(folder/r['file'])==r['sha256']
        for arm in protocol['arms']:
            current=dest/arm;current.mkdir();torch.manual_seed(SEED);model=MaterialSequenceModel()
            with INITIAL.open('rb') as h:state=torch.load(h,map_location='cpu')
            model.load_baseline(state['model']);baseline=AdjacencySequenceModel();baseline.load_state_dict(state['model']);del state
            model.eval();baseline.eval();r=records[0];x,c,_,_,v=load_material_core(r,0,min(CORE,r['points']))
            with torch.no_grad():
                old=baseline(x[...,:65],c,v);new=model(x,c,v)
            assert all(torch.equal(a,b) for a,b in zip(old,new));del baseline,old,new
            write(current/'initial_equivalence.json',dict(exact_output_equal=True,material_projection_zero=True))
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001);updates=completed=0
            write(current/'registered_protocol.json',dict(protocol,arm=arm))
            with (current/'epochs.jsonl').open('w',encoding='utf-8') as log:
                for epoch in range(1,EPOCHS+1):
                    model.train();check_resources();optimizer.zero_grad(set_to_none=True);points=0;losses=[];started=time.perf_counter()
                    order=np.random.default_rng(SEED+epoch).permutation(len(schedule))
                    for number,index in enumerate(order,1):
                        if number%200==1:check_resources()
                        ri,start,end=schedule[int(index)];x,condition,truth,core,valid=load_material_core(records[ri],start,end,arm=='control')
                        loss,values=objective(tuple(v[:,core] for v in model(x,condition,valid)),truth);assert torch.isfinite(loss)
                        group_start=((number-1)//ACCUMULATE)*ACCUMULATE
                        (loss/min(ACCUMULATE,len(schedule)-group_start)).backward();points+=end-start;losses.append(values)
                        if number%ACCUMULATE==0 or number==len(schedule):
                            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                        if number%200==0 or number==len(schedule):print('WELD_CONTEXT_EPOCH',arm,epoch,'/',EPOCHS,'CORE',number,'/',len(schedule),'ACTUAL_UPDATES',updates,flush=True)
                    assert points==protocol['grid_points_per_epoch'];completed=epoch
                    row=dict(arm=arm,epoch=epoch,epochs=EPOCHS,actual_updates=updates,grid_points=points,
                        mean_losses=np.mean(losses,0).tolist(),seconds=time.perf_counter()-started)
                    log.write(json.dumps(row)+'\n');log.flush();print('WELD_CONTEXT_EPOCH_COMPLETE',row,flush=True)
                    with (current/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,protocol=protocol,arm=arm),h)
            active=bool(torch.count_nonzero(model.material.weight));assert active==(arm=='material_context')
            write(current/'receipt.json',dict(status='CONTINUOUS_SEQUENCE_TRAINING_COMPLETE',actual_optimizer_updates=updates,
                completed_full_grid_epochs=completed,material_projection_nonzero=active,checkpoint_sha256=digest(current/'last.pt'),
                full_contract_pass=False,end_utc=utc()))
            del model,optimizer;gc.collect()
        write(dest/'receipt.json',dict(status='TWO_MATERIAL_CONTEXT_ARMS_TRAINED',actual_optimizer_updates=2*protocol['updates_per_arm'],full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',arm=current.name if current else None,actual_optimizer_updates=updates,
            completed_full_grid_epochs=completed,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
