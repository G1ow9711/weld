"""Small-cohort continuous profile training; no development target access."""
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

DATA=OUT/'run38'
EPOCHS=8
CORE=1024
ACCUMULATE=4
SEED=822
DEVICE='cpu'


def training_crop(record,rng):
    with np.load(DATA/record['file']) as z:data={k:z[k] for k in z.files if k!='q_mm'}
    count=len(data['x']);start=0
    if count>CORE:
        events=np.flatnonzero(data['offset_mask'].any(1))
        if len(events) and rng.random()<.5:
            center=int(rng.choice(events));start=int(np.clip(center-rng.integers(128,CORE-128),0,count-CORE))
        else:start=int(rng.integers(count-CORE+1))
    end=min(count,start+CORE);left=max(0,start-JointSequenceModel.receptive_radius)
    right=min(count,end+JointSequenceModel.receptive_radius)
    x=torch.as_tensor(data['x'][left:right],device=DEVICE)[None]
    valid=torch.zeros((1,CORE+2*JointSequenceModel.receptive_radius),device=DEVICE)
    valid[:,:x.shape[1]]=1
    x=F.pad(x,(0,0,0,valid.shape[1]-x.shape[1]))
    condition=torch.as_tensor(data['condition'],device=DEVICE)[None]
    targets={k:torch.as_tensor(v[start:end],device=DEVICE)[None] for k,v in data.items() if k not in ('x','condition')}
    return x,condition,targets,slice(start-left,end-left),end-start,valid


def objective(result,truth):
    logits,offset,types=result;y=truth['targets'];mask=truth['offset_mask']
    fg=F.binary_cross_entropy_with_logits(logits[...,:1],y[...,:1],pos_weight=logits.new_tensor([2.]))
    event=F.binary_cross_entropy_with_logits(logits[...,1:],y[...,1:],pos_weight=logits.new_tensor([20.,20.]))
    foreground=logits[...,0].sigmoid();target=y[...,0]
    dice=1-(2*(foreground*target).sum()+1)/(foreground.sum()+target.sum()+1)
    regression=(F.smooth_l1_loss(offset,truth['offsets'],reduction='none')*mask).sum()/mask.sum().clamp_min(1)
    known=truth['types']>=0
    kind=F.cross_entropy(types[known],truth['types'][known]) if known.any() else types.sum()*0
    continuity=F.l1_loss(foreground[:,1:]-foreground[:,:-1],target[:,1:]-target[:,:-1]) if target.shape[1]>1 else foreground.sum()*0
    loss=fg+event+.25*dice+.4*regression+.25*kind+.5*continuity
    return loss,torch.stack([fg,event,dice,regression,kind,continuity]).detach().cpu().numpy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(SEED);configure_precision()
    manifest=read(DATA/'manifest.json');records=manifest['sequences'];updates=0;completed=0
    protocol=dict(stage='CONTINUOUS_JOINT_SEQUENCE_TRAINING',train=TRAIN,development_not_read=VALID,
        arc_not_read=ARC,epochs=EPOCHS,profiles_per_epoch=len(records),
        epoch_definition='One new core crop per TRAIN profile, at most1024 points with126-point inference-consistent context',
        event_near_crop_probability=.5,gradient_accumulation=ACCUMULATE,updates_per_epoch=int(np.ceil(len(records)/ACCUMULATE)),
        learning_rate=.0003,seed=SEED,TF32_disabled=True,grid_step_mm=1.,
        device=DEVICE,fixed_input_points=CORE+2*JointSequenceModel.receptive_radius,
        padding_mask_applied_after_every_convolution=True,
        technical_restart_after_run39_and_run40_memory_guard_25_updates_each=True,
        PTv3_and_joint_graph_frozen=True,type_prediction_added=True,material_parent_E16_unresolved=True,
        data_manifest_sha256=digest(DATA/'manifest.json'),source_sha256=digest(__file__),
        model_source_sha256=digest(ROOT/'weld_pipeline/sjg_sequence_model_820.py'),
        no_validation_selection=True,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        assert read(DATA/'receipt.json')['status']=='WHOLE_PROFILE_TRAIN_SEQUENCES_COMPLETE'
        for record in records:assert record['source'] in TRAIN and digest(DATA/record['file'])==record['sha256']
        check_resources();model=JointSequenceModel().to(DEVICE);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        protocol['trainable_parameters']=sum(p.numel() for p in model.parameters());write(dest/'registered_protocol.json',protocol)
        if DEVICE=='cuda':torch.cuda.reset_peak_memory_stats()
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,EPOCHS+1):
                model.train();rng=np.random.default_rng(SEED+epoch);order=rng.permutation(len(records))
                losses=[];points=0;started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                for number,index in enumerate(order,1):
                    if number%100==1:
                        resource=check_resources()
                        print('SEQUENCE_MEMORY',number,{k:round(v/2**30,3) for k,v in resource.items() if k in ('available_physical_bytes','remaining_commit_bytes','private_commit_bytes')},flush=True)
                    x,condition,truth,core,count,valid=training_crop(records[int(index)],rng)
                    result=tuple(v[:,core] for v in model(x,condition,valid));loss,values=objective(result,truth)
                    assert torch.isfinite(loss)
                    group_start=((number-1)//ACCUMULATE)*ACCUMULATE
                    group_count=min(ACCUMULATE,len(records)-group_start)
                    (loss/group_count).backward();losses.append(values);points+=count
                    if number%ACCUMULATE==0 or number==len(records):
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
                    if number%250==0 or number==len(records):
                        print('EPOCH',epoch,'/',EPOCHS,'PROFILE',number,'/',len(records),'ACTUAL_UPDATES',updates,flush=True)
                completed=epoch
                row=dict(epoch=epoch,epochs=EPOCHS,actual_updates=updates,profiles_seen=len(records),
                    core_points_seen=points,mean_losses=np.mean(losses,0).tolist(),elapsed_seconds=time.perf_counter()-started)
                log.write(json.dumps(row)+'\n');log.flush();print('SEQUENCE_EPOCH_COMPLETE',row,flush=True)
                with (dest/'last.pt').open('wb') as handle:
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,protocol=protocol),handle)
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v.cpu(),initial[k])]
        assert changed and any(k.startswith('blocks.') for k in changed)
        write(dest/'receipt.json',dict(status='CONTINUOUS_SEQUENCE_TRAINING_COMPLETE',actual_optimizer_updates=updates,
            completed_crop_epochs=completed,changed_parameter_tensors=len(changed),checkpoint_sha256=digest(dest/'last.pt'),
            device=DEVICE,peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20 if DEVICE=='cuda' else 0.,end_utc=utc(),
            free_output_evaluation_pending=True,full_contract_pass=False))
    except Exception:
        from weld_pipeline.reports.full25_source_geometry_263_20261003.run_remaining_source_batch_263 import memory
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,completed_crop_epochs=completed,
            memory=memory(),end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
