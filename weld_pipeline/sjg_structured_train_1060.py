"""Bounded single-TRAIN proof of feasibility; no recursive decoder reused."""
import argparse,gc,random,time,traceback,shutil
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_cuda_startup_pages_1040 import startup_pages
from weld_pipeline.sjg_local_event_train_930 import save_checkpoint
from weld_pipeline.sjg_contact_tcn_1052 import neighbours
from weld_pipeline.sjg_structured_contact_1057 import StructuredContactTCN as ContactTCN,StructuredLayout,controls
from weld_pipeline.sjg_structured_objective_1059 import objective

def main():
    if (OUT/'local_potential_retired.json').exists():
        raise RuntimeError('Local-potential main predictor retired; no fourth event-loss tuning round. Frozen inference remains allowed.')
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    protocol=dict(stage='LATENT_PARENT_STRUCTURED_SINGLE_TRAIN_QUICK_PROBE',source='21',data_run='run224',TRAIN_G=1,all_G=500,
        epoch_budget=32,update_budget=32,wall_budget_seconds=180,seed=1031,learning_rate=.0003,weight_decay=.0001,clip_norm=5,
        architecture='width64 residual depthwise kernel3 local TCN dilations 1,2,4,8',frozen_PTv3=True,
        no_relation_family_root_gates=True,full_population_each_update=True,last_checkpoint_only=True,
        input_scope='all 4080 pure-input geometric families',normalization='TRAIN21 mean/std only, no clipping',
        broader_pilot='Seven TRAIN and BM/10/DEV64 tests require this first feasibility gate; G500 unchanged',
        score_semantics='shared marked-path potentials; structured logZ minus one complete-parent witness score',
        preflight_run='run238',model_change='same width64 TCN; independent 8-logit fixed input endpoint head because movable-slot tying duplicates shared breaks',
        boundary_events='all input boundaries optional; no GT anchor creation',process_rules='predicted flags/states/numeric context feeds path and boundary heads; known TRAIN process supervision; no GT process at inference; no hard 35mm without clearance',user_steering='Integrate extracted process rules into model before this new trial',loss='sum family structured NLL + 2 position SmoothL1 + known direction/type + masked process state/numeric loss; parent connection in marked states',
        limitations=['cached pure-input upstream components','single TRAIN source not independent test','partial process heads only; E16 completeness and formal truth not passed'],start_utc=utc())
    write(dest/'registered_protocol.json',protocol);(dest/'source_snapshot').mkdir()
    for number in (1052,1057,1058,1059,1060,1061,1062,1063):
        for p in (ROOT/'weld_pipeline').glob('*_'+str(number)+'.py'):shutil.copy2(p,dest/'source_snapshot'/p.name)
    model=None;updates=0;history=[];start=None
    try:
        torch.set_num_threads(2);torch.manual_seed(1031);np.random.seed(1031);random.seed(1031)
        write(dest/'controls.json',controls());preflight=OUT/protocol['preflight_run'];pr=read(preflight/'receipt.json');assert pr['status']=='STRUCTURED_FULL_INPUT_PREFLIGHT_PASS'
        assert digest(preflight/'witness_targets.npz')==pr['target_sha256'];assert digest(preflight/'normalization.npz')==pr['normalization_sha256']
        start=time.perf_counter();write(dest/'startup_pages.json',startup_pages());torch.manual_seed(1031);np.random.seed(1031);random.seed(1031)
        data=OUT/protocol['data_run'];receipt=read(data/'receipt.json');assert receipt['status']=='QUICK_CONTACT_DATA_READY'
        freeze=read(data/'input_freeze.json')
        for name,sha in freeze['files'].items():assert digest(data/name)==sha
        assert digest(data/'labels.npz')==receipt['label_sha256']
        raw=np.load(data/'features.npy',mmap_mode='r');mean=raw.mean(0,dtype=np.float64).astype(np.float32)
        variance=np.zeros(raw.shape[1],np.float64)
        for i in range(0,len(raw),4096):variance+=np.square(np.asarray(raw[i:i+4096],np.float64)-mean).sum(0)
        scale=np.sqrt(variance/len(raw)).astype(np.float32);scale[scale<1e-6]=1
        np.savez(dest/'normalization.npz',mean=mean,scale=scale)
        x=torch.from_numpy((raw-mean)/scale).cuda();meta=read(data/'input_domain.json')
        index=torch.from_numpy(neighbours(meta['sequences'],len(raw))).cuda()
        with np.load(data/'input_domain.npz') as z:length=torch.from_numpy(z['cells'][:,1]-z['cells'][:,0]).cuda()
        with np.load(data/'labels.npz') as z:y={k:torch.from_numpy(z[k]).cuda() for k in z.files}
        with np.load(preflight/'witness_targets.npz') as z:target={k:torch.from_numpy(z[k]).cuda() for k in z.files}
        witnesses=read(preflight/'witness_manifest.json')['witness_sequences'];layout=StructuredLayout(meta['sequences'],len(raw),'cuda')
        assert digest(preflight/'process_targets.npz')==pr['process_target_sha256']
        with np.load(preflight/'process_targets.npz') as z:process_target={k:torch.from_numpy(z[k]).cuda() for k in z.files}
        write(dest/'process_target_manifest.json',read(preflight/'process_target_manifest.json'))
        model=ContactTCN(raw.shape[1]).cuda();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);check_resources()
        for epoch in range(32):
            if time.perf_counter()-start>=180:break
            check_resources();opt.zero_grad(set_to_none=True);out=model(x,index);loss,parts=objective(out,y,target,layout,length,witnesses,process_target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();updates+=1
            row=dict(epoch=epoch+1,updates=updates,loss=float(loss.detach()),gradient_norm=float(norm),parts=parts,elapsed_seconds=time.perf_counter()-start)
            history.append(row);write(dest/'progress.json',row)
            print('EPOCH',epoch+1,'/32','UPDATE',updates,'/32','LOSS',round(row['loss'],5),'WITNESS',parts['witness_family'],'POSITION_MM',round(parts['position_mm'],4),'SECONDS',round(row['elapsed_seconds'],2),flush=True)
            del out,loss;opt.zero_grad(set_to_none=True)
            save_checkpoint(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),updates=updates,witness_diagnostics=history,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),dest/'recovery.pt')
        final={k:v.detach().cpu() for k,v in model.state_dict().items()};change=float(sum((final[k]-initial[k]).double().square().sum() for k in final).sqrt())
        save_checkpoint(dict(architecture='latent_parent_structured_contact_tcn',features=raw.shape[1],state_dict=final,updates=updates,data_run=protocol['data_run'],normalization_sha256=digest(dest/'normalization.npz')),dest/'final.pt')
        seconds=time.perf_counter()-start;write(dest/'history.json',history)
        write(dest/'receipt.json',dict(status='CONTACT_TRAIN_COMPLETE' if updates==32 and seconds<=180 else 'CONTACT_TRAIN_INCOMPLETE',
            updates=updates,training_seconds=seconds,parameter_change_l2=change,checkpoint_sha256=digest(dest/'final.pt'),
            source='21',TRAIN_G=1,all_G=500,full_contract_pass=False,end_utc=utc()))
        print('CONTACT_TRAIN_END',updates,'SECONDS',round(seconds,3),flush=True)
    except BaseException:
        if model is not None:save_checkpoint(dict(state_dict=model.state_dict(),updates=updates),dest/'failure_checkpoint.pt')
        write(dest/'failure.json',dict(error=traceback.format_exc(),updates=updates,utc=utc()));raise
if __name__=='__main__':main()
