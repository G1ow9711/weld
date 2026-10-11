"""Bounded single-TRAIN proof of feasibility; no recursive decoder reused."""
import argparse,gc,random,time,traceback,shutil
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_cuda_startup_pages_1040 import startup_pages
from weld_pipeline.sjg_local_event_train_930 import save_checkpoint
from weld_pipeline.sjg_contact_tcn_1052 import ContactTCN,neighbours,objective,controls

def main():
    if (OUT/'local_potential_retired.json').exists():
        raise RuntimeError('Local-potential main predictor retired after failed acceptance gates. Do not start another event-loss tuning round.')
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    protocol=dict(stage='ALL_RELATION_NONRECURSIVE_SINGLE_TRAIN_QUICK_PROBE',source='21',data_run='run224',TRAIN_G=1,all_G=500,
        epoch_budget=32,update_budget=32,wall_budget_seconds=180,seed=1031,learning_rate=.0003,weight_decay=.0001,clip_norm=5,
        architecture='width64 residual depthwise kernel3 local TCN dilations 1,2,4,8',frozen_PTv3=True,
        no_relation_family_root_gates=True,full_population_each_update=True,last_checkpoint_only=True,
        input_scope='all 4080 pure-input geometric families',normalization='TRAIN21 mean/std only, no clipping',
        broader_pilot='Seven TRAIN and BM/10/DEV64 tests require this first feasibility gate; G500 unchanged',
        score_semantics='balanced classification energies, not calibrated probabilities',
        limitations=['cached pure-input upstream components','single TRAIN source not independent test','E16 not implemented in this probe'],start_utc=utc())
    write(dest/'registered_protocol.json',protocol);(dest/'source_snapshot').mkdir()
    for number in (1051,1052,1053,1054,1055):
        for p in (ROOT/'weld_pipeline').glob('*_'+str(number)+'.py'):shutil.copy2(p,dest/'source_snapshot'/p.name)
    model=None;updates=0;history=[];start=None
    try:
        torch.set_num_threads(2);torch.manual_seed(1031);np.random.seed(1031);random.seed(1031)
        write(dest/'controls.json',controls());start=time.perf_counter();write(dest/'startup_pages.json',startup_pages())
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
        model=ContactTCN(raw.shape[1]).cuda();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);check_resources()
        for epoch in range(32):
            if time.perf_counter()-start>=180:break
            check_resources();opt.zero_grad(set_to_none=True);out=model(x,index);loss,parts=objective(out,y,length)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();updates+=1
            row=dict(epoch=epoch+1,updates=updates,loss=float(loss.detach()),gradient_norm=float(norm),parts=parts,elapsed_seconds=time.perf_counter()-start)
            history.append(row);write(dest/'progress.json',row)
            print('EPOCH',epoch+1,'/32','UPDATE',updates,'/32','LOSS',round(row['loss'],5),'POSITION_MM',round(parts['position_mm'],4),'SECONDS',round(row['elapsed_seconds'],2),flush=True)
            del out,loss;opt.zero_grad(set_to_none=True)
            save_checkpoint(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),updates=updates,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),dest/'recovery.pt')
        final={k:v.detach().cpu() for k,v in model.state_dict().items()};change=float(sum((final[k]-initial[k]).double().square().sum() for k in final).sqrt())
        save_checkpoint(dict(architecture='all_relation_contact_tcn',features=raw.shape[1],state_dict=final,updates=updates,data_run=protocol['data_run'],normalization_sha256=digest(dest/'normalization.npz')),dest/'final.pt')
        seconds=time.perf_counter()-start;write(dest/'history.json',history)
        write(dest/'receipt.json',dict(status='CONTACT_TRAIN_COMPLETE' if updates==32 and seconds<=180 else 'CONTACT_TRAIN_INCOMPLETE',
            updates=updates,training_seconds=seconds,parameter_change_l2=change,checkpoint_sha256=digest(dest/'final.pt'),
            source='21',TRAIN_G=1,all_G=500,full_contract_pass=False,end_utc=utc()))
        print('CONTACT_TRAIN_END',updates,'SECONDS',round(seconds,3),flush=True)
    except BaseException:
        if model is not None:save_checkpoint(dict(state_dict=model.state_dict(),updates=updates),dest/'failure_checkpoint.pt')
        write(dest/'failure.json',dict(error=traceback.format_exc(),updates=updates,utc=utc()));raise
if __name__=='__main__':main()
