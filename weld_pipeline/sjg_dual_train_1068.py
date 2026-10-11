"""Fixed eight process updates, freeze, then twenty-four extraction updates."""
import argparse,random,time,traceback,shutil,gc
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_cuda_startup_pages_1040 import startup_pages
from weld_pipeline.sjg_local_event_train_930 import save_checkpoint
from weld_pipeline.sjg_contact_tcn_1052 import neighbours
from weld_pipeline.sjg_structured_contact_1057 import StructuredLayout
from weld_pipeline.sjg_dual_models_1066 import ProcessModel,ExtractionModel,ProcessBags
from weld_pipeline.sjg_margin_paths_1067 import objective,FamilyRisk,controls

def isolation_control():
    a=ProcessModel(7);b=ExtractionModel(9);x=torch.randn(5,7);z=torch.randn(5,9);idx=torch.full((4,5,2),-1,dtype=torch.long)
    code=a(x);semantic_loss=code['process_code'].sum();semantic_loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in a.parameters())
    a.zero_grad(set_to_none=True);before=a(x)['process_code'].detach().clone()
    with torch.no_grad():b.fusion.weight.fill_(.1)
    b(z,idx,a(x)['process_code'])['occupancy'].sum().backward()
    assert all(p.grad is None for p in a.parameters())
    with torch.no_grad():
        for p in b.parameters():
            if p.grad is not None:p.add_(p.grad,alpha=-.001)
    assert torch.equal(before,a(x)['process_code'].detach())
    return dict(process_semantics_has_gradient=True,extraction_to_process_gradient_zero=True,fixed_input_process_output_unchanged=True,scope='software toy, zero scientific updates')

def main():
    if (OUT/'local_potential_retired.json').exists():
        raise RuntimeError('Local-potential main predictor retired after three failed full gates. Frozen inference remains allowed; replace object selection before new training.')
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run245');ap.add_argument('--preflight-only',action='store_true');a=ap.parse_args()
    dest=OUT/a.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    write(dest/'registered_protocol.json',dict(stage='TWO_INDEPENDENT_MODELS',preprocessing='run243',process_sources='seven TRAIN, 222 parents/363 children',
        extraction_source='21',extraction_G=1,all_G=500,process_updates=8,extraction_updates=24,maximum_actual_updates=32,
        budget_seconds=180,seed=1031,AdamW_lr=.0003,weight_decay=.0001,clip=5,third_and_last_local_potential_trial=True,
        full_contract_pass=False,utc=utc()))
    (dest/'source_snapshot').mkdir()
    for n in range(1065,1071):
        for p in (ROOT/'weld_pipeline').glob('*_'+str(n)+'.py'):shutil.copy2(p,dest/'source_snapshot'/p.name)
    updates=0;history=[];start=None
    try:
        write(dest/'controls.json',dict(structure=controls(),isolation=isolation_control()))
        write(dest/'startup_pages.json',startup_pages());torch.manual_seed(1031);np.random.seed(1031);random.seed(1031)
        pre=OUT/'run243';pr=read(pre/'receipt.json');assert pr['status']=='DUAL_PREPROCESS_READY';assert digest(pre/'process_bags.json')==pr['bags_sha256']
        bindings=read(pre/'input_freeze.json');bags=ProcessBags(read(pre/'process_bags.json'),'cuda');inputs={}
        assert digest(pre/'process_normalization.npz')==bindings['normalization_sha256']
        shutil.copy2(pre/'process_normalization.npz',dest/'process_normalization.npz')
        with np.load(pre/'process_normalization.npz') as z:am=z['mean'];ascale=z['scale']
        for binding in bindings['cases']:
            path=pre/binding['file'];assert digest(path)==binding['sha256'];raw=np.load(path,mmap_mode='r')
            inputs[binding['source']]=torch.from_numpy((raw[bags.selected[binding['source']]]-am)/ascale).cuda()
        araw=np.load(pre/'source_21_features.npy',mmap_mode='r');all_a=torch.from_numpy((araw-am)/ascale).cuda()
        data=OUT/'run224';freeze=read(data/'input_freeze.json')
        for name,sha in freeze['files'].items():assert digest(data/name)==sha
        raw=np.load(data/'features.npy',mmap_mode='r')
        shutil.copy2(OUT/'run238/normalization.npz',dest/'normalization.npz')
        with np.load(dest/'normalization.npz') as z:x=torch.from_numpy((raw-z['mean'])/z['scale']).cuda()
        meta=read(data/'input_domain.json');index=torch.from_numpy(neighbours(meta['sequences'],len(raw))).cuda()
        with np.load(data/'input_domain.npz') as z:
            length=torch.from_numpy(z['cells'][:,1]-z['cells'][:,0]).cuda();family=z['family'];cellfamily=torch.as_tensor(family,dtype=torch.long,device='cuda')
        with np.load(data/'labels.npz') as z:y={k:torch.from_numpy(z[k]).cuda() for k in z.files}
        with np.load(OUT/'run238/witness_targets.npz') as z:target={k:torch.from_numpy(z[k]).cuda() for k in z.files}
        witnesses=read(OUT/'run238/witness_manifest.json')['witness_sequences'];layout=StructuredLayout(meta['sequences'],len(raw),'cuda');families=FamilyRisk(layout,family)
        model_a=ProcessModel().cuda();model_b=ExtractionModel().cuda()
        initial_a={k:v.detach().cpu().clone() for k,v in model_a.state_dict().items()};initial_b={k:v.detach().cpu().clone() for k,v in model_b.state_dict().items()}
        with torch.no_grad():before=bags.loss({k:model_a(v) for k,v in inputs.items()})[1]
        write(dest/'process_before.json',before)
        # Full input forward/backward control, no optimizer update or RNG mutation.
        code=model_a(all_a)['process_code'][cellfamily];out=model_b(x,index,code)
        loss,parts,terms=objective(out,y,target,layout,length,witnesses,families,True)
        gradients=[]
        for term in terms:
            g=torch.autograd.grad(term,model_b.input.weight,retain_graph=True)[0];gradients.append(g.detach().flatten())
        loss.backward();assert all(p.grad is None for p in model_a.parameters());assert torch.isfinite(loss)
        write(dest/'gradient_control.json',dict(positive_norm=float(gradients[0].norm()),worst_background_norm=float(gradients[1].norm()),
            cosine=float(torch.nn.functional.cosine_similarity(gradients[0],gradients[1],dim=0)),model_A_gradient_zero=True,optimizer_updates=0,parts=parts))
        model_b.zero_grad(set_to_none=True);del out,loss,code,gradients,terms;gc.collect();check_resources()
        if a.preflight_only:
            write(dest/'receipt.json',dict(status='DUAL_PREFLIGHT_PASS',optimizer_updates=0,utc=utc()));return
        opt_a=torch.optim.AdamW(model_a.parameters(),lr=.0003,weight_decay=.0001);opt_b=torch.optim.AdamW(model_b.parameters(),lr=.0003,weight_decay=.0001)
        start=time.perf_counter()
        for step in range(8):
            if time.perf_counter()-start>=180:raise RuntimeError('Fixed training time budget exceeded')
            check_resources();opt_a.zero_grad(set_to_none=True);loss,metrics=bags.loss({k:model_a(v) for k,v in inputs.items()})
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model_a.parameters(),5.);opt_a.step();updates+=1
            row=dict(epoch=updates,stage='A_PROCESS',loss=float(loss.detach()),metrics=metrics,gradient_norm=float(norm),seconds=time.perf_counter()-start);history.append(row);write(dest/'progress.json',row)
            print('EPOCH',updates,'/32 MODEL_A',step+1,'/8 LOSS',round(row['loss'],5),'SECONDS',round(row['seconds'],2),flush=True)
        opt_a.zero_grad(set_to_none=True)
        for p in model_a.parameters():p.requires_grad_(False)
        model_a.eval()
        with torch.no_grad():
            after=bags.loss({k:model_a(v) for k,v in inputs.items()})[1];frozen_code=model_a(all_a)['process_code'].detach().clone()
        write(dest/'process_after.json',after)
        code=frozen_code[cellfamily]
        for step in range(24):
            if time.perf_counter()-start>=180:raise RuntimeError('Fixed training time budget exceeded')
            check_resources();opt_b.zero_grad(set_to_none=True);out=model_b(x,index,code);loss,parts=objective(out,y,target,layout,length,witnesses,families)
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in model_a.parameters())
            norm=torch.nn.utils.clip_grad_norm_(model_b.parameters(),5.);opt_b.step();updates+=1
            row=dict(epoch=updates,stage='B_EXTRACTION',loss=float(loss.detach()),parts=parts,gradient_norm=float(norm),seconds=time.perf_counter()-start);history.append(row);write(dest/'progress.json',row)
            print('EPOCH',updates,'/32 MODEL_B',step+1,'/24 LOSS',round(row['loss'],5),'WITNESS',parts['witness_family'],'SECONDS',round(row['seconds'],2),flush=True)
            del out,loss;opt_b.zero_grad(set_to_none=True)
        with torch.no_grad():assert torch.equal(frozen_code,model_a(all_a)['process_code'])
        write(dest/'process_isolation_actual.json',dict(fixed_input_outputs_bitwise_unchanged_during_24_B_updates=True,extraction_gradient_to_A_zero=True))
        final_a={k:v.detach().cpu() for k,v in model_a.state_dict().items()};final_b={k:v.detach().cpu() for k,v in model_b.state_dict().items()}
        save_checkpoint(dict(state_dict=final_a,features=669,preprocessing='run243',updates=8,normalization_sha256=digest(dest/'process_normalization.npz')),dest/'model_A.pt')
        save_checkpoint(dict(state_dict=final_b,features=671,data_run='run224',updates=24,normalization_sha256=digest(dest/'normalization.npz'),model_A_sha256=digest(dest/'model_A.pt')),dest/'final.pt')
        seconds=time.perf_counter()-start;write(dest/'history.json',history)
        write(dest/'receipt.json',dict(status='CONTACT_TRAIN_COMPLETE' if updates==32 and seconds<=180 else 'CONTACT_TRAIN_INCOMPLETE',updates=updates,
            training_seconds=seconds,checkpoint_sha256=digest(dest/'final.pt'),model_A_sha256=digest(dest/'model_A.pt'),
            parameter_change_A_l2=float(sum((final_a[k]-initial_a[k]).double().square().sum() for k in final_a).sqrt()),
            parameter_change_B_l2=float(sum((final_b[k]-initial_b[k]).double().square().sum() for k in final_b).sqrt()),
            semantic_before=before,semantic_after=after,all_G=500,full_contract_pass=False,utc=utc()))
        print('DUAL_TRAIN_END',updates,'SECONDS',round(seconds,3),flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),actual_updates=updates,history=history,utc=utc()));raise

if __name__=='__main__':main()
