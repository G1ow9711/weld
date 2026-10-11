"""Final pooled-inlet trial: macro standardization and balanced gate-risk learning."""
import argparse,gc,os,random,shutil,time,traceback
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_local_event_train_930 import save_checkpoint
from weld_pipeline.sjg_hierarchical_resource_control_1038 import context,new_model,mapped_loader
from weld_pipeline.sjg_balanced_source_step_1045 import source_step,memory
from weld_pipeline.sjg_cuda_startup_pages_1040 import startup_pages
from weld_pipeline.sjg_balanced_hierarchy_1043 import BalancedHierarchicalJointModel

def recovery(dest,model,optimizer,rng,updates,elapsed,binding):
    checkpoint=dict(architecture='balanced_normalized_hierarchy',configuration=model.configuration,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),edge_rng=rng.bit_generator.state,updates=updates,next_epoch=updates//7,
        next_source_index=updates%7,elapsed_seconds=elapsed,binding=binding)
    path=dest/'recovery.tmp.pt';save_checkpoint(checkpoint,path);os.replace(path,dest/'recovery.pt')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args()
    if read(OUT/'hierarchical_round_ledger.json').get('current_scheme_status')=='RETIRED':
        raise RuntimeError('Third trial failed: recursive decoder retired; no further minor retuning')
    dest=OUT/a.run;dest.mkdir(exist_ok=False)
    (dest/'source_snapshot').mkdir()
    for number in (1028,1031,1036,1038,1040,1043,1044,1045,1046):
        for p in (ROOT/'weld_pipeline').glob('*_'+str(number)+'.py'):shutil.copy2(p,dest/'source_snapshot'/p.name)
    protocol=read(OUT/'run207/registered_protocol.json');protocol.update(stage='THIRD_ROUND_BALANCED_NORMALIZED_HIERARCHY',
        initial_checkpoint='run207/initial.pt',resource_control='run212',storage='run210',scientific_parameters_unchanged=False,
        normalization_run='run217',architecture='balanced_normalized_hierarchy',relation_family_loss='class means balanced and divided by ln2',
        relation_family_zero_bias=True,no_positive_class_reweight=False,balanced_class_risk=True,score_semantics='screening scores not calibrated weld probabilities',
        own_page_trimming='one startup call, supported by run211 zero-update CUDA startup probe',start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    model=None;updates=0;start=None;history=[]
    try:
        assert read(OUT/'run212/receipt.json')['status']=='RESOURCE_CONTROL_PASSED'
        manifest,rp,fp,priors=context();torch.set_num_threads(2);torch.manual_seed(1031)
        startup_start=time.perf_counter();write(dest/'startup_pages.json',startup_pages());startup_seconds=time.perf_counter()-startup_start
        assert read(OUT/'run217/receipt.json')['status']=='MACRO_TRAIN_NORMALIZATION_READY'
        statistics=read(OUT/'run217/normalization.json')
        assert digest(OUT/'run217/normalization.json')==read(OUT/'run217/receipt.json')['normalization_sha256']
        state=torch.load(OUT/'run207/initial.pt',map_location='cpu');model=BalancedHierarchicalJointModel(**state['configuration'])
        missing=model.load_state_dict(state['state_dict'],strict=False)
        assert set(missing.missing_keys)=={'relation_input_mean','relation_input_scale','family_input_mean','family_input_scale'} and not missing.unexpected_keys
        model.set_normalization(statistics)
        with torch.no_grad():
            for head in (model.relation_presence,model.family_presence):head.weight.zero_();head.bias.zero_()
        model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);rng=np.random.default_rng(1031)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        binding=dict(data_manifest_sha256=digest(OUT/'run206/training_manifest.json'),source_step_sha256=digest(ROOT/'weld_pipeline/sjg_balanced_source_step_1045.py'),
            initial_sha256=digest(OUT/'run207/initial.pt'),normalization_sha256=digest(OUT/'run217/normalization.json'),storage_receipt_sha256=digest(OUT/'run210/receipt.json'))
        prior_record=read(OUT/'run207/population_priors.json');prior_record.update(gate_initial_bias=0,gate_initial_score=.5,natural_gate_priors_used_only_for_diagnostics=True)
        write(dest/'population_priors.json',prior_record)
        start=time.perf_counter()-startup_seconds;stop=False
        for epoch in range(32):
            for case in manifest['cases']:
                check_resources()
                if time.perf_counter()-start>=180:stop=True;break
                row=source_step(model,opt,mapped_loader(case['source_key']),rng,rp,fp,priors)
                updates+=1;opt.zero_grad(set_to_none=True);gc.collect();row.pop('gradient')
                row.update(epoch=epoch+1,source=case['source_key'],update=updates,elapsed_seconds=time.perf_counter()-start,after_return=memory())
                history.append(row);write(dest/'progress.json',row)
                print('EPOCH',epoch+1,'/32','SOURCE',case['source_key'],'UPDATE',updates,'/224','LOSS',round(row['loss'],5),'SECONDS',round(row['elapsed_seconds'],2),flush=True)
                if updates%7==0:recovery(dest,model,opt,rng,updates,time.perf_counter()-start,binding)
            if stop:break
        final={k:v.detach().cpu() for k,v in model.state_dict().items()}
        change=float(sum((final[k]-initial[k]).double().square().sum() for k in final).sqrt())
        save_checkpoint(dict(architecture='balanced_normalized_hierarchy',configuration=model.configuration,state_dict=final,updates=updates,data_run='run206',binding=binding),dest/'final.pt')
        write(dest/'history.json',history)
        elapsed=time.perf_counter()-start;complete=updates==224 and elapsed<=180
        assert digest(OUT/'run187/final.pt')==manifest['encoder_sha256']
        write(dest/'receipt.json',dict(status='HIERARCHICAL_TRAINING_COMPLETE' if complete else 'TRAINING_BUDGET_INCOMPLETE',updates=updates,
            completed_epochs=updates//7,training_seconds=elapsed,parameter_change_l2=change,final_sha256=digest(dest/'final.pt'),
            scientific_change='fixed TRAIN macro normalization plus balanced-risk relation/family objective and zero biases',
            TRAIN_G=363,full_contract_pass=False,end_utc=utc()))
        print('BOUNDED_TRAINING_END',updates,'SECONDS',round(elapsed,2),'COMPLETE',complete,flush=True)
    except BaseException:
        if model is not None:
            save_checkpoint(dict(architecture='balanced_normalized_hierarchy',configuration=model.configuration,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},updates=updates),dest/'failure_checkpoint.pt')
        write(dest/'failure.json',dict(error=traceback.format_exc(),updates=updates,seconds=None if start is None else time.perf_counter()-start,utc=utc()));raise

if __name__=='__main__':main()
