"""Single resource replay and storage parity control; never a scientific model."""
import argparse,gc,traceback,time
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_hierarchical_joint_model_1028 import HierarchicalJointModel
from weld_pipeline.sjg_hierarchical_array_store_1036 import ArrayStore
from weld_pipeline.sjg_hierarchical_source_step_1037 import source_step,memory
from weld_pipeline.sjg_cuda_startup_pages_1040 import startup_pages

def context():
    receipt=read(OUT/'run210/receipt.json');assert receipt['status']=='LOSSLESS_ARRAY_STORE_READY'
    manifest=read(OUT/'run206/training_manifest.json');assert digest(OUT/'run206/training_manifest.json')==receipt['data_manifest_sha256']
    for item in receipt['items']:
        folder=OUT/'run210'/item['folder'];assert digest(folder/'arrays.json')==item['manifest_sha256']
        for row in read(folder/'arrays.json')['arrays'].values():assert digest(folder/row['file'])==row['file_sha256']
    p=read(OUT/'run207/population_priors.json')
    return manifest,p['relation'],p['family'],np.asarray(p['interval_states'])

def new_model():
    state=torch.load(OUT/'run207/initial.pt',map_location='cpu');model=HierarchicalJointModel(**state['configuration']).cuda()
    model.load_state_dict(state['state_dict']);return model,torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)

def mapped_loader(key):return lambda kind:ArrayStore(OUT/'run210'/(key+'_'+kind))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1031);rows=[];control_updates=0;stage='binding'
    try:
        manifest,rp,fp,priors=context();case=manifest['cases'][0];b=next(x for x in manifest['static_bindings'] if x['source_key']==case['source_key'])
        write(dest/'startup_pages.json',startup_pages())
        results=[];states=[]
        for kind in ('original_npz','mapped_npy'):
            stage=kind;write(dest/'stage.json',dict(stage=stage,control_updates=control_updates))
            check_resources();model,opt=new_model();rng=np.random.default_rng(1031)
            loader=(lambda which:np.load(OUT/'run206'/(b['file'] if which=='static' else case['file']),allow_pickle=False)) if kind=='original_npz' else mapped_loader(case['source_key'])
            result=source_step(model,opt,loader,rng,rp,fp,priors,capture=True)
            control_updates+=1
            results.append(result);states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            del model,opt;gc.collect();torch.cuda.empty_cache()
        loss_equal=results[0]['loss']==results[1]['loss']
        grad_equal=all(torch.equal(v,results[1]['gradient'][k]) for k,v in results[0]['gradient'].items())
        params_equal=all(torch.equal(v,states[1][k]) for k,v in states[0].items())
        write(dest/'storage_parity.json',dict(loss_bitwise_equal=loss_equal,gradient_bitwise_equal=grad_equal,parameters_after_one_update_bitwise_equal=params_equal))
        assert loss_equal and grad_equal and params_equal,'Storage repair changed fixed one-step computation'
        del results,states,result;gc.collect();torch.cuda.empty_cache()
        model,opt=new_model();rng=np.random.default_rng(1031)
        stage='mapped_two_cycles'
        for cycle in range(2):
            for case in manifest['cases']:
                check_resources();key=case['source_key']
                row=source_step(model,opt,mapped_loader(key),rng,rp,fp,priors)
                control_updates+=1
                opt.zero_grad(set_to_none=True);gc.collect();row.update(cycle=cycle+1,source=key,after_return=memory())
                assert row.pop('gradient') is None
                rows.append(row);write(dest/'progress.json',dict(completed=len(rows),last=row))
                check_resources()
                print('RESOURCE_CONTROL_CYCLE',cycle+1,'/2','SOURCE',key,'STEP',len(rows),'/14','PRIVATE_GIB',round(row['after_return']['private_commit_bytes']/1024**3,3),flush=True)
        write(dest/'receipt.json',dict(status='RESOURCE_CONTROL_PASSED',parity_verified=True,control_updates=16,scientific_updates=0,
            original_source_order_twice=True,all_resource_gates_unchanged=True,traces=rows,seconds=time.perf_counter()-start,full_contract_pass=False,end_utc=utc()))
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),stage=stage,control_updates=control_updates,completed_replay_steps=len(rows),rows=rows,utc=utc()));raise

if __name__=='__main__':main()
