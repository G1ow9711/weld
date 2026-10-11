"""Fixed seven-case full-population training, at most 224 CUDA updates."""
import argparse,gc,shutil,time,traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_local_event_train_930 import save_checkpoint
from weld_pipeline.sjg_hierarchical_joint_model_1028 import HierarchicalJointModel,initialize_priors


def entropy(prior):
    p=float(prior);return max(1e-8,-p*np.log(p)-(1-p)*np.log1p(-p))


def masked_bce(logits,target,mask):
    known=mask.to(logits.dtype)
    return (F.binary_cross_entropy_with_logits(logits,target,reduction='none')*known).sum()/known.sum().clamp_min(1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--data-run',default='run206');args=ap.parse_args()
    ledger=OUT/'hierarchical_round_ledger.json'
    if ledger.exists() and read(ledger).get('current_scheme_status')=='RETIRED':
        raise RuntimeError('Three failed rounds: recursive hierarchy retired; implement the replacement scheme before further training')
    dest=OUT/args.run;dest.mkdir(exist_ok=False);(dest/'source_snapshot').mkdir()
    for name in ('sjg_hierarchical_joint_model_1028.py','sjg_hierarchical_features_1029.py','sjg_hierarchical_data_1030.py','sjg_hierarchical_train_1031.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,dest/'source_snapshot'/name)
    data=OUT/args.data_run;updates=0;history=[];model=None;start=None
    protocol=dict(stage='SEVEN_CASE_HIERARCHICAL_TRAINING',data_run=args.data_run,epochs=32,max_updates=224,
        training_wall_budget_seconds=180,optimizer='AdamW',lr=3e-4,weight_decay=1e-4,gradient_clip=5,
        device='cuda',all_known_relations_families_teacher_forced_nodes_used_each_source_update=True,
        no_positive_class_reweight=True,final_weights_only=True,TRAIN_G=363,all_G=500,
        path_edges='all eligible CSR edges; 8192 uniform-with-replacement samples per source update; q=1/N',
        PTv3_and_structure_encoder_frozen=True,closure_edge_truth_unknown_masked=True,
        full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        assert read(data/'receipt.json')['status']=='HIERARCHICAL_POPULATION_READY'
        manifest=read(data/'training_manifest.json');cases=manifest['cases'];assert len(cases)==7 and sum(c['G'] for c in cases)==363
        bindings={r['source_key']:r for r in manifest['static_bindings']}
        for case in cases:
            assert digest(data/case['file'])==case['sha256']
            assert digest(data/bindings[case['source_key']]['file'])==bindings[case['source_key']]['sha256']
        torch.set_num_threads(2);torch.manual_seed(1031);rng=np.random.default_rng(1031)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU budget change')
        torch.cuda.set_device(0);device=torch.device('cuda')
        if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('Insufficient free CUDA memory for fixed trial')
        rp=sum(c['relation_positive'] for c in cases)/sum(c['relation_total'] for c in cases)
        fp=sum(c['family_positive'] for c in cases)/sum(c['family_total'] for c in cases)
        counts=np.sum([c['state_counts'] for c in cases],axis=0).astype(float);priors=np.maximum(counts,1)/np.maximum(counts,1).sum()
        first=bindings[cases[0]['source_key']]
        model=HierarchicalJointModel(first['relation_features'],first['family_features'],186).to(device)
        initialize_priors(model,rp,fp,priors)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        save_checkpoint(dict(configuration=model.configuration,state_dict=initial),dest/'initial.pt')
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        write(dest/'population_priors.json',dict(relation=rp,family=fp,interval_states=priors.tolist(),
            relation_entropy=entropy(rp),family_entropy=entropy(fp),population='new full enumerated hierarchy'))
        start=time.perf_counter();budget_hit=False
        for epoch in range(32):
            epoch_losses=[]
            for case in cases:
                check_resources()
                if time.perf_counter()-start>=180:budget_hit=True;break
                key=case['source_key'];before=time.perf_counter()
                with np.load(data/bindings[key]['file'],allow_pickle=False) as z:
                    static={k:torch.from_numpy(z[k]).to(device) for k in z.files}
                with np.load(data/case['file'],allow_pickle=False) as z:
                    t={k:torch.from_numpy(z[k]).to(device) for k in z.files if not k.startswith('link_') and not k.startswith('leaf_')}
                    offsets=z['link_offsets'];targets=z['link_targets'];count=int(offsets[-1])
                    selected=rng.integers(count,size=8192) if count>8192 else np.arange(count)
                    left=np.searchsorted(offsets,selected,side='right')-1;right=targets[selected]
                    leaf_nodes=z['leaf_nodes'];pair=np.c_[leaf_nodes[left],leaf_nodes[right]].astype(np.int64)
                    delta=z['leaf_starts'][right]-z['leaf_ends'][left]
                    node_family=z['node_family'];node_lo=z['node_lo'];node_hi=z['node_hi'];same_family=node_family[pair[:,0]]==node_family[pair[:,1]]
                    edge_x=np.c_[delta/1000,np.linalg.norm(delta,axis=1)/1000,same_family,
                        np.where(same_family,node_lo[pair[:,1]]-node_hi[pair[:,0]],0)/1000].astype(np.float32)
                    same=z['leaf_child'][left]==z['leaf_child'][right];parent=z['leaf_parent'][left]==z['leaf_parent'][right]
                    adjacent=parent&(z['leaf_segment'][right]==z['leaf_segment'][left]+1)
                    gap=np.linalg.norm(z['leaf_true_begin'][right]-z['leaf_true_end'][left],axis=1)
                    edge_y=np.c_[same,parent,adjacent&(gap<=1e-7),adjacent&(gap>1e-7),np.zeros(len(left))].astype(np.float32)
                    edge_mask=np.ones_like(edge_y,dtype=bool);edge_mask[:,4]=False
                    for k,a in dict(link_pairs=pair,link_x=edge_x,link_y=edge_y,link_mask=edge_mask).items():t[k]=torch.from_numpy(a).to(device)
                optimizer.zero_grad(set_to_none=True)
                relations,rl=model.relations(static['relation_x'])
                families,fl=model.families(relations,static['family_relation'],static['family_x'])
                nodes=model.intervals(families,t['node_family'],t['interval_x'])
                losses={}
                losses['relation']=F.binary_cross_entropy_with_logits(rl,t['relation_y'])/entropy(rp)
                losses['family']=F.binary_cross_entropy_with_logits(fl,t['family_y'])/entropy(fp)
                losses['state']=F.cross_entropy(nodes['state_logits'],t['state'])/max(1e-8,float(-(priors*np.log(priors)).sum()))
                mask=t['bounds_mask'][:,None].to(torch.float32)
                losses['bounds']=(F.smooth_l1_loss(nodes['bounds'],t['bounds'],reduction='none')*mask).sum()/(2*mask.sum()).clamp_min(1)
                losses['direction']=masked_bce(nodes['direction_logit'],t['direction'],t['direction_mask'])
                tm=t['weld_type_mask']
                losses['type']=F.cross_entropy(nodes['weld_type_logits'][tm],t['weld_type'][tm]) if tm.any() else nodes['weld_type_logits'].sum()*0
                # Chunk link logits to reduce peak CUDA activation memory;
                # denominator remains the complete known edge population.
                link_loss=rl.sum()*0;den=t['link_mask'].sum().clamp_min(1)
                for offset in range(0,len(t['link_pairs']),8192):
                    sl=slice(offset,offset+8192);logits=model.links(nodes['latent'],t['link_pairs'][sl],t['link_x'][sl])
                    link_loss=link_loss+(F.binary_cross_entropy_with_logits(logits,t['link_y'][sl],reduction='none')*t['link_mask'][sl]).sum()/den
                losses['links']=link_loss;loss=sum(losses.values())
                if not torch.isfinite(loss):raise RuntimeError('Non-finite complete-population loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();torch.cuda.synchronize();updates+=1
                row=dict(epoch=epoch+1,source=key,update=updates,loss=float(loss),parts={k:float(v) for k,v in losses.items()},
                    source_seconds=time.perf_counter()-before,elapsed_seconds=time.perf_counter()-start)
                history.append(row);epoch_losses.append(float(loss));write(dest/'progress.json',row)
                print('EPOCH',epoch+1,'/32','SOURCE',key,'UPDATE',updates,'/224','LOSS',round(float(loss),5),
                    'ELAPSED',round(row['elapsed_seconds'],2),flush=True)
                del static,t,relations,families,nodes,rl,fl,loss,losses,link_loss;gc.collect()
            if budget_hit:break
        elapsed=time.perf_counter()-start;complete=updates==224 and elapsed<=180
        final={k:v.detach().cpu() for k,v in model.state_dict().items()}
        change=float(sum((final[k]-initial[k]).double().square().sum() for k in final).sqrt())
        save_checkpoint(dict(configuration=model.configuration,state_dict=final,data_run=args.data_run,
            data_manifest_sha256=digest(data/'training_manifest.json'),updates=updates,complete_budget=complete),dest/'final.pt')
        write(dest/'history.json',history)
        assert digest(OUT/'run187/final.pt')==manifest['encoder_sha256']
        write(dest/'receipt.json',dict(status='HIERARCHICAL_TRAINING_COMPLETE' if complete else 'TRAINING_BUDGET_INCOMPLETE',
            updates=updates,completed_epochs=updates//7,training_seconds=elapsed,parameter_change_l2=change,
            final_sha256=digest(dest/'final.pt'),frozen_structure_encoder_unchanged=True,TRAIN_G=363,
            test_accuracy_not_yet_measured=True,full_contract_pass=False,end_utc=utc()))
        print('HIERARCHICAL_TRAINING_END','updates',updates,'seconds',round(elapsed,2),'COMPLETE',complete,flush=True)
    except BaseException:
        if model is not None:
            save_checkpoint(dict(configuration=model.configuration,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},updates=updates),dest/'failure_checkpoint.pt')
        write(dest/'failure.json',dict(error=traceback.format_exc(),updates=updates,elapsed=None if start is None else time.perf_counter()-start,utc=utc()));raise


if __name__=='__main__':main()
