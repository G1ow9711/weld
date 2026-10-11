"""Matched dense-core control versus additional core-to-context negatives."""
import argparse
import gc
import json
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_train_ptv3_cohort_806 import load_block
from weld_pipeline.sjg_train_ptv3_804 import losses as surface_losses
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
from weld_pipeline.sjg_dense_material_loss_863 import dense_material_loss
from weld_pipeline.sjg_context_negative_867 import context_negative_loss

EPOCHS=2;SEED=869;ARMS=['dense_control','context_negative']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    data=read(OUT/'run73/receipt.json');records=read(OUT/'run73/manifest.json')['records']
    assert data['status']=='TRAIN_CONTEXT_NEGATIVE_PAIRS_FROZEN'
    initial=OUT/'run70/dense_material/last.pt';current=None;updates=completed=0
    protocol=dict(stage='MATCHED_CORE_CONTEXT_NEGATIVE_SUPERVISION',train=TRAIN,development_not_read=VALID,arc_not_read=ARC,
        arms=ARMS,epochs_each=EPOCHS,updates_each=EPOCHS*len(records),seed=SEED,learning_rate=.0001,
        initial_checkpoint=str(initial.relative_to(OUT)),initial_checkpoint_sha256=digest(initial),
        negative_weight=1.,negative_margin=1.,fixed_relation_threshold=.5,all_known_core_dense_loss_retained=True,
        all_other_owners_nearest_target_in_input_context=True,unknown_not_negative=True,
        same_architecture_optimizer_order_surface_loss_and_budget=True,all_arms_before_reference_reads=True,
        source_sha256=digest(__file__),objective_sha256=digest(ROOT/'weld_pipeline/sjg_context_negative_867.py'),
        full_contract_G=496,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        for r in records:
            assert r['source'] in TRAIN and digest(OUT/r['input_file'])==r['input_sha256']
            assert digest(OUT/'run67'/r['material_file'])==r['material_sha256']
            assert digest(OUT/'run73'/r['negative_file'])==r['negative_sha256']
        teachers=[r for r in read(OUT/'run22/label_summary.json') if r['source'] in TRAIN]
        weight=torch.tensor(min(20.,sum(r['boundary_negative'] for r in teachers)/max(1,sum(r['boundary_positive'] for r in teachers))),device='cuda')
        for arm in ARMS:
            current=dest/arm;current.mkdir();torch.manual_seed(SEED);model=SurfaceMaterialModel().cuda()
            with initial.open('rb') as h:state=torch.load(h,map_location='cuda')
            model.load_state_dict(state['model']);del state
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01);updates=completed=0;torch.cuda.reset_peak_memory_stats()
            write(current/'registered_protocol.json',dict(protocol,arm=arm))
            with (current/'epochs.jsonl').open('w',encoding='utf-8') as log:
                for epoch in range(1,EPOCHS+1):
                    model.train();started=time.perf_counter();points=known_points=direct_points=negative_pairs=negative_anchors=0;values=[]
                    for batch,index in enumerate(np.random.default_rng(SEED+epoch).permutation(len(records)),1):
                        if batch%10==1:check_resources()
                        r=records[int(index)];block,y=load_block(OUT/r['input_file'])
                        labels=torch.from_numpy(np.load(OUT/'run67'/r['material_file']).astype(np.int64)).cuda();core=torch.from_numpy(y['core']).cuda()
                        optimizer.zero_grad(set_to_none=True);output=model(block);surface,_=surface_losses(output[:3],y,weight)
                        owner,stats=dense_material_loss(output[3],labels,core);negative=output[3].sum()*0
                        if arm=='context_negative':
                            with np.load(OUT/'run73'/r['negative_file']) as p:
                                pairs=torch.from_numpy(p['pairs']).cuda();group=torch.from_numpy(p['group']).cuda()
                            negative=context_negative_loss(output[3],pairs,group)
                            negative_pairs+=r['negative_pairs'];negative_anchors+=r['unique_core_anchors']
                        loss=surface+owner+negative;assert torch.isfinite(loss);loss.backward()
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);optimizer.step();updates+=1
                        points+=int(core.sum());known_points+=int(((labels>=0)&core).sum());direct_points+=stats['sampled_points']
                        values.append([float(surface.detach()),float(owner.detach()),float(negative.detach())])
                        if batch==1 or batch%40==0 or batch==len(records):print('CONTEXT_MATERIAL_EPOCH',arm,epoch,'/',EPOCHS,'BLOCK',batch,'/',len(records),'ACTUAL_UPDATES',updates,flush=True)
                    assert points==1880869 and known_points==direct_points==1747537
                    assert negative_pairs==(data['negative_pairs_per_epoch'] if arm=='context_negative' else 0)
                    assert negative_anchors==(data['unique_core_anchors_per_epoch'] if arm=='context_negative' else 0)
                    completed=epoch
                    row=dict(arm=arm,epoch=epoch,epochs=EPOCHS,actual_updates=updates,core_points=points,known_material_points=known_points,
                        directly_supervised_material_points=direct_points,local_negative_pairs=negative_pairs,local_negative_core_anchors=negative_anchors,
                        mean_surface_material_negative_loss=np.mean(values,0).tolist(),seconds=time.perf_counter()-started)
                    log.write(json.dumps(row)+'\n');log.flush();print('CONTEXT_MATERIAL_EPOCH_COMPLETE',row,flush=True)
                    with (current/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,protocol=protocol,arm=arm),h)
            write(current/'receipt.json',dict(status='PTV3_MATERIAL_MULTITASK_TRAINING_COMPLETE',completed_full_epochs=completed,
                actual_optimizer_updates=updates,checkpoint_sha256=digest(current/'last.pt'),gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
                full_contract_pass=False,end_utc=utc()))
            del model,optimizer,output,surface,owner,negative,loss;gc.collect();torch.cuda.empty_cache()
        write(dest/'receipt.json',dict(status='TWO_CONTEXT_MATERIAL_ARMS_TRAINED',actual_optimizer_updates=2*EPOCHS*len(records),full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',arm=current.name if current else None,actual_optimizer_updates=updates,
            completed_full_epochs=completed,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
