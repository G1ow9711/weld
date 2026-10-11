"""Two complete TRAIN epochs of separate material and surface PTv3 objectives."""
import argparse
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
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel,material_loss

EPOCHS=2;SEED=860


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(SEED)
    records=read(OUT/'run67/manifest.json')['records'];data=read(OUT/'run67/receipt.json');updates=completed=0
    protocol=dict(stage='PTV3_SURFACE_AND_MATERIAL_MULTITASK',train=TRAIN,development_not_read=VALID,arc_not_read=ARC,
        epochs=EPOCHS,blocks_per_epoch=len(records),updates=EPOCHS*len(records),core_points_per_epoch=data['original_points'],
        known_material_core_points_per_epoch=data['known_owner_core_points'],unknown_material_core_points_per_epoch=data['unknown_owner_core_points'],
        seed=SEED,learning_rate=.0001,optimizer='new AdamW weight_decay0.01',initial_checkpoint='run26',
        initial_checkpoint_sha256=digest(OUT/'run26/last.pt'),model_source_sha256=digest(ROOT/'weld_pipeline/sjg_material_model_859.py'),
        source_sha256=digest(__file__),material_manifest_sha256=digest(OUT/'run67/manifest.json'),
        loss='unchanged surface instance/normal/boundary objective plus1.0 separate material pull/push loss',
        material_loss_sampling='equal owner groups, at most64 known core points per owner and about2048 total per block',
        relation_evaluation_threshold=.5,embeddings_compared_only_within_same_input_block=True,
        reference_owner_numbers_targets_only=True,no_unknown_as_negative=True,no_development_selection=True,
        full_contract_G=496,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        assert data['status']=='TRAIN_MATERIAL_BLOCK_TARGETS_COMPLETE'
        assert read(OUT/'material_model_859_controls.json')['status']=='PASS'
        for r in records:
            assert r['source'] in TRAIN and digest(OUT/r['input_file'])==r['input_sha256']
            assert digest(OUT/'run67'/r['material_file'])==r['material_sha256']
        model=SurfaceMaterialModel().cuda()
        with (OUT/'run26/last.pt').open('rb') as h:state=torch.load(h,map_location='cuda')
        model.load_surface(state['model']);del state
        initial_head=model.material.weight.detach().clone();initial_backbone=next(model.backbone.parameters()).detach().clone()
        teachers=[r for r in read(OUT/'run22/label_summary.json') if r['source'] in TRAIN]
        weight=torch.tensor(min(20.,sum(r['boundary_negative'] for r in teachers)/max(1,sum(r['boundary_positive'] for r in teachers))),device='cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01);torch.cuda.reset_peak_memory_stats()
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,EPOCHS+1):
                model.train();started=time.perf_counter();points=known_points=0;values=[]
                order=np.random.default_rng(SEED+epoch).permutation(len(records))
                for batch,index in enumerate(order,1):
                    if batch%10==1:check_resources()
                    r=records[int(index)];block,y=load_block(OUT/r['input_file'])
                    material=torch.from_numpy(np.load(OUT/'run67'/r['material_file']).astype(np.int64)).cuda()
                    core=torch.from_numpy(y['core']).cuda();optimizer.zero_grad(set_to_none=True);output=model(block)
                    surface,s=surface_losses(output[:3],y,weight);ownership,m=material_loss(output[3],material,core)
                    loss=surface+ownership;assert torch.isfinite(loss);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);optimizer.step();updates+=1
                    points+=int(core.sum());known_points+=int(((material>=0)&core).sum());values.append([float(surface.detach()),float(ownership.detach()),float(loss.detach())])
                    if batch==1 or batch%20==0 or batch==len(records):
                        print('MATERIAL_EPOCH',epoch,'/',EPOCHS,'BLOCK',batch,'/',len(records),'SOURCE',r['source'],'ACTUAL_UPDATES',updates,'LOSS',values[-1],flush=True)
                assert points==data['original_points'] and known_points==data['known_owner_core_points'];completed=epoch
                row=dict(epoch=epoch,epochs=EPOCHS,actual_updates=updates,core_points=points,known_material_core_points=known_points,
                    unknown_material_core_points=points-known_points,mean_surface_material_total_loss=np.mean(values,0).tolist(),seconds=time.perf_counter()-started)
                log.write(json.dumps(row)+'\n');log.flush();print('MATERIAL_EPOCH_COMPLETE',row,flush=True)
                with (dest/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,protocol=protocol),h)
        assert not torch.equal(initial_head,model.material.weight.detach()) and not torch.equal(initial_backbone,next(model.backbone.parameters()).detach())
        write(dest/'receipt.json',dict(status='PTV3_MATERIAL_MULTITASK_TRAINING_COMPLETE',completed_full_epochs=completed,
            actual_optimizer_updates=updates,checkpoint_sha256=digest(dest/'last.pt'),head_and_backbone_updated=True,
            gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,completed_full_epochs=completed,
            error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
