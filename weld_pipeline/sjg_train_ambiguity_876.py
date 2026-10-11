"""Train material ambiguity alone on frozen XYZ PTv3 features; no surface gate reuse."""
import argparse
import gc
import json
import time
import traceback
import numpy as np
import torch
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_material_ambiguity_874 import MaterialAmbiguityHead,ambiguity_loss

EPOCHS=8;BATCH=16384;SEED=876


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED);updates=completed=0
    data=OUT/'run78';assert read(data/'receipt.json')['status']=='AMBIGUITY_FEATURES_AND_TRAIN_TARGETS_READY'
    bindings={r['source']:r for r in read(data/'all_inputs_frozen.json')['sources']};targets=read(data/'training_targets.json')['rows']
    expected=read(data/'receipt.json');expected_known=expected['positive']+expected['negative']
    mean=np.zeros(32,np.float64);squared=mean.copy();total=0
    for r in targets:
        assert r['source'] in TRAIN and digest(data/r['file'])==r['sha256']
        b=bindings[r['source']];assert digest(data/b['file'])==b['sha256']
        x=np.load(data/b['file'],mmap_mode='r')
        for start in range(0,len(x),65536):
            chunk=np.asarray(x[start:start+65536],np.float64);mean+=chunk.sum(0);squared+=(chunk*chunk).sum(0);total+=len(chunk)
    assert total==expected['train_points'];mean/=total;std=np.sqrt(np.maximum(squared/total-mean**2,1e-8))
    model=MaterialAmbiguityHead(mean,std).cuda()
    with torch.no_grad():
        model.layers[-1].weight.zero_();model.layers[-1].bias.fill_(np.log(expected['positive']/expected['negative']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    updates_per_epoch=sum((r['positive']+r['negative']+BATCH-1)//BATCH for r in targets)
    protocol=dict(stage='FROZEN_PTV3_SEPARATE_MATERIAL_AMBIGUITY_HEAD',train=TRAIN,development_not_read=VALID,arc_not_read=ARC,
        epochs=EPOCHS,batch_size=BATCH,seed=SEED,updates_per_epoch=updates_per_epoch,expected_total_updates=EPOCHS*updates_per_epoch,
        known_points_per_epoch=expected_known,unknown_points_excluded_from_loss=expected['unknown'],
        positive=expected['positive'],negative=expected['negative'],learning_rate=.001,weight_decay=.0001,
        no_class_reweighting=True,loss='natural-frequency BCE',initial_bias='TRAIN log positive/negative',
        fixed_sigmoid_threshold=.5,backbone_and_all_other_heads_frozen=True,
        pretrained_backbone_sha256=read(data/'registered_protocol.json')['checkpoint_sha256'],
        all_original_known_targets_retained_even_if_voxel_alias=True,no_development_selection=True,
        source_sha256=digest(__file__),model_sha256=digest(ROOT/'weld_pipeline/sjg_material_ambiguity_874.py'),
        full_contract_G=496,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);torch.cuda.reset_peak_memory_stats()
    try:
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,EPOCHS+1):
                started=time.perf_counter();model.train();known=positive=negative=0;loss_sum=0.;rng=np.random.default_rng(SEED+epoch)
                for j in rng.permutation(len(targets)):
                    check_resources();r=targets[int(j)];source=r['source'];b=bindings[source]
                    x=torch.from_numpy(np.load(data/b['file'])).cuda();y=torch.from_numpy(np.load(data/r['file']).astype(np.int64)).cuda()
                    ids=torch.from_numpy(rng.permutation(np.flatnonzero(np.load(data/r['file'])>=0)).astype(np.int64)).cuda()
                    for offset in range(0,len(ids),BATCH):
                        selected=ids[offset:offset+BATCH];optimizer.zero_grad(set_to_none=True);out=model(x[selected]);loss=ambiguity_loss(out,y[selected])
                        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm)
                        optimizer.step();updates+=1;known+=len(selected);positive+=int((y[selected]==1).sum());negative+=int((y[selected]==0).sum())
                        loss_sum+=float(loss.detach())*len(selected)
                    print('MATERIAL_AMBIGUITY_EPOCH',epoch,'/',EPOCHS,'SOURCE',source,'ACTUAL_UPDATES',updates,'KNOWN_POINTS',known,flush=True)
                    del x,y,ids,out,loss;gc.collect();torch.cuda.empty_cache()
                assert known==expected_known and positive==expected['positive'] and negative==expected['negative']
                assert updates==epoch*updates_per_epoch;completed=epoch
                row=dict(epoch=epoch,actual_updates=updates,known_points=known,positive=positive,negative=negative,
                    unknown_retained_outside_loss=expected['unknown'],streaming_mean_loss=loss_sum/known,seconds=time.perf_counter()-started)
                log.write(json.dumps(row)+'\n');log.flush();print('MATERIAL_AMBIGUITY_EPOCH_COMPLETE',row,flush=True)
                with (dest/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,protocol=protocol),h)
        write(dest/'receipt.json',dict(status='SEPARATE_MATERIAL_AMBIGUITY_HEAD_TRAINED',completed_full_epochs=completed,
            actual_optimizer_updates=updates,checkpoint_sha256=digest(dest/'last.pt'),gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,
            backbone_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,completed_full_epochs=completed,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
