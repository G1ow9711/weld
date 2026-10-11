"""Matched pair-relation heads, complete known TRAIN pairs, fixed natural BCE."""
import argparse
import gc
import json
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_relation_features_880 import RelationHead,FEATURE_NAMES

ARMS=['distance_only','finite_geometry'];EPOCHS=4;BATCH=65536;SEED=882


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);data=OUT/'run83';expected=read(data/'receipt.json');assert expected['status']=='PAIR_RELATION_INPUTS_AND_TRAIN_TARGETS_READY'
    inputs={r['source']:r for r in read(data/'all_inputs_frozen.json')['sources']};targets=read(data/'training_targets.json')['rows']
    total=0;mean=np.zeros(len(FEATURE_NAMES),np.float64);squared=mean.copy()
    for r in targets:
        assert r['source'] in TRAIN and digest(data/r['file'])==r['sha256'];b=inputs[r['source']];assert digest(data/b['features_file'])==b['features_sha256']
        x=np.load(data/b['features_file'],mmap_mode='r');y=np.load(data/r['file'],mmap_mode='r')
        for i in range(0,len(x),BATCH):
            chunk=np.asarray(x[i:i+BATCH][y[i:i+BATCH]>=0],np.float64);total+=len(chunk);mean+=chunk.sum(0);squared+=(chunk*chunk).sum(0)
    assert total==expected['known_pairs'];mean/=total;std=np.sqrt(np.maximum(squared/total-mean**2,1e-8))
    updates_epoch=sum((r['same']+r['different']+BATCH-1)//BATCH for r in targets)
    protocol=dict(stage='MATCHED_DISTANCE_AND_FINITE_GEOMETRY_RELATION_HEADS',train=TRAIN,development_not_read=VALID,arc_not_read=ARC,
        arms=ARMS,epochs_each=EPOCHS,batch_size=BATCH,seed=SEED,updates_per_epoch=updates_epoch,updates_each=EPOCHS*updates_epoch,
        known_pairs_per_epoch=total,unknown_train_pairs_not_negative=expected['unknown_train_pairs'],same=expected['same'],different=expected['different'],
        feature_names=FEATURE_NAMES,distance_only_masks_columns_after_2=True,same_architecture_and_initialization=True,
        learning_rate=.001,weight_decay=.0001,loss='natural-frequency BCE, no class weights',no_early_stopping_or_development_selection=True,
        backbone_updates=0,source_sha256=digest(__file__),model_sha256=digest(ROOT/'weld_pipeline/sjg_relation_features_880.py'),
        full_contract_G=496,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);arm=None;updates=completed=0
    try:
        for arm in ARMS:
            folder=dest/arm;folder.mkdir();torch.manual_seed(SEED);model=RelationHead(mean,std,arm=='finite_geometry').cuda()
            with torch.no_grad():model.layers[-1].weight.zero_();model.layers[-1].bias.fill_(np.log(expected['different']/expected['same']))
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);updates=completed=0;torch.cuda.reset_peak_memory_stats()
            write(folder/'registered_protocol.json',dict(protocol,arm=arm))
            with (folder/'epochs.jsonl').open('w',encoding='utf-8') as log:
                for epoch in range(1,EPOCHS+1):
                    started=time.perf_counter();model.train();known=same=different=0;loss_sum=0.;rng=np.random.default_rng(SEED+epoch)
                    for index in rng.permutation(len(targets)):
                        check_resources();r=targets[int(index)];b=inputs[r['source']]
                        y_cpu=np.load(data/r['file']);ids_cpu=rng.permutation(np.flatnonzero(y_cpu>=0)).astype(np.int64)
                        x=torch.from_numpy(np.load(data/b['features_file'])).cuda();y=torch.from_numpy(y_cpu.astype(np.int64)).cuda();ids=torch.from_numpy(ids_cpu).cuda()
                        for offset in range(0,len(ids),BATCH):
                            selected=ids[offset:offset+BATCH];optimizer.zero_grad(set_to_none=True);out=model(x[selected]);loss=F.binary_cross_entropy_with_logits(out,y[selected].float())
                            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm)
                            optimizer.step();updates+=1;known+=len(selected);different+=int((y[selected]==1).sum());same+=int((y[selected]==0).sum());loss_sum+=float(loss.detach())*len(selected)
                        print('RELATION_HEAD_EPOCH',arm,epoch,'/',EPOCHS,'SOURCE',r['source'],'ACTUAL_UPDATES',updates,'KNOWN_PAIRS',known,flush=True)
                        del x,y,ids,out,loss,y_cpu,ids_cpu;gc.collect();torch.cuda.empty_cache()
                    assert known==expected['known_pairs'] and same==expected['same'] and different==expected['different']
                    assert updates==epoch*updates_epoch;completed=epoch
                    row=dict(arm=arm,epoch=epoch,actual_updates=updates,known_pairs=known,same=same,different=different,
                        streaming_mean_loss=loss_sum/known,seconds=time.perf_counter()-started)
                    log.write(json.dumps(row)+'\n');log.flush();print('RELATION_HEAD_EPOCH_COMPLETE',row,flush=True)
                    with (folder/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,protocol=protocol,arm=arm),h)
            write(folder/'receipt.json',dict(status='PAIR_RELATION_HEAD_TRAINED',completed_full_epochs=completed,actual_optimizer_updates=updates,
                checkpoint_sha256=digest(folder/'last.pt'),gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,backbone_updates=0,full_contract_pass=False,end_utc=utc()))
            del model,optimizer;gc.collect();torch.cuda.empty_cache()
        write(dest/'receipt.json',dict(status='TWO_PAIR_RELATION_HEADS_TRAINED',actual_optimizer_updates=2*EPOCHS*updates_epoch,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',arm=arm,actual_optimizer_updates=updates,completed_full_epochs=completed,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
