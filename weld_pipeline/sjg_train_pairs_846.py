"""TRAIN-only interval classification for global endpoint matching."""
import argparse
import json
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_pair_model_845 import IntervalPairModel

DATA=OUT/'run55';EPOCHS=40;STEPS=8;SEED=846


def balanced(y,rng):
    positive=np.flatnonzero(y>0);negative=np.flatnonzero(y==0)
    if not len(positive) or not len(negative):return rng.choice(len(y),256,replace=len(y)<256).astype(np.int64)
    return np.r_[rng.choice(positive,128,replace=len(positive)<128),rng.choice(negative,128,replace=len(negative)<128)].astype(np.int64)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(SEED);updates=completed=0
    protocol=dict(stage='LEARNED_INTERVAL_PAIR_VALIDITY',train=TRAIN,development_not_read=VALID,arc_not_read=ARC,
        epochs=EPOCHS,steps_per_case=STEPS,epoch_definition='Eight balanced256-candidate minibatches per TRAIN case',
        learning_rate=.001,seed=SEED,device='cpu',features=156,normalization_TRAIN_only=True,
        score_for_matching='learned log odds; unmatched0; positive edges only',threshold=.5,
        no_topK=True,no_cross_case_or_profile_matching=True,no_GT_ID_as_feature=True,
        source_sha256=digest(__file__),model_sha256=digest(ROOT/'weld_pipeline/sjg_pair_model_845.py'),
        input_manifest_sha256=digest(DATA/'all_inputs_frozen.json'),full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol)
    try:
        assert read(DATA/'receipt.json')['status']=='ALL_INTERVAL_INPUTS_AND_TRAIN_LABELS_COMPLETE';cases={}
        bindings={r['source']:r for r in read(DATA/'all_inputs_frozen.json')['sources']}
        labels={r['source']:r for r in read(DATA/'receipt.json')['rows']}
        for source in TRAIN:
            file=DATA/bindings[source]['file'];assert digest(file)==bindings[source]['sha256']
            path=DATA/f'source_{source:02d}_labels.npy';assert digest(path)==labels[source]['label_sha256']
            with np.load(file) as z:x=torch.from_numpy(z['features'])
            cases[source]=(x,np.load(path))
        model=IntervalPairModel();all_x=torch.cat([x for x,y in cases.values()])
        model.mean.copy_(all_x.mean(0));model.scale.copy_(all_x.std(0,unbiased=False).clamp_min(.01));del all_x
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        initial={k:v.clone() for k,v in model.state_dict().items()}
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,EPOCHS+1):
                check_resources();rng=np.random.default_rng(SEED+epoch);losses=[];model.train()
                for source in rng.permutation(TRAIN):
                    x,y=cases[int(source)]
                    for _ in range(STEPS):
                        ids=balanced(y,rng);target=torch.from_numpy(y[ids]);prediction=model(x[ids])
                        loss=F.binary_cross_entropy_with_logits(prediction,target);assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                        updates+=1;losses.append(float(loss.detach()))
                completed=epoch;row=dict(epoch=epoch,epochs=EPOCHS,actual_updates=updates,mean_loss=float(np.mean(losses)))
                log.write(json.dumps(row)+'\n');log.flush();print('PAIR_EPOCH',row,flush=True)
        with (dest/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=completed,updates=updates,protocol=protocol),h)
        changed=sum(not torch.equal(v,initial[k]) for k,v in model.state_dict().items());assert changed>0
        write(dest/'receipt.json',dict(status='INTERVAL_PAIR_TRAINING_COMPLETE',actual_optimizer_updates=updates,epochs=completed,
            changed_parameter_tensors=changed,checkpoint_sha256=digest(dest/'last.pt'),full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',actual_optimizer_updates=updates,epochs=completed,error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
