"""Matched-budget pairing heads without direct global joint embedding."""
import argparse
import gc
import json
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_train_pairs_846 import balanced,DATA,EPOCHS,STEPS,SEED


class GeometryPairModel(nn.Module):
    def __init__(self,invariant=False):
        super().__init__();self.invariant=invariant
        self.register_buffer('mean',torch.zeros(60));self.register_buffer('scale',torch.ones(60))
        self.network=nn.Sequential(nn.Linear(60,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))

    def features(self,x):
        assert x.shape[-1]==156
        z=x[...,96:].clone()
        if not self.invariant:return z
        tangent=z[...,:3].clone();tangent=tangent/tangent.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        a=z[...,5:8].clone();b=z[...,8:11].clone()
        def local(v):
            axial=(v*tangent).sum(-1)
            radial=(v-(axial[...,None]*tangent)).norm(dim=-1)
            return torch.stack([axial,radial,v[...,2]],-1)
        a=local(a);b=local(b)
        z[...,0]=tangent[...,2].abs();z[...,1]=tangent[...,:2].norm(dim=-1);z[...,2]=0.
        z[...,5:8]=a+b;z[...,8:11]=(a-b).abs()
        return z

    def forward(self,x):return self.network((self.features(x)-self.mean)/self.scale).squeeze(-1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    protocol=dict(stage='JOINT_EMBEDDING_AND_GEOMETRY_PAIR_ABLATION',train=TRAIN,development_not_read=VALID,arc_not_read=ARC,
        arms=['raw_geometry','invariant_geometry'],baseline='run56',features=60,epochs=EPOCHS,
        steps_per_case=STEPS,updates_per_arm=EPOCHS*len(TRAIN)*STEPS,seed=SEED,learning_rate=.001,
        input_manifest_sha256=digest(DATA/'all_inputs_frozen.json'),source_sha256=digest(__file__),
        sampler_source_sha256=digest(ROOT/'weld_pipeline/sjg_train_pairs_846.py'),
        upstream_PTv3_graph_and_events_frozen=True,all_candidates_retained=True,
        invariant_scope='world yaw and two-face order in pairing geometry channels only; not full raw model rotation invariance',
        no_development_checkpoint_selection=True,full_contract_pass=False,start_utc=utc())
    write(dest/'registered_protocol.json',protocol);cases={};current=None;updates=completed=0
    try:
        bindings={r['source']:r for r in read(DATA/'all_inputs_frozen.json')['sources']}
        labels={r['source']:r for r in read(DATA/'receipt.json')['rows']}
        for source in TRAIN:
            path=DATA/bindings[source]['file'];assert digest(path)==bindings[source]['sha256']
            y_path=DATA/f'source_{source:02d}_labels.npy';assert digest(y_path)==labels[source]['label_sha256']
            with np.load(path) as z:x=torch.from_numpy(z['features'])
            cases[source]=(x,np.load(y_path))
        for arm,invariant in (('raw_geometry',False),('invariant_geometry',True)):
            current=dest/arm;current.mkdir();torch.manual_seed(SEED);model=GeometryPairModel(invariant)
            all_x=torch.cat([x for x,y in cases.values()]);values=model.features(all_x)
            model.mean.copy_(values.mean(0));model.scale.copy_(values.std(0,unbiased=False).clamp_min(.01));del all_x,values
            initial={k:v.clone() for k,v in model.state_dict().items()};optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            updates=completed=0;write(current/'registered_protocol.json',dict(protocol,arm=arm))
            with (current/'epochs.jsonl').open('w',encoding='utf-8') as log:
                for epoch in range(1,EPOCHS+1):
                    check_resources();rng=np.random.default_rng(SEED+epoch);losses=[];model.train()
                    for source in rng.permutation(TRAIN):
                        x,y=cases[int(source)]
                        for _ in range(STEPS):
                            ids=balanced(y,rng);loss=F.binary_cross_entropy_with_logits(model(x[ids]),torch.from_numpy(y[ids]));assert torch.isfinite(loss)
                            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                            updates+=1;losses.append(float(loss.detach()))
                    completed=epoch;row=dict(arm=arm,epoch=epoch,epochs=EPOCHS,actual_updates=updates,mean_loss=float(np.mean(losses)))
                    log.write(json.dumps(row)+'\n');log.flush();print('GEOMETRY_PAIR_EPOCH',row,flush=True)
            with (current/'last.pt').open('wb') as h:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                epoch=completed,updates=updates,protocol=protocol,arm=arm,invariant=invariant),h)
            changed=sum(not torch.equal(v,initial[k]) for k,v in model.state_dict().items());assert changed>0
            write(current/'receipt.json',dict(status='INTERVAL_PAIR_TRAINING_COMPLETE',epochs=completed,
                actual_optimizer_updates=updates,checkpoint_sha256=digest(current/'last.pt'),
                changed_parameter_tensors=changed,full_contract_pass=False,end_utc=utc()))
            del model,optimizer,initial;gc.collect()
        write(dest/'receipt.json',dict(status='TWO_GEOMETRY_PAIR_HEADS_TRAINED',actual_optimizer_updates=3840,
            full_contract_pass=False,end_utc=utc()))
    except Exception:
        error=dict(status='FAILED',arm=current.name if current else None,actual_optimizer_updates=updates,
            epochs=completed,error=traceback.format_exc(),end_utc=utc())
        write(dest/'receipt.json',error)
        if current:write(current/'receipt.json',error)
        raise


if __name__=='__main__':main()
