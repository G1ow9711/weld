"""Register and train a parent head on one fixed geometry/interval model."""
import json
import math
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from weld_pipeline.sjg_geometry_776 import read, write, digest
from weld_pipeline.sjg_prepare_776 import ROOT, OUT, TRAIN
from weld_pipeline.sjg_path_781 import ParentLink, free_children, candidate_features, train_examples
from weld_pipeline.sjg_train_776 import check_resources, utc


def prediction_examples(si, predictions, contacts, h, inventory):
    children=free_children(predictions,contacts)
    pairs,x=candidate_features(children,contacts,h)
    coverage=read(OUT/f'source_{si:02d}/coverage_v2.json')['assignments']
    target_path=ROOT/inventory[si]['target']['target_path']
    assert digest(target_path)==inventory[si]['target']['target_sha256']
    target=read(target_path);labels={}
    for ci in {c['contact'] for c in children}:
        ids=[i for i,c in enumerate(children) if c['contact']==ci]
        truth=[a for a in coverage if a['contact']==ci]
        if not truth: continue
        actual=np.asarray([[children[i]['lo'],children[i]['hi']] for i in ids])
        expected=np.asarray([[a['lo'],a['hi']] for a in truth])
        error=np.abs(actual[:,None]-expected[None]).max(2)
        cost=np.c_[np.where(error<=3,error,1e6),np.full((len(ids),len(ids)),1e4)]
        ii,jj=linear_sum_assignment(cost)
        for i,j in zip(ii,jj):
            if j<len(truth) and error[i,j]<=3: labels[ids[i]]=truth[j]
    y=np.zeros((len(pairs),2),np.float32)
    for k,(i,j) in enumerate(pairs):
        a,b=labels.get(int(i)),labels.get(int(j))
        if a is None or b is None: continue
        link=a['parent']==b['parent'] and abs(a['child']-b['child'])==1
        shared=link and target['parents'][a['parent']]['gaps'][min(a['child'],b['child'])]=='SHARED'
        y[k]=[float(link),float(shared)]
    return x,y


def train_parent(predictions,contexts,inventory,out):
    pieces=[]
    for si in TRAIN:
        contacts,h=contexts[si][:2]
        pieces.append(train_examples(si,contacts,h,inventory))
        pieces.append(prediction_examples(si,predictions[si],contacts,h,inventory))
    x=np.concatenate([v[0] for v in pieces]);y=np.concatenate([v[1] for v in pieces])
    assert 0<y[:,0].sum()<len(y)
    batch=128;batches=math.ceil(len(x)/batch);epochs=min(100,2000//batches)
    assert epochs>=1
    protocol=dict(train=TRAIN,epochs=epochs,batch=batch,batches_per_epoch=batches,
        maximum_updates=2000,training_examples=len(x),positive_links=int(y[:,0].sum()),
        learning_rate=.003,threshold=.5,seed=2026100788,source_sha256=digest(__file__),
        holdout_labels_used_for_training=False,full_contract_pass=False)
    write(out/'parent_protocol.json',protocol)
    torch.manual_seed(protocol['seed']);rng=np.random.default_rng(protocol['seed'])
    model=ParentLink(x.shape[1]).cuda()
    model.mean.copy_(torch.from_numpy(x.mean(0)).cuda());model.scale.copy_(torch.from_numpy(x.std(0).clip(.01)).cuda())
    xx=torch.from_numpy(x).cuda();yy=torch.from_numpy(y).cuda()
    weight=torch.tensor(float((len(y)-y[:,0].sum())/y[:,0].sum()),device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001);updates=0
    with (out/'parent_epochs.jsonl').open('w',encoding='utf8') as log:
        for epoch in range(1,epochs+1):
            order=rng.permutation(len(x));loss_sum=0.
            for begin in range(0,len(x),batch):
                ids=torch.as_tensor(order[begin:begin+batch],dtype=torch.long,device='cuda')
                p=model(xx[ids]);t=yy[ids]
                link=F.binary_cross_entropy_with_logits(p[:,0],t[:,0],pos_weight=weight)
                shared=(F.binary_cross_entropy_with_logits(p[:,1],t[:,1],reduction='none')*t[:,0]).sum()/t[:,0].sum().clamp_min(1)
                loss=link+shared
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite parent loss')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                updates+=1;loss_sum+=float(loss.detach())
            status=dict(utc=utc(),epoch=epoch,epochs=epochs,updates=updates,complete_epoch=True,loss=loss_sum/batches)
            log.write(json.dumps(status)+'\n');log.flush();write(out/'parent_progress.json',status)
            print('LOCAL_FRAME_PARENT Epoch',f'{epoch}/{epochs}','Updates',updates,'Loss',round(status['loss'],6),flush=True)
            if epoch%10==0:check_resources()
    with (out/'parent_link.pt').open('wb') as f:torch.save(dict(model=model.state_dict(),dim=x.shape[1],protocol=protocol),f)
    model.eval();return model,updates
