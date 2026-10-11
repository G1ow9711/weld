"""One source-balanced optimizer update; only newly added heads can change."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.legacy_incremental_loss_396 import event_loss,graded_attribute_loss

def tensor(value,device):
    return torch.from_numpy(np.asarray(value).copy()).to(device)

def attribute_backward(module,features,target,available,device,weight=.25,chunk=128):
    indices=np.flatnonzero(available);parts=[]
    if not len(indices):return 0.,0
    # Every available original row is used, without duplicate point weighting.
    for start in range(0,len(indices),chunk):
        chosen=indices[start:start+chunk]
        yy={k:tensor(v[chosen],device) for k,v in target.items()}
        loss,detail=graded_attribute_loss(module(tensor(features[chosen],device)),yy)
        normalized=loss*(weight*len(chosen)/len(indices));normalized.backward()
        parts.append(float(normalized.detach().cpu()))
    return sum(parts),len(indices)

def shadow_backward(module,items,device,weight=1.,chunk=2048,halo=256,batch=8):
    admitted=[row for row in items if row['background_known'] or row['has_supervised_positive']]
    if not admitted:return 0.,0
    blocks=[]
    for index,row in enumerate(admitted):
        size=len(row['features'])
        for start in range(0,size,chunk):blocks.append((index,start,min(size,start+chunk)))
    total=0.;forwards=0
    # Equal candidate weight within each source. Positive/negative field cells
    # balance separately; an unresolved background region is never a zero GT.
    for offset in range(0,len(blocks),batch):
        group=blocks[offset:offset+batch];size=max(z-a+2*halo for _,a,z in group)
        xx=np.zeros((len(group),84,size),np.float32)
        for bi,(index,a,z) in enumerate(group):
            f=admitted[index]['features'];lo=a-halo;hi=z+halo;l=max(0,lo);r=min(len(f),hi)
            xx[bi,:,l-lo:r-lo]=f[l:r].T
        residual=module(tensor(xx,device));terms=[]
        for bi,(index,a,z) in enumerate(group):
            row=admitted[index];p=np.asarray(row['frozen_probabilities'][a:z,:4],np.float32)
            logits=np.log(np.clip(p,1e-6,1.-1e-6))-np.log1p(-np.clip(p,1e-6,1.-1e-6))
            yy=tensor(row['targets'][a:z],device)
            correction=residual[bi,:4,halo:halo+z-a].transpose(0,1)
            losses=F.binary_cross_entropy_with_logits(tensor(logits,device)+correction,yy,reduction='none')
            bins=[]
            for field in range(4):
                for positive in (False,True):
                    selected=(yy[:,field]>.01)==positive
                    if not row['background_known'] and not positive:continue
                    if bool(selected.any()):bins.append(losses[selected,field].mean())
            if bins:
                value=torch.stack(bins).mean()+.001*correction.square().mean()
                terms.append(value*((z-a)/len(row['features']))/len(admitted))
        if terms:
            value=torch.stack(terms).sum()*weight;assert torch.isfinite(value)
            value.backward();total+=float(value.detach().cpu());forwards+=1
    return total,forwards

def source_update(model,optimizer,source,device):
    model.train();optimizer.zero_grad(set_to_none=True)
    base={key:tensor(value,device) for key,value in source['base'].items()}
    output=model.events(tensor(source['features'],device),base)
    event,detail=event_loss(output,source['parents'],base,
        background_known=tensor(source['background_known'],device),group_ids=tensor(source['group_ids'],device))
    event.backward()
    ordinary,path_rows=attribute_backward(model.path_attributes,source['path_features'],source['path_targets'],
        source['path_available'],device)
    local,local_rows=attribute_backward(model.attributes,source['local_features'],source['local_targets'],
        source['local_available'],device)
    shadow,shadow_forwards=shadow_backward(model.horizontal_residual,source['shadow'],device)
    parameters=model.incremental_parameters()
    assert all(p.grad is None for old in (model.original115,model.original_hybrid_horizontal) for p in old.parameters())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
    norm=torch.nn.utils.clip_grad_norm_(parameters,10.);assert bool(torch.isfinite(norm))
    optimizer.step()
    losses=dict(event_total=float(event.detach().cpu()),whole_path_attribute_weighted=ordinary,
        local_attribute_weighted=local,horizontal_shadow=shadow,preclip_gradient_norm=float(norm.cpu()))
    assert all(math.isfinite(value) for value in losses.values())
    return dict(losses=losses,event_details={key:float(value.detach().cpu()) for key,value in detail.items() if isinstance(value,torch.Tensor)},
        original_G=source['original_G'],original_E16_records=source['original_E16_records'],
        original_local_points=source['original_local_points'],available_whole_path_rows=path_rows,
        masked_background_candidate_count=source.get('masked_background_candidate_count',0),
        available_local_reference_rows=local_rows,representation_miss_children=source['representation_miss_children'],
        event_NN_forwards=1,horizontal_shadow_NN_forwards=shadow_forwards,
        ordinary_attribute_NN_forwards=math.ceil(path_rows/128),local_attribute_NN_forwards=math.ceil(local_rows/128),
        optimizer_steps=1,published_H_unchanged=True,formal_E_records_created=0)
