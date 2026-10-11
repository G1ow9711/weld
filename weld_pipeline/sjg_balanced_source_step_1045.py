"""Third-round kernel: relation/family balanced risk; other losses unchanged from run213."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_hierarchical_train_1031 import entropy,masked_bce
from weld_pipeline.sjg_balanced_hierarchy_1043 import balanced_gate
from weld_pipeline.sjg_local_runtime_904 import resources

def memory():
    return dict(**resources(),cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved())

def source_step(model,optimizer,loader,rng,rp,fp,priors,capture=False):
    start=time.perf_counter();device=next(model.parameters()).device;trace=[dict(stage='before',**memory())]
    with loader('static') as z:
        static={k:torch.from_numpy(np.array(z[k],copy=True)).to(device) for k in z.files}
    with loader('train') as z:
        t={k:torch.from_numpy(np.array(z[k],copy=True)).to(device) for k in z.files if not k.startswith('link_') and not k.startswith('leaf_')}
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
    trace.append(dict(stage='loaded',**memory()))
    relations,rl=model.relations(static['relation_x'])
    families,fl=model.families(relations,static['family_relation'],static['family_x'])
    nodes=model.intervals(families,t['node_family'],t['interval_x'])
    losses={}
    losses['relation']=balanced_gate(rl,t['relation_y'])
    losses['family']=balanced_gate(fl,t['family_y'])
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
    trace.append(dict(stage='forward',**memory()))
    if not torch.isfinite(loss):raise RuntimeError('Non-finite complete-population loss')
    loss.backward()
    trace.append(dict(stage='backward',**memory()))
    gradient={k:p.grad.detach().cpu().clone() for k,p in model.named_parameters() if p.grad is not None} if capture else None
    torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();torch.cuda.synchronize()

    return dict(loss=float(loss),parts={k:float(v) for k,v in losses.items()},trace=trace,gradient=gradient,seconds=time.perf_counter()-start)
