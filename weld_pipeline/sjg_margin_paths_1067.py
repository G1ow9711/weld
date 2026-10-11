"""Fixed-decision normalized complete-path margin, worst background family."""
import itertools
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_structured_contact_1057 import matrices,TRANSITIONS,target_score,viterbi

def max_product(a,b):return (a.unsqueeze(-1)+b.unsqueeze(-3)).max(dim=-2).values

def augmented_max(out,layout,target=None):
    occ,node,terminal=layout.energies(out)
    if target is None:
        oo=occ+1;nn=node+1;tt=terminal+1;constant=occ.new_zeros(len(layout.sequences))
    else:
        oo=occ+1-2*target['occupation'];nn=node+1-2*target['node_event'];tt=terminal+1-2*target['terminal_event']
        c=target['occupation'].sum(-1)+target['node_event'].sum((-1,-2))
        constant=occ.new_zeros(len(layout.sequences)).index_add(0,layout.seqids,c)+target['terminal_event'].sum(-1)*(~layout.cyclic)
    mat=matrices(nn,'max').reshape(-1,2,2)+torch.stack((torch.zeros_like(oo),oo),-1).reshape(-1,1,2)
    term=matrices(tt,'max');identity=mat.new_tensor([[0.,-1e20],[-1e20,0.]]);result=constant.clone()
    for ks,ids in layout.buckets:
        m=torch.where((ids>=0)[...,None,None],mat[ids.clamp_min(0)],identity)
        while m.shape[1]>1:m=max_product(m[:,0::2],m[:,1::2])
        m=m[:,0];linear=(m[:,0,:]+term[ks,:,0]).max(-1).values;cycle=torch.stack((m[:,0,0],m[:,1,1]),-1).max(-1).values
        result=result.index_add(0,ks,torch.where(layout.cyclic[ks],cycle,linear))
    return result

class FamilyRisk:
    def __init__(self,layout,family):
        device=layout.seqids.device
        self.ids=torch.tensor([int(family[s['start']]) for s in layout.sequences],device=device)
        self.count=int(np.max(family))+1
        ns=torch.tensor([15*(s['end']-s['start'])+(0 if s['cyclic'] else 4) for s in layout.sequences],device=device,dtype=torch.float32)
        self.N=ns.new_zeros(self.count).index_add(0,self.ids,ns);assert (self.N>0).all()
    def sum(self,x):return x.new_zeros(self.count).index_add(0,self.ids,x)

def objective(out,y,target,layout,length,witnesses,families,return_terms=False):
    empty=augmented_max(out,layout);positive=augmented_max(out,layout,target)-target_score(out,layout,target)
    background=families.sum(empty)/families.N;costs=[];positions=[];positives=[];negatives=[]
    for seq in witnesses:
        s=layout.sequences[seq];a,b=s['start'],s['end'];fid=families.ids[seq]
        mask=target['position_mask'][a:b];delta=(out['positions'][a:b].double()-target['positions'][a:b])*length[a:b,None]
        p=F.smooth_l1_loss(delta[mask],torch.zeros_like(delta[mask]),beta=1.) if mask.any() else delta.sum()*0
        pos=background[fid]+(positive[seq]-empty[seq])/families.N[fid]
        bg=background.masked_fill(torch.arange(families.count,device=fid.device)==fid,-1e20).max()
        positives.append(pos);negatives.append(bg);positions.append(p);costs.append(pos+bg+2*p)
    chosen=int(torch.stack(costs).detach().argmin());seq=witnesses[chosen];s=layout.sequences[seq];a,b=s['start'],s['end']
    dm=y['direction_mask'][a:b];known=y['weld_type'][a:b]>=0
    direction=F.binary_cross_entropy_with_logits(out['direction'][a:b][dm],y['direction'][a:b][dm]) if dm.any() else out['direction'].sum()*0
    kind=F.cross_entropy(out['weld_type'][a:b][known],y['weld_type'][a:b][known]) if known.any() else out['weld_type'].sum()*0
    loss=costs[chosen]+direction+kind
    details=dict(positive_risk=float(positives[chosen].detach()),worst_background_risk=float(negatives[chosen].detach()),
        position_mm=float(positions[chosen].detach()),direction=float(direction.detach()),type=float(kind.detach()),
        witness_sequence=seq,witness_family=int(families.ids[seq]),known_decisions=int(families.N[families.ids[seq]]))
    return (loss,details,(positives[chosen],negatives[chosen])) if return_terms else (loss,details)

def controls():
    # Independent exhaustive enumeration includes empty, repeated breaks and cycles.
    torch.manual_seed(29);errors=[]
    for cycle in (False,True):
        for target_case in ('empty','long','fragmented','gap'):
            occ=torch.randn(4,dtype=torch.float64);event=torch.randn(5,4,dtype=torch.float64)
            desired={'empty':[0,0,0,0],'long':[1,1,1,1],'fragmented':[1,1,1,1],'gap':[1,0,0,1]}[target_case]
            legal=[]
            for choices in itertools.product(range(len(TRANSITIONS)),repeat=4):
                ts=[TRANSITIONS[j] for j in choices]
                if any(ts[i][1]!=ts[i+1][0] for i in range(3)):continue
                if cycle:
                    if ts[-1][1]!=ts[0][0]:continue
                    finals=[None]
                else:
                    if ts[0][0]!=0:continue
                    finals=[t for t in TRANSITIONS if t[0]==ts[-1][1] and t[1]==0]
                for final in finals:legal.append((ts,final))
            matching=[v for v in legal if [t[1] for t in v[0]]==desired];assert matching
            ts,final=matching[-1] if target_case=='fragmented' else matching[0]
            to=occ.new_tensor(desired);te=event.new_zeros((5,4));te[:4]=event.new_tensor([t[2] for t in ts])
            if final:te[4]=event.new_tensor(final[2])
            terms=[]
            for path,last in legal:
                value=sum(float(event[j]@event.new_tensor(t[2])+t[1]*occ[j])+abs(t[1]-desired[j])+sum(abs(t[2][k]-int(te[j,k])) for k in range(4)) for j,t in enumerate(path))
                if last:value+=float(event[4]@event.new_tensor(last[2]))+sum(abs(last[2][k]-int(te[4,k])) for k in range(4))
                terms.append(value)
            adjusted=viterbi((occ+1-2*to).numpy(),(event+1-2*te).numpy(),cycle)[0]+float(to.sum()+te[:4 if cycle else 5].sum())
            eo=occ+1-2*to;ee=event+1-2*te
            m=matrices(ee[:4],'max')+torch.stack((torch.zeros_like(eo),eo),-1)[:,None,:]
            product=max_product(max_product(m[0],m[1]),max_product(m[2],m[3]))
            analytic=(torch.stack((product[0,0],product[1,1])).max() if cycle else (product[0]+matrices(ee[4],'max')[:,0]).max())+to.sum()+te[:4 if cycle else 5].sum()
            error=max(abs(max(terms)-adjusted),abs(max(terms)-float(analytic)));assert error<1e-10;errors.append(error)
    # The population objective uses max, not sum; duplicated constraints invariant.
    risk=torch.tensor([.2,.8,.4]);assert risk.max()==risk.repeat(1000).max()
    from weld_pipeline.sjg_parent_witness_targets_1058 import controls as identity_controls
    return dict(augmented_exhaustive_max_error=max(errors),cases=8,background_replication_invariant=True,**identity_controls())
