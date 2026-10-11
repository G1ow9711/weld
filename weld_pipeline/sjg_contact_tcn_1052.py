"""All-cell nonrecursive TCN and joint binary-chain decoding.

Index gathers implement depthwise noncausal kernel-3 convolution without padding
different physical curves into the same sequence. Inactive cells remain encoded.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def neighbours(sequences,n):
    result=[]
    for d in (1,2,4,8):
        left=np.full(n,-1,np.int64);right=left.copy()
        for seq in sequences:
            lo,hi=seq['start'],seq['end'];ids=np.arange(lo,hi)
            if seq['cyclic']:
                left[ids]=lo+(ids-lo-d)%(hi-lo);right[ids]=lo+(ids-lo+d)%(hi-lo)
            else:
                left[ids]=np.where(ids-d>=lo,ids-d,-1);right[ids]=np.where(ids+d<hi,ids+d,-1)
        result.append(np.c_[left,right])
    return np.stack(result)

class ContactTCN(nn.Module):
    def __init__(self,features,width=64):
        super().__init__();self.input=nn.Linear(features,width)
        self.kernel=nn.Parameter(torch.empty(4,3,width));nn.init.normal_(self.kernel,std=.05)
        self.mix=nn.ModuleList([nn.Linear(width,width) for _ in range(4)])
        self.norm=nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        # 3 subcell occupation energies, 2*(start,end,break,parent-connection),
        # direction, closure and two weld-type logits. Positions are cell-local.
        self.classify=nn.Linear(width,15);self.position=nn.Linear(width,3)
        nn.init.normal_(self.classify.weight,std=.01);nn.init.zeros_(self.classify.bias)
        nn.init.zeros_(self.position.weight);nn.init.zeros_(self.position.bias)
    def forward(self,x,index):
        h=F.silu(self.input(x))
        for k in range(4):
            ids=index[k];valid=ids>=0;safe=ids.clamp_min(0)
            v=h*self.kernel[k,1]
            v=v+h[safe[:,0]]*valid[:,0,None]*self.kernel[k,0]
            v=v+h[safe[:,1]]*valid[:,1,None]*self.kernel[k,2]
            h=self.norm[k](h+F.silu(self.mix[k](v)))
        c=self.classify(h);fractions=self.position(h).softmax(-1)
        return dict(occupancy=c[:,:3],event=c[:,3:11].reshape(-1,2,4),direction=c[:,11],
                    closure=c[:,12],weld_type=c[:,13:15],positions=fractions.cumsum(-1)[:,:2])

def balanced(logits,y,mask=None):
    if mask is not None:logits=logits[mask];y=y[mask]
    if logits.numel()==0:return logits.sum()*0
    loss=F.binary_cross_entropy_with_logits(logits,y,reduction='none');parts=[]
    for value in (0,1):
        selected=(y==value)
        if selected.any():parts.append(loss[selected].mean())
    return torch.stack(parts).mean()

def objective(out,y,length):
    occ=balanced(out['occupancy'],y['occupancy'])
    events=torch.stack([balanced(out['event'][...,i],y['event'][...,i]) for i in range(3)]).mean()
    connect=balanced(out['event'][...,3],y['event'][...,3])
    mask=y['position_mask'];delta=(out['positions'].double()-y['positions'])*length[:,None]
    position=F.smooth_l1_loss(delta[mask],torch.zeros_like(delta[mask]),beta=1.) if mask.any() else out['positions'].sum()*0
    direction=balanced(out['direction'],y['direction'],y['direction_mask'])
    closure=balanced(out['closure'],y['closure'],y['closure_mask'])
    known=y['weld_type']>=0
    types=F.cross_entropy(out['weld_type'][known],y['weld_type'][known]) if known.any() else out['weld_type'].sum()*0
    total=occ+events+connect+2*position+direction+closure+types
    return total,dict(occupancy=float(occ.detach()),events=float(events.detach()),connection=float(connect.detach()),
        position_mm=float(position.detach()),direction=float(direction.detach()),closure=float(closure.detach()),type=float(types.detach()))

def _logsigmoid(x):return -np.logaddexp(0,-np.asarray(x))

def binary_chain(occupation,event,cyclic=False):
    """Maximum full-path log score. No ad hoc confidence sum per fragment.

    occupation has M cells; event has M+1 nodes with start/end/break energies.
    Initial and terminal exterior states are background. Both negative and
    positive event evidence enter transitions; same-child continuation costs
    no invented segment reward. A true shared break is a single node.
    """
    m=len(occupation);emit=np.c_[_logsigmoid(-occupation),_logsigmoid(occupation)]
    pos=_logsigmoid(event[:,:3]);neg=_logsigmoid(-event[:,:3])
    normal=neg.sum(1);begin=pos[:,0]+neg[:,1]+neg[:,2];end=neg[:,0]+pos[:,1]+neg[:,2]
    shared=pos.sum(1);break_selected=shared>normal
    transition=np.empty((m+1,2,2));transition[:,0,0]=normal;transition[:,0,1]=begin
    transition[:,1,0]=end;transition[:,1,1]=np.maximum(normal,shared)
    best=None
    for exterior in ((0,1) if cyclic else (0,)):
        dp=np.full(2,-np.inf);dp[exterior]=0;back=np.zeros((m,2),np.int8)
        for j in range(m):
            score=dp[:,None]+transition[j];back[j]=score.argmax(0);dp=score.max(0)+emit[j]
        if cyclic:
            # Node zero has already scored the ring closure. Last state must
            # equal the state immediately preceding node zero.
            state=exterior;value=float(dp[state])
        else:
            score=dp+transition[m,:,0];state=int(np.argmax(score));value=float(score[state])
        states=np.zeros(m,np.int8)
        for j in range(m-1,-1,-1):states[j]=state;state=int(back[j,state])
        if best is None or value>best[0]:best=(value,states)
    return best[1],break_selected

def decode(out,cells,sequences,family_ids):
    """Decode every sequence; preserve separate semantic families and all FPs."""
    leaves=[]
    for seq in sequences:
        first,last=seq['start'],seq['end'];lo=cells[first:last,0];hi=cells[first:last,1]
        p=lo[:,None]+(hi-lo)[:,None]*out['positions'][first:last]
        # Each cell contributes three subintervals. Fixed cell boundaries cannot
        # fabricate start/end events; predicted two slots provide events.
        nodes=np.stack([lo,p[:,0],p[:,1]],axis=1).reshape(-1)
        nodes=np.r_[nodes,hi[-1]];occupation=out['occupancy'][first:last].reshape(-1)
        event=np.full((len(nodes),4),-30.,np.float64)
        event[1:-1:3]=out['event'][first:last,0];event[2:-1:3]=out['event'][first:last,1]
        states,breaks=binary_chain(occupation,event,seq['cyclic']);cellmap=first+np.arange(len(states))//3
        if seq['cyclic'] and states.any():
            if not states.all():cut=int(np.flatnonzero(states==0)[0])
            elif breaks[:-1].any():cut=int(np.flatnonzero(breaks[:-1])[0])
            else:
                ci=cellmap
                leaves.append(dict(family=int(family_ids[first]),lo=float(lo[0]),hi=float(hi[-1]),
                    forward=bool(out['direction'][ci].mean()>=0),weld_type=['horizontal','vertical'][int(out['weld_type'][ci].mean(0).argmax())],
                    parent_connection_at_start=False,source_sequence=seq['family'],closed=True))
                continue
            circumference=float(hi[-1]-lo[0]);oldnodes=nodes[:-1]
            nodes=np.r_[oldnodes[cut:],oldnodes[:cut]+circumference,oldnodes[cut]+circumference]
            states=np.roll(states,-cut);cellmap=np.roll(cellmap,-cut)
            event=np.r_[np.roll(event[:-1],-cut,axis=0),event[cut:cut+1]]
            breaks=np.r_[np.roll(breaks[:-1],-cut),breaks[cut]]
        active=None;begin_j=None
        for j,on in enumerate(states):
            if on and active is None:active=float(nodes[j]);begin_j=j
            if active is not None and ((not on) or (j>0 and on and breaks[j])):
                stop=float(nodes[j])
                if stop>active+1e-8:
                    inds=np.arange(begin_j,j);ci=cellmap[inds]
                    leaves.append(dict(family=int(family_ids[first]),lo=active,hi=stop,
                        forward=bool(out['direction'][ci].mean()>=0),weld_type=['horizontal','vertical'][int(out['weld_type'][ci].mean(0).argmax())],
                        parent_connection_at_start=bool(event[begin_j,3]>=0),source_sequence=seq['family']))
                active=float(nodes[j]) if on else None;begin_j=j if on else None
        if active is not None and nodes[-1]>active+1e-8:
            ci=cellmap[begin_j:]
            leaves.append(dict(family=int(family_ids[first]),lo=active,hi=float(nodes[-1]),forward=bool(out['direction'][ci].mean()>=0),
                weld_type=['horizontal','vertical'][int(out['weld_type'][ci].mean(0).argmax())],parent_connection_at_start=bool(event[begin_j,3]>=0),source_sequence=seq['family']))
    return leaves

def controls():
    # Deterministic likelihood controls independent of training labels.
    occ=np.array([-12.,12.,12.,-12.,12.,-12.]);ev=np.full((7,4),-12.)
    ev[1,0]=12;ev[3,1]=12;ev[4,0]=12;ev[5,1]=12
    states,br=binary_chain(occ,ev);assert states.tolist()==[0,1,1,0,1,0]
    ev[2,:3]=12;states,br=binary_chain(occ,ev);assert br[2] and states[1] and states[2]
    ev=np.full((7,4),-12.);states,_=binary_chain(np.full(6,-12.),ev);assert not states.any()
    seq=[dict(start=0,end=2,cyclic=False),dict(start=2,end=5,cyclic=False)]
    idx=neighbours(seq,5);assert idx[0,1,1]==-1 and idx[0,2,0]==-1
    states,_=binary_chain(np.full(6,12.),np.full((7,4),-12.),cyclic=True);assert states.all()
    e=np.full((7,4),-12.);e[2,1]=12;e[4,0]=12
    states,_=binary_chain(np.array([12,12,-12,-12,12,12]),e,cyclic=True)
    assert states.tolist()==[1,1,0,0,1,1]
    return dict(background_evidence=True,positive_gap_not_filled=True,shared_break_kept=True,null_path_allowed=True,
                no_convolution_between_unrelated_families=True,circular_chain_and_wrap_gap=True,arc_contract_not_yet_evaluated=True)
