"""Structured finite-contact paths; one transition definition for logZ and Viterbi.

No target identity or desired count is consumed by inference. Fixed input cell
boundaries are optional events, with energies from adjacent predicted slots.
"""
import itertools
import numpy as np
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_contact_tcn_1052 import ContactTCN

class StructuredContactTCN(ContactTCN):
    """Same width/depth encoder, independent optional input-boundary event head.

    Tying fixed-boundary events to movable events would force duplicate shared
    breaks. Two endpoint heads avoid this representational coupling.
    """
    def __init__(self,features,width=64):
        super().__init__(features,width)
        self.boundary=torch.nn.Linear(width,8)
        torch.nn.init.normal_(self.boundary.weight,std=.01);torch.nn.init.zeros_(self.boundary.bias)
        self.process=torch.nn.Linear(width,26);self.process_fusion=torch.nn.Linear(26,width,bias=False)
        torch.nn.init.normal_(self.process.weight,std=.01);torch.nn.init.zeros_(self.process.bias)
        torch.nn.init.normal_(self.process_fusion.weight,std=.01)
    def forward(self,x,index):
        h=F.silu(self.input(x))
        for k in range(4):
            ids=index[k];valid=ids>=0;safe=ids.clamp_min(0)
            v=h*self.kernel[k,1]
            v=v+h[safe[:,0]]*valid[:,0,None]*self.kernel[k,0]
            v=v+h[safe[:,1]]*valid[:,1,None]*self.kernel[k,2]
            h=self.norm[k](h+F.silu(self.mix[k](v)))
        process=self.process(h);flags=process[:,:2];state=process[:,2:18].reshape(-1,8,2);numeric=F.softplus(process[:,18:])
        code=torch.cat((flags.sigmoid(),state.softmax(-1).flatten(1),numeric),1)
        fused=h+self.process_fusion(code);c=self.classify(fused);fractions=self.position(fused).softmax(-1)
        return dict(occupancy=c[:,:3],event=c[:,3:11].reshape(-1,2,4),direction=c[:,11],closure=c[:,12],weld_type=c[:,13:15],positions=fractions.cumsum(-1)[:,:2],boundary=self.boundary(fused).reshape(-1,2,4),process_flags=flags,process_state=state,process_numeric=numeric,process_code=code)


# Legal marked transitions (previous occupation, next occupation, event bits).
# Connection is an explicit alternative, not an unused auxiliary head.
TRANSITIONS=((0,0,(0,0,0,0)),(0,1,(1,0,0,0)),(0,1,(1,0,0,1)),
             (1,0,(0,1,0,0)),(1,1,(0,0,0,0)),(1,1,(1,1,1,0)),(1,1,(1,1,1,1)))

class StructuredLayout:
    def __init__(self,sequences,n,device='cpu'):
        self.sequences=sequences;self.n=n;prev=np.arange(n,dtype=np.int64)-1;seqids=np.empty(n,np.int64);ends=[]
        buckets={}
        for k,s in enumerate(sequences):
            lo,hi=s['start'],s['end'];seqids[lo:hi]=k;prev[lo]=hi-1 if s['cyclic'] else lo
            ends.append(hi-1);size=1<<(3*(hi-lo)-1).bit_length();buckets.setdefault(size,[]).append(k)
        self.previous=torch.as_tensor(prev,device=device);self.seqids=torch.as_tensor(seqids,device=device)
        self.ends=torch.as_tensor(ends,device=device);self.linear_starts=torch.as_tensor([s['start'] for s in sequences if not s['cyclic']],device=device)
        self.cyclic=torch.as_tensor([s['cyclic'] for s in sequences],device=device)
        self.buckets=[]
        for size,ks in sorted(buckets.items()):
            ids=np.full((len(ks),size),-1,np.int64)
            for j,k in enumerate(ks):
                s=sequences[k];a,b=3*s['start'],3*s['end'];ids[j,:b-a]=np.arange(a,b)
            self.buckets.append((torch.as_tensor(ks,device=device),torch.as_tensor(ids,device=device)))
    def energies(self,out):
        slot=out['event'];fixed=out['boundary'];boundary=(fixed[:,0]+fixed[self.previous,1])*.5
        # Outer boundary has one observed adjacent cell.
        boundary=boundary.index_copy(0,self.linear_starts,fixed[self.linear_starts,0])
        node=torch.stack((boundary,slot[:,0],slot[:,1]),1)
        return out['occupancy'],node,fixed[self.ends,1]


def transition_values(event):
    bits=event.new_tensor([x[2] for x in TRANSITIONS]);return event@bits.T

def matrices(event,semiring='sum'):
    values=transition_values(event);v=[]
    for a,b in ((0,0),(0,1),(1,0),(1,1)):
        ids=[i for i,t in enumerate(TRANSITIONS) if t[:2]==(a,b)];x=values[...,ids]
        v.append(torch.logsumexp(x,-1) if semiring=='sum' else x.max(-1).values)
    return torch.stack(v,-1).reshape(*event.shape[:-1],2,2)

def log_product(a,b):
    return torch.logsumexp(a.unsqueeze(-1)+b.unsqueeze(-3),dim=-2)

def log_partition(out,layout):
    occ,node,terminal=layout.energies(out);mat=matrices(node).reshape(-1,2,2)
    emit=torch.stack((torch.zeros_like(occ),occ),-1).reshape(-1,2)
    mat=mat+emit[:,None,:];terminal=matrices(terminal)
    # Associative reduction: O(log sequence length) graph depth, no per-node Python graph.
    identity=mat.new_tensor([[0.,-1e20],[-1e20,0.]])
    result=mat.new_zeros(len(layout.sequences))
    for ks,ids in layout.buckets:
        m=mat[ids.clamp_min(0)];m=torch.where((ids>=0)[...,None,None],m,identity)
        while m.shape[1]>1:m=log_product(m[:,0::2],m[:,1::2])
        m=m[:,0];linear=torch.logsumexp(m[:,0,:]+terminal[ks,:,0],-1)
        cyclic=torch.logsumexp(torch.stack((m[:,0,0],m[:,1,1]),-1),-1)
        result=result.index_copy(0,ks,torch.where(layout.cyclic[ks],cyclic,linear))
    return result

def target_score(out,layout,target):
    occ,node,terminal=layout.energies(out)
    cell=(occ*target['occupation']).sum(-1)+(node*target['node_event']).sum((-1,-2))
    values=cell.new_zeros(len(layout.sequences)).index_add(0,layout.seqids,cell)
    return values+(terminal*target['terminal_event']).sum(-1)*(~layout.cyclic)

def viterbi(occ,node,cyclic=False):
    """Reference max semiring; preserves selected event/parent-connection marks."""
    occ=np.asarray(occ);node=np.asarray(node);bits=np.array([t[2] for t in TRANSITIONS]);energies=node@bits.T
    best=None;m=len(occ)
    for exterior in ((0,1) if cyclic else (0,)):
        dp=np.full(2,-np.inf);dp[exterior]=0;back=np.zeros((m,2),np.int8)
        for j in range(m):
            nxt=np.full(2,-np.inf)
            for t,(a,b,_) in enumerate(TRANSITIONS):
                score=dp[a]+energies[j,t]+b*occ[j]
                if score>nxt[b]:nxt[b]=score;back[j,b]=t
            dp=nxt
        if cyclic:state=exterior;score=dp[state];lastmark=np.zeros(4)
        else:
            candidates=[(dp[a]+energies[m,t],a,t) for t,(a,b,_) in enumerate(TRANSITIONS) if b==0]
            score,state,last=max(candidates,key=lambda x:x[0]);lastmark=bits[last]
        states=np.zeros(m,np.int8);marks=np.zeros((m+1,4),np.int8);marks[m]=lastmark
        for j in range(m-1,-1,-1):
            states[j]=state;t=back[j,state];marks[j]=bits[t];state=TRANSITIONS[t][0]
        if best is None or score>best[0]:best=(float(score),states,marks)
    return best

def decode(out,cells,sequences,family_ids):
    leaves=[]
    for seq in sequences:
        first,last=seq['start'],seq['end'];lo,hi=cells[first:last,0],cells[first:last,1]
        p=lo[:,None]+(hi-lo)[:,None]*out['positions'][first:last]
        coords=np.r_[np.stack((lo,p[:,0],p[:,1]),1).reshape(-1),hi[-1]]
        slot=out['event'][first:last];fixed=out['boundary'][first:last];bound=(fixed[:,0]+np.roll(fixed[:,1],1,axis=0))*.5
        if not seq['cyclic']:bound[0]=fixed[0,0]
        ev=np.r_[np.stack((bound,slot[:,0],slot[:,1]),1).reshape(-1,4),fixed[-1,1][None]]
        _,states,marks=viterbi(out['occupancy'][first:last].reshape(-1),ev,seq['cyclic']);ci=first+np.arange(len(states))//3
        if seq['cyclic'] and states.any():
            if not states.all():cut=int(np.flatnonzero(states==0)[0])
            elif marks[:-1,2].any():cut=int(np.flatnonzero(marks[:-1,2])[0])
            else:
                leaves.append(dict(family=int(family_ids[first]),lo=float(lo[0]),hi=float(hi[-1]),forward=bool(out['direction'][ci].mean()>=0),weld_type=['horizontal','vertical'][int(out['weld_type'][ci].mean(0).argmax())],parent_connection_at_start=False,source_sequence=seq['family'],closed=True));continue
            length=hi[-1]-lo[0];old=coords[:-1];coords=np.r_[old[cut:],old[:cut]+length,old[cut]+length];states=np.roll(states,-cut);ci=np.roll(ci,-cut);marks=np.r_[np.roll(marks[:-1],-cut,axis=0),marks[cut:cut+1]]
        active=None;start=None
        def append(end):
            if coords[end]<=coords[start]+1e-8:return
            ii=ci[start:end];leaves.append(dict(family=int(family_ids[first]),lo=float(coords[start]),hi=float(coords[end]),forward=bool(out['direction'][ii].mean()>=0),weld_type=['horizontal','vertical'][int(out['weld_type'][ii].mean(0).argmax())],parent_connection_at_start=bool(marks[start,3]),source_sequence=seq['family'],closed=False))
        for j,on in enumerate(states):
            if active is not None and (not on or marks[j,2]):append(j);active=None
            if on and active is None:active=True;start=j
        if active is not None:append(len(states))
    return leaves

def controls():
    torch.manual_seed(9)
    for cyclic in (False,True):
        # Three intervals, all marked legal paths enumerated independently.
        occ=torch.randn(3,dtype=torch.float64,requires_grad=True);event=torch.randn(4,4,dtype=torch.float64,requires_grad=True)
        terms=[];paths=[]
        for choices in itertools.product(range(len(TRANSITIONS)),repeat=3):
            ts=[TRANSITIONS[c] for c in choices]
            if any(ts[i][1]!=ts[i+1][0] for i in range(2)):continue
            if cyclic:
                if ts[-1][1]!=ts[0][0]:continue
                finals=[None]
            else:
                if ts[0][0]!=0:continue
                finals=[j for j,t in enumerate(TRANSITIONS) if t[0]==ts[-1][1] and t[1]==0]
            for final in finals:
                term=sum(event[j]@event.new_tensor(t[2])+t[1]*occ[j] for j,t in enumerate(ts))
                if final is not None:term=term+event[3]@event.new_tensor(TRANSITIONS[final][2])
                terms.append(term)
        exhaustive=torch.logsumexp(torch.stack(terms),0)
        mat=matrices(event[:3])+torch.stack((torch.zeros_like(occ),occ),-1)[:,None,:]
        product=log_product(log_product(mat[0],mat[1]),mat[2])
        analytic=torch.logsumexp(torch.stack((product[0,0],product[1,1])),0) if cyclic else torch.logsumexp(product[0]+matrices(event[3])[:,0],0)
        assert torch.allclose(exhaustive,analytic,atol=1e-10)
        g1=torch.autograd.grad(exhaustive,(occ,event),retain_graph=True);g2=torch.autograd.grad(analytic,(occ,event))
        assert all(torch.allclose(a,b,atol=1e-10) for a,b in zip(g1,g2))
        best=viterbi(occ.detach().numpy(),event.detach().numpy(),cyclic)[0];assert abs(best-float(torch.stack(terms).max()))<1e-10
    return dict(exhaustive_logZ=True,exhaustive_max=True,exhaustive_gradient=True,cyclic_and_open=True,parent_connection_in_state_space=True,optimizer_updates=0)
