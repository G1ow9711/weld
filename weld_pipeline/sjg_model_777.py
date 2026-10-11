"""Local revision: joint semantics, physical positions, balanced event peaks."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_model_776 import SurfaceJointNet

class SurfaceJointNet777(SurfaceJointNet):
    def __init__(self,node_dim=26,width=96,message_passing=True):
        super().__init__(node_dim,width,message_passing)
        self.register_buffer("node_mean",torch.zeros(node_dim))
        self.register_buffer("node_scale",torch.ones(node_dim))
        self.register_buffer("periods",torch.tensor([16.,32.,64.,128.,256.,512.,1024.,2048.]))
        self.register_buffer("end_scales",torch.tensor([2.,5.,10.,35.,50.,100.]))
        self.local=nn.Sequential(nn.Linear(width+64+48,192),nn.SiLU(),
                                 nn.Linear(192,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.joint_head=nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,1))
        self.message_gate=nn.Parameter(torch.tensor([.1,.1]))
        nn.init.constant_(self.events.bias,-2.19)

    def encode_graph(self,nodes,edges,types):
        h=self.node((nodes-self.node_mean)/self.node_scale)
        if self.message_passing and edges.shape[1]:
            src,dst=edges;typ=F.one_hot(types-1,num_classes=2).to(h.dtype)
            degree=torch.zeros(len(h),device=h.device).index_add_(0,dst,torch.ones(len(dst),device=h.device)).clamp_min(1)
            for i,(layer,norm) in enumerate(zip(self.messages,self.norms)):
                msg=layer(torch.cat([h[src],h[dst],typ],dim=1))
                agg=torch.zeros_like(h).index_add_(0,dst,msg)/degree[:,None]
                h=h+self.message_gate[i].tanh()*F.silu(norm(agg))
        return h

    def decode(self,h,patches,query,query_node):
        p=self.point(patches)
        pooled=torch.cat([p.max(1).values,p.mean(1)],dim=1)
        qmm=query[:,1:3]*1000
        phase=2*math.pi*qmm[:,:,None]/self.periods
        fourier=torch.cat([phase.sin().flatten(1),phase.cos().flatten(1)],dim=1)
        end=torch.exp(-qmm[:,:,None].clamp_min(0)/self.end_scales).flatten(1)
        position=torch.cat([query,fourier,end],dim=1)
        z=self.local(torch.cat([h[query_node],pooled,position],dim=1))
        return self.events(z),self.offsets(z)

    def forward(self,nodes,edges,types,patches,query,query_node):
        h=self.encode_graph(nodes,edges,types)
        self.joint_logits=self.joint_head(h).squeeze(-1)
        return self.decode(h,patches,query,query_node)

def balanced_event_loss(logits,targets):
    p=logits.sigmoid().clamp(1e-5,1-1e-5)
    positive=(targets>=1-1e-6).to(p.dtype)
    negative=(targets<1-1e-6).to(p.dtype)
    pos=-torch.log(p)*(1-p).square()*positive
    neg=-torch.log1p(-p)*p.square()*(1-targets).pow(4)*negative
    return (pos.sum()+neg.sum())/positive.sum().clamp_min(1)
