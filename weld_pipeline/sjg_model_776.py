"""Small conditional contact graph + local XYZ encoder. No file/label/CAD I/O."""
import torch
from torch import nn
import torch.nn.functional as F

class SurfaceJointNet(nn.Module):
    def __init__(self,node_dim=26,width=64,message_passing=True):
        super().__init__()
        self.message_passing=message_passing
        self.node=nn.Sequential(nn.Linear(node_dim,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(width*2+2,width),nn.SiLU(),nn.Linear(width,width)) for _ in range(2)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.point=nn.Sequential(nn.Linear(3,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.local=nn.Sequential(nn.Linear(width+64+4,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.events=nn.Linear(64,3)
        self.offsets=nn.Linear(64,2)

    def encode_graph(self,nodes,edges,types):
        h=self.node(nodes)
        if self.message_passing and edges.shape[1]:
            src,dst=edges
            typ=F.one_hot(types-1,num_classes=2).to(h.dtype)
            degree=torch.zeros(len(h),device=h.device).index_add_(0,dst,torch.ones(len(dst),device=h.device)).clamp_min(1)
            for layer,norm in zip(self.messages,self.norms):
                msg=layer(torch.cat([h[src],h[dst],typ],dim=1))
                agg=torch.zeros_like(h).index_add_(0,dst,msg)/degree[:,None]
                h=F.silu(norm(h+agg))
        return h

    def forward(self,nodes,edges,types,patches,query,query_node):
        return self.decode(self.encode_graph(nodes,edges,types),patches,query,query_node)

    def decode(self,h,patches,query,query_node):
        p=self.point(patches)
        pooled=torch.cat([p.max(1).values,p.mean(1)],dim=1)
        z=self.local(torch.cat([h[query_node],pooled,query],dim=1))
        return self.events(z),self.offsets(z)
