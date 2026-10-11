"""Learned face graph, joint relevance and local weld interval event heads."""
import torch
from torch import nn


class FaceMessage(nn.Module):
    def __init__(self, width=96):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(2*width+4,width),nn.GELU(),nn.Linear(width,width))
        self.attention=nn.Linear(width,1)
        self.update=nn.Sequential(nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width)

    def forward(self,h,neighbors,relation):
        own=h[:,None].expand(-1,neighbors.shape[1],-1)
        message=self.message(torch.cat([own,h[neighbors],relation],-1))
        weight=self.attention(message).softmax(1)
        aggregate=(message*weight).sum(1)
        return self.norm(h+self.update(torch.cat([h,aggregate],-1)))


class JointEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.face=nn.Sequential(nn.Linear(46,96),nn.GELU(),nn.Linear(96,96),nn.LayerNorm(96))
        self.layers=nn.ModuleList([FaceMessage(),FaceMessage()])
        self.joint=nn.Sequential(nn.Linear(96*3+16,128),nn.GELU(),nn.Linear(128,96),nn.LayerNorm(96))
        self.relevance=nn.Linear(96,1)
        self.patch=nn.Sequential(nn.Linear(23,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.event=nn.Sequential(nn.Linear(96+64+7,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,5))

    def encode_faces(self,face_x,neighbors,relation):
        h=self.face(face_x)
        for layer in self.layers:h=layer(h,neighbors,relation)
        return h

    def encode_joints(self,h,pairs,geometry):
        a,b=h[pairs[:,0]],h[pairs[:,1]]
        return self.joint(torch.cat([a+b,(a-b).abs(),a*b,geometry],-1))

    def query_events(self,joint,patch,query):
        local=self.patch(patch).amax(1)
        result=self.event(torch.cat([joint,local,query],-1))
        return result[:,:3],result[:,3:]
