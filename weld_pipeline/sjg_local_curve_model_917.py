"""Finite-face message passing and typed local curve occupancy/event heads."""
import torch
from torch import nn
from weld_pipeline.sjg_joint_model_817 import FaceMessage

class LocalCurveEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.face=nn.Sequential(nn.Linear(46,96),nn.GELU(),nn.Linear(96,96),nn.LayerNorm(96))
        self.layers=nn.ModuleList([FaceMessage(),FaceMessage()])
        self.joint=nn.Sequential(nn.Linear(96*3+20,128),nn.GELU(),nn.Linear(128,96),nn.LayerNorm(96))
        self.patch=nn.Sequential(nn.Linear(23,64),nn.GELU(),nn.Linear(64,64),nn.GELU())
        self.event=nn.Sequential(nn.Linear(96+64+10,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,7))

    def encode_faces(self,face_x,neighbors,relation):
        h=self.face(face_x)
        for layer in self.layers:h=layer(h,neighbors,relation)
        return h

    def encode_curves(self,faces,pairs,geometry):
        a=faces[pairs[:,0]]
        # A boundary curve with one observed face does not invent a second
        # contact/material. The explicit availability flag masks that input.
        b=faces[pairs[:,1]]*geometry[:,7:8]
        return self.joint(torch.cat([a+b,(a-b).abs(),a*b,geometry],dim=-1))

    def query_events(self,joint,patch,query):
        local=self.patch(patch).amax(1)
        output=self.event(torch.cat([joint,local,query],dim=-1))
        return output[:,:5],output[:,5:]

    def forward(self,data):
        faces=self.encode_faces(data['face_x'],data['neighbors'],data['relation'])
        # Compute only queried carriers during training; no loss changes.
        nodes,inverse=torch.unique(data['nodes'],sorted=True,return_inverse=True)
        joint=self.encode_curves(faces,data['pairs'][nodes],data['geometry'][nodes])
        return self.query_events(joint[inverse],data['patch'],data['query'])
