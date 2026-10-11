"""Continuous joint-coordinate context instead of independent point events."""
import torch
from torch import nn


def configure_precision():
    # Avoid input-length-dependent TF32 convolution rounding at chunk seams.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


class SequenceBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.conv=nn.Conv1d(64,64,5,padding=2*dilation,dilation=dilation)
        self.norm=nn.LayerNorm(64)
        self.out=nn.Linear(64,64)

    def forward(self,x):
        z=self.conv(x.transpose(1,2)).transpose(1,2)
        return x+self.out(torch.nn.functional.gelu(self.norm(z)))


class JointSequenceModel(nn.Module):
    # Six kernel-5 layers: exact receptive radius 2*(1+2+4+8+16+32)=126.
    receptive_radius=126

    def __init__(self):
        super().__init__()
        self.input=nn.Linear(53,64)
        self.condition=nn.Sequential(nn.Linear(96+53+53,128),nn.GELU(),nn.Linear(128,128))
        self.blocks=nn.ModuleList([SequenceBlock(d) for d in (1,2,4,8,16,32)])
        self.output=nn.Sequential(nn.LayerNorm(64),nn.Linear(64,64),nn.GELU(),nn.Linear(64,7))

    def forward(self,x,condition,valid=None):
        scale,bias=self.condition(condition).chunk(2,-1)
        z=self.input(x)*(1+.1*scale[:,None])+bias[:,None]
        if valid is not None:z=z*valid[...,None]
        for block in self.blocks:
            z=block(z)
            if valid is not None:z=z*valid[...,None]
        result=self.output(z)
        return result[...,:3],result[...,3:5],result[...,5:7]

    def infer(self,x,condition,core=2048):
        outputs=[[],[],[]]
        for begin in range(0,x.shape[1],core):
            end=min(x.shape[1],begin+core)
            left=max(0,begin-self.receptive_radius)
            right=min(x.shape[1],end+self.receptive_radius)
            result=self(x[:,left:right],condition)
            for collected,value in zip(outputs,result):collected.append(value[:,begin-left:end-left])
        return tuple(torch.cat(values,1) for values in outputs)
