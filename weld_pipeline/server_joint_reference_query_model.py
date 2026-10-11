"""Joint reference-interval occupancy and geometry, not physical absence."""
import torch
from torch import nn
class JointReferenceModel(nn.Module):
 def __init__(self):
  super().__init__();self.trunk=nn.Sequential(nn.Linear(457,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU());self.interval=nn.Linear(128,4);self.reference=nn.Linear(128,1)
 def forward(self,x,q):
  h=self.trunk(x);y=self.interval(h);d=torch.sigmoid(y[:,:2]);bounds=torch.stack((q*(1-d[:,0]),q+(1-q)*d[:,1]),1);return bounds,y[:,2:],self.reference(h).squeeze(1)
