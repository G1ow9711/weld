"""Retain physical face directions in the agreed frame, remove absolute cuts."""
import torch

from weld_pipeline.invariant_horizontal_model import InvariantHorizontal

MODEL_NAME='oriented_point_context_horizontal_v2'


class OrientedContextHorizontal(InvariantHorizontal):
    def forward(self,x):
        local=x.clone();axes=torch.stack([x[:,60:63],x[:,63:66],x[:,57:60]],dim=1)
        local[:,49:52]=(axes*x[:,None,49:52]).sum(2)
        a,b,c,d,e,f=[x[:,i] for i in range(134,140)]
        covariance=torch.stack([torch.stack([a,b,c],dim=1),torch.stack([b,d,e],dim=1),torch.stack([c,e,f],dim=1)],dim=1)
        local[:,52:55]=torch.sqrt(torch.einsum('baip,bijp,bajp->bap',axes,covariance,axes).clamp_min(0.)+1e-12)
        local[:,67:70]=0.;local[:,134:140]=0.
        return self.net(local)


def initialize(state):
    model=OrientedContextHorizontal();current=model.state_dict()
    for name,value in current.items():
        source=state['model']['horizontal.'+name]
        if name=='net.0.weight':
            value.zero_();value[:,:84]=source[:,:84];value[:,49:55]=0.
        else:
            value.copy_(source)
    model.load_state_dict(current);return model


def from_checkpoint(state):
    model=OrientedContextHorizontal(state.get('width',64));model.load_state_dict(state['model'],strict=True);return model
