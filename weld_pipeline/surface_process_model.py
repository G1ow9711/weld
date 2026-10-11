"""Shared contact network with surface-restricted point features and cut planes."""
import torch
from torch import nn
from weld_pipeline.typed_contact_model import TypedContactNet


class SurfaceProcessContactNet(nn.Module):
    def __init__(self, features, planes, width=64):
        super().__init__()
        if features != 84:
            raise ValueError('Expected 70 base plus 14 surface feature channels')
        self.input_features = features
        self.register_buffer('plane_axes', torch.tensor([p['axis'] for p in planes], dtype=torch.long))
        self.cut_positions_mm = nn.Parameter(torch.tensor([p['position_mm'] for p in planes], dtype=torch.float32))
        self.register_buffer('scales_mm', torch.tensor([2., 8., 32., 128.]))
        self.net = TypedContactNet(features + len(planes)*5, width).net

    def forward(self, x):
        world = x[:, 67:70].float()*10000.
        delta = world[:, self.plane_axes] - self.cut_positions_mm[None, :, None]
        radial = torch.exp(-.5*(delta[:, :, None]/self.scales_mm[None, None, :, None]).square()).flatten(1, 2)
        signed = (delta/128.).clamp(-1, 1)
        return self.net(torch.cat([x, radial, signed], dim=1))

    def load_process_state(self, state):
        current = self.state_dict()
        for name, value in state.items():
            if name == 'net.0.weight':
                current[name].zero_()
                current[name][:, :70].copy_(value[:, :70])
                current[name][:, 84:].copy_(value[:, 70:])
            else:
                if name not in current or current[name].shape != value.shape:
                    raise ValueError('Incompatible initialization: ' + name)
                current[name].copy_(value)
        self.load_state_dict(current)
