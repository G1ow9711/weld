"""Frozen original horizontal branch plus wide-context vertical branch."""
import torch
from torch import nn
from weld_pipeline.crossing_plane_model import CrossingPlaneContactNet
from weld_pipeline.vertical_plane_model import VerticalPlaneContactNet
from weld_pipeline.wide_vertical_features import BASE_CHANNELS, CHANNELS

MODEL_NAME = 'wide_branch_surface'


class WideBranchContactNet(nn.Module):
    def __init__(self, features, planes, width=64, horizontal_planes=None):
        super().__init__()
        if features != CHANNELS:
            raise ValueError('Expected wide-context feature width')
        self.horizontal = CrossingPlaneContactNet(BASE_CHANNELS, horizontal_planes if horizontal_planes is not None else planes, width=width)
        self.vertical = VerticalPlaneContactNet(BASE_CHANNELS, planes, width=width)
        self.vertical.net[0] = nn.Conv1d(CHANNELS + len(planes) * 5, width, 1)
        self.vertical.input_features = CHANNELS
        self.horizontal.requires_grad_(False)
        self.horizontal.eval()

    def load_vertical_base_state(self, state):
        state = dict(state)
        original = state['net.0.weight']
        expanded = torch.zeros_like(self.vertical.net[0].weight)
        plane_channels = original.shape[1] - BASE_CHANNELS
        if expanded.shape[1] != CHANNELS + plane_channels:
            raise ValueError('Incompatible process-plane encoding')
        expanded[:, :BASE_CHANNELS] = original[:, :BASE_CHANNELS]
        expanded[:, CHANNELS:] = original[:, BASE_CHANNELS:]
        state['net.0.weight'] = expanded
        self.vertical.load_state_dict(state)

    def train(self, mode=True):
        super().train(mode)
        self.horizontal.eval()
        return self

    def forward(self, x):
        vertical = (x[:, 66] > .5).any(dim=-1)
        if bool(vertical.all()):
            return self.vertical(x)
        with torch.no_grad():
            output = self.horizontal(x[:, :BASE_CHANNELS].contiguous())
        if bool(vertical.any()):
            output = output.clone()
            output[vertical] = self.vertical(x[vertical]).to(output.dtype)
        return output
