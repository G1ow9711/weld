"""Frozen horizontal baseline with a local, distance-only vertical branch."""
import torch
from torch import nn
from weld_pipeline.typed_contact_model import ChannelNorm, Residual
from weld_pipeline.typed_branch_model import TypedBranchContactNet

MODEL_NAME = 'local_axial_branch_surface'
DISTANCE_CHANNELS = list(range(49)) + list(range(70, 84))


class LocalVerticalContactNet(nn.Module):
    """Radius 15 samples (30 mm at 2-mm spacing); no absolute pose inputs."""
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(63, width, 1), ChannelNorm(width), nn.GELU(),
            *[Residual(width, d) for d in [1, 2, 4, 8]], nn.Conv1d(width, 7, 1))

    def forward(self, x):
        return self.net(x[:, DISTANCE_CHANNELS])

    def initialize_from_vertical(self, state):
        """Transfer shared local filters; exclude global planes and late blocks."""
        target = self.state_dict()
        mapped = {}
        for name in target:
            source = name.replace('net.7.', 'net.11.') if name.startswith('net.7.') else name
            value = state[source]
            if name == 'net.0.weight':
                value = value[:, DISTANCE_CHANNELS]
            if value.shape != target[name].shape:
                raise ValueError('Local transfer shape mismatch: ' + name)
            mapped[name] = value.clone()
        self.load_state_dict(mapped, strict=True)


class LocalAxialBranchContactNet(TypedBranchContactNet):
    def __init__(self, features, planes, width=64, horizontal_planes=None):
        if features != 84:
            raise ValueError('Expected original 84-channel point features')
        super().__init__(features, planes, width, horizontal_planes)
        self.vertical = LocalVerticalContactNet(width)
