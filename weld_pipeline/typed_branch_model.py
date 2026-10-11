"""One checkpoint with a frozen horizontal branch and trainable vertical branch."""
import torch
from torch import nn
from weld_pipeline.crossing_plane_model import CrossingPlaneContactNet
from weld_pipeline.vertical_plane_model import VerticalPlaneContactNet

MODEL_NAME = 'typed_branch_surface'


class TypedBranchContactNet(nn.Module):
    def __init__(self, features, planes, width=64, horizontal_planes=None):
        super().__init__()
        self.horizontal = CrossingPlaneContactNet(features, horizontal_planes if horizontal_planes is not None else planes, width=width)
        self.vertical = VerticalPlaneContactNet(features, planes, width=width)
        self.horizontal.requires_grad_(False)
        self.horizontal.eval()

    def train(self, mode=True):
        super().train(mode)
        self.horizontal.eval()
        return self

    def forward(self, x):
        # Channel 66 comes from point-derived candidate geometry, not labels.
        vertical = (x[:, 66] > .5).any(dim=-1)
        if bool(vertical.all()):
            return self.vertical(x)
        with torch.no_grad():
            # Preserve the original batch shape and padding for horizontals.
            output = self.horizontal(x)
        if bool(vertical.any()):
            output = output.clone()
            output[vertical] = self.vertical(x[vertical]).to(output.dtype)
        return output
