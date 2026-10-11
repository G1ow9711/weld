"""One checkpoint: refit horizontals while preserving the existing vertical branch."""
import torch
from torch import nn
from weld_pipeline.typed_branch_model import TypedBranchContactNet

MODEL_NAME = 'horizontal_refit_branch_surface'


class HorizontalRefitBranchContactNet(TypedBranchContactNet):
    def __init__(self, features, planes, width=64, horizontal_planes=None):
        super().__init__(features, planes, width, horizontal_planes)
        self.horizontal.requires_grad_(True)
        self.vertical.requires_grad_(False)
        self.vertical.eval()

    def train(self, mode=True):
        nn.Module.train(self, mode)
        self.vertical.eval()
        return self

    def forward(self, x):
        vertical = (x[:, 66] > .5).any(dim=-1)
        if bool(vertical.all()):
            with torch.no_grad():
                return self.vertical(x)
        output = self.horizontal(x)
        if bool(vertical.any()):
            with torch.no_grad():
                frozen_output = self.vertical(x[vertical]).to(output.dtype)
            output = output.clone()
            output[vertical] = frozen_output
        return output
