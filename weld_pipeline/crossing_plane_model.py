"""Expose process-plane position only to lines that cross that plane."""
import torch
from weld_pipeline.invariant_surface_model import InvariantSurfaceContactNet

MODEL_NAME = 'crossing_plane_surface'


class CrossingPlaneContactNet(InvariantSurfaceContactNet):
    def forward(self, x):
        world = x[:, 67:70].float() * 10000.
        delta = world[:, self.plane_axes] - self.cut_positions_mm[None, :, None]
        crossing = (x[:, 57 + self.plane_axes].abs() > 1e-4).to(delta.dtype)
        radial = torch.exp(-.5 * (delta[:, :, None] / self.scales_mm[None, None, :, None]).square())
        radial = (radial * crossing[:, :, None]).flatten(1, 2)
        signed = (delta / 128.).clamp(-1, 1) * crossing
        local = x.clone()
        local[:, 49:57] = 0.
        local[:, 67:70] = 0.
        return self.net(torch.cat([local, radial, signed], dim=1))
