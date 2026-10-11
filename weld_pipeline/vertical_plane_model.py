"""Preserve horizontal process context; remove transverse pose from verticals."""
import torch
from weld_pipeline.invariant_surface_model import InvariantSurfaceContactNet

MODEL_NAME = 'vertical_plane_surface'


class VerticalPlaneContactNet(InvariantSurfaceContactNet):
    def forward(self, x):
        world = x[:, 67:70].float() * 10000.
        delta = world[:, self.plane_axes] - self.cut_positions_mm[None, :, None]
        # One candidate per sample. Include padded context in the same gate.
        # A Z-normal process plane can still cut a vertical candidate.
        vertical = (x[:, 66] > .5).any(dim=-1)
        transverse = self.plane_axes != 2
        keep = (~(vertical[:, None] & transverse[None, :])).to(delta.dtype)
        radial = torch.exp(-.5 * (delta[:, :, None] / self.scales_mm[None, None, :, None]).square())
        radial = (radial * keep[:, :, None, None]).flatten(1, 2)
        signed = (delta / 128.).clamp(-1, 1) * keep[:, :, None]
        local = x.clone()
        local[:, 49:57] = 0.
        local[:, 67:70] = 0.
        return self.net(torch.cat([local, radial, signed], dim=1))
