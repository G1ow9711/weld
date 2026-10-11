"""Ablation: local contact geometry plus shared process planes, no pose shortcut."""
import torch
from weld_pipeline.surface_process_model import SurfaceProcessContactNet


class LocalSurfaceProcessContactNet(SurfaceProcessContactNet):
    def forward(self, x):
        # World XYZ participates only in the explicitly learned cut planes.
        world = x[:, 67:70].float()*10000.
        delta = world[:, self.plane_axes] - self.cut_positions_mm[None, :, None]
        radial = torch.exp(-.5*(delta[:, :, None]/self.scales_mm[None, None, :, None]).square()).flatten(1, 2)
        signed = (delta/128.).clamp(-1, 1)
        geometry = torch.cat([x[:, :49], torch.zeros_like(x[:, 49:55]), x[:, 55:67],
                              torch.zeros_like(x[:, 67:70]), x[:, 70:]], dim=1)
        return self.net(torch.cat([geometry, radial, signed], dim=1))
