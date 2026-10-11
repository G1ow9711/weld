"""Remove dependence on the arbitrary extent of a point-derived candidate."""
from weld_pipeline.local_surface_model import LocalSurfaceProcessContactNet


class InvariantSurfaceContactNet(LocalSurfaceProcessContactNet):
    def forward(self,x):
        local=x.clone()
        local[:,55:57]=0.
        return super().forward(local)
