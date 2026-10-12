"""Query-local XYZ evidence using the original run446-era neighborhood method."""
import numpy as np
from scipy.spatial import cKDTree

from .sjg_global_neighborhood_sequence_1239 import neighborhood


class RawQueryNeighborhood:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        if (self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or len(self.xyz) < 32
                or not np.isfinite(self.xyz).all()):
            raise ValueError('Complete finite XYZ with at least 32 points is required')
        self.tree = cKDTree(self.xyz)

    def features(self, inp, pid, positions_mm):
        curve = inp.curve(int(pid))
        positions = np.asarray(positions_mm, dtype=np.float64)
        if (positions.ndim != 1 or not np.isfinite(positions).all()
                or np.any(positions < 0) or np.any(positions > curve.length)):
            raise ValueError('Query positions must belong to the original finite input curve')
        if not len(positions):
            return np.empty((0, 72), np.float32)
        # Original endpoint context uses the point and its two 50-mm neighbors.
        # These positions depend only on the supplied input curve, never labels.
        locations = positions[:, None] + np.array([-50., 0., 50.])[None]
        if curve.kind == 'arc' and curve.arc.is_closed:
            locations %= curve.length
        else:
            locations = np.clip(locations, 0, curve.length)
        locations = locations.ravel()
        values = neighborhood(self.tree, self.xyz, curve.at(locations),
            curve.tangent(locations), locations, np.full(len(locations), curve.length))
        result = values.reshape(len(positions), 72)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite input-only local point evidence')
        return result
