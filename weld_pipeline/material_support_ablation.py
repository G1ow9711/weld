"""Destroy independent material signs while retaining the same observed values."""
import numpy as np

from weld_pipeline.signed_material_support import FEATURE_COUNT


def side_sorted(features):
    x = np.asarray(features, np.float32)
    if x.ndim != 2 or x.shape[1] != FEATURE_COUNT:
        raise ValueError('Expected material feature rows')
    result = np.empty_like(x)
    for i, row in enumerate(x):
        rays = row[:84].reshape(2, 3, 2, 7)
        extents = row[84:108].reshape(2, 3, 2, 2)
        peaks = row[108:156].reshape(2, 2, 4, 3)
        planes = []
        for plane in range(2):
            windows = []
            for window in range(3):
                sides = [np.r_[rays[plane,window,s],extents[plane,window,s]] for s in range(2)]
                sides.sort(key=lambda a:tuple(a))
                windows.extend(np.asarray(sides).ravel())
            back = [peaks[plane,s].ravel() for s in range(2)]
            back.sort(key=lambda a:tuple(a))
            planes.append(np.r_[windows,np.asarray(back).ravel()])
        planes.sort(key=lambda a:tuple(a))
        result[i] = np.r_[np.asarray(planes).ravel(),row[156:]]
    return result
