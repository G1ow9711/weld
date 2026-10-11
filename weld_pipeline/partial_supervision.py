"""Explicit unknown supervision for geometry-preserving TRAIN sequence crops."""
import numpy as np


def clean_targets(target):
    target=np.asarray(target)
    unknown=np.all(target==-1,axis=-1)
    if not np.isfinite(target).all() or np.any((target[~unknown]<0)|(target[~unknown]>1)):
        raise ValueError('Targets must be probabilities or all-channel -1 unknown rows')
    return np.where(unknown[:,None],0.,target).astype(np.float32),(~unknown).astype(np.float32)


def restrict_supervision(target,positions,focus_bounds,other_intervals,margin_mm=10.):
    positions=np.asarray(positions);a,b=sorted(focus_bounds)
    valid=(positions>=a-margin_mm)&(positions<=b+margin_mm)
    for lo,hi in other_intervals:
        lo,hi=sorted((lo,hi));valid&=~((positions>=lo-4)&(positions<=hi+4))
    for endpoint in [a,b]:
        if not valid[np.abs(positions-endpoint)<=4].all():
            raise ValueError('Other weld supervision overlaps a required endpoint neighborhood')
    result=target.copy();result[~valid]=-1.
    clean_targets(result)
    return result
