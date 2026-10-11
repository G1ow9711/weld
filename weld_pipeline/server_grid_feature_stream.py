"""Shared, input-only finite grid features, independent of detector admission.

Grid points are proposals, never observed contacts. Feature rows are selected by
finite input fragments and explicit aliases, not reference endpoints. Streaming
avoids materializing hundreds of feature columns for every grid coordinate.
"""
import numpy as np
from .server_finite_carrier_grid import finite_carrier_grid


def grid_feature_chunks(length_mm, spans_mm, fragment_feature_rows, raw_features,
                        spacing_mm=1., chunk_size=2048):
    spans = np.asarray(spans_mm, dtype=np.float64)
    rows = np.asarray(fragment_feature_rows, dtype=np.int64)
    length = float(length_mm)
    if chunk_size < 1 or spans.ndim != 2 or spans.shape[1] != 2 or not len(spans):
        raise ValueError('Nonempty finite fragment partition and positive chunk size required')
    if rows.shape != (len(spans),) or np.any(rows < 0) or np.any(2*rows+1 >= len(raw_features)):
        raise ValueError('Invalid explicit fragment feature mapping')
    if not np.isfinite(spans).all() or np.any(spans[:,1] <= spans[:,0]):
        raise ValueError('Finite positive fragment widths required')
    if spans[0,0] != 0 or spans[-1,1] != length or not np.array_equal(spans[1:,0], spans[:-1,1]):
        raise ValueError('Fragments must exactly partition the finite carrier')
    grid = finite_carrier_grid(length, np.empty(0), spacing_mm)['positions_mm']
    for begin in range(0, len(grid), chunk_size):
        position = grid[begin:begin+chunk_size]
        # At internal shared boundaries use the right fragment deterministically;
        # at the final endpoint use the final fragment. No extrapolation.
        local = np.minimum(np.searchsorted(spans[:,1], position, side='right'), len(spans)-1)
        feature_rows = rows[local]
        raw = np.asarray(raw_features[np.column_stack((2*feature_rows, 2*feature_rows+1))], dtype=np.float64)
        features = np.column_stack((raw.mean(1), raw.min(1), raw.max(1), position/length)).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError('Nonfinite input features')
        yield dict(grid_ids=np.arange(begin, begin+len(position)), positions_mm=position,
                   feature_rows=feature_rows, features=features,
                   proposal_is_grid=np.ones(len(position), dtype=bool))
