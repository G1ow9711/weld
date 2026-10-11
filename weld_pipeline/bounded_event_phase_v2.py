"""Generic bounded event features, including candidates with zero sampled rows."""
import numpy as np

from weld_pipeline.bounded_event_phase import CELL_MM, PERIODS_MM, cell_residuals, cell_offsets
from weld_pipeline.neural_physical_event_votes import point_features

VERSION = 'bounded_physical_cell_center_features_v2'
FEATURE_COUNT = 425


def phase_features(sequence, indices):
    indices = np.asarray(indices, np.int64)
    positions = indices.astype(float)*float(sequence['sample_step_mm'])
    coordinates = np.c_[positions, np.asarray(sequence['xyz'], float)[indices]]
    angles = coordinates[:, :, None]*(2.*np.pi/PERIODS_MM)[None, None, :]
    phases = np.mod(positions, CELL_MM)/CELL_MM
    # A fixed column count also preserves the valid (0, 57) empty query table.
    columns = 4*len(PERIODS_MM)
    return np.c_[phases, np.sin(angles).reshape(len(indices), columns),
                 np.cos(angles).reshape(len(indices), columns)].astype(np.float32)


def bounded_features(sequence, data, entry, indices):
    values = np.c_[point_features(sequence, data, entry, indices, True),
                   phase_features(sequence, indices)]
    if values.shape != (len(indices), FEATURE_COUNT) or not np.isfinite(values).all():
        raise ValueError('Finite shared physical phase features required')
    return values.astype(np.float32)
