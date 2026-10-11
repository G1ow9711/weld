"""Equivalent exchange of the two contact-plane axes of a vertical line."""
import numpy as np
from weld_pipeline.sequence_augmentation import change_frame


def swap_vertical_planes(x, valid):
    if x.shape[0] != 84:
        raise ValueError('Expected 84-channel surface features')
    occupied = valid[0] > 0
    if not occupied.any() or not np.all(x[66, occupied] > .5):
        raise ValueError('Plane exchange is valid only for vertical sequences')
    # Both transverse grids use the same OFFSETS on vertical candidates.
    # XYZ, tangent, sample order, target channels, and process planes stay fixed.
    x[:49] = x[:49].reshape(7, 7, -1).transpose(1, 0, 2).reshape(49, -1).copy()
    x[60:66] = x[60:66].reshape(2, 3, -1)[::-1].reshape(6, -1).copy()
    x[70:84] = x[70:84].reshape(2, 7, -1)[::-1].reshape(14, -1).copy()


def augment_complete_frames(x, y, valid, rng):
    for i in range(len(x)):
        is_vertical = bool(np.any(x[i, 66] > .5))
        if is_vertical and rng.random() < .5:
            swap_vertical_planes(x[i], valid[i])
        change_frame(x[i], y[i], valid[i], reverse=rng.random() < .5,
                     flip_normal=rng.random() < .5,
                     flip_vertical_cross=is_vertical and rng.random() < .5)
