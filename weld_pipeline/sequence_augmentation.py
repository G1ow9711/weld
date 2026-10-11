"""Equivalent sequence frames; never move geometry or process cut locations."""
import numpy as np


def change_frame(x, y, valid, reverse=False, flip_normal=False, flip_vertical_cross=False):
    """Transform a C x L feature block and endpoint labels in-place.

    Reversing sample order swaps start/end and the tangent. Flipping a
    transverse basis permutes the distance grid, leaving world XYZ intact.
    The second cross-axis may only be flipped on a vertical sequence: the
    horizontal depth offsets are asymmetric and cannot be permuted that way.
    """
    if flip_vertical_cross:
        occupied = valid[0] > 0
        if not occupied.any() or not np.all(x[66, occupied] > .5):
            raise ValueError('Second cross-axis flip requires a vertical sequence')
    if reverse:
        x[:] = x[:, ::-1].copy()
        y[:] = y[:, ::-1].copy()
        valid[:] = valid[:, ::-1].copy()
        x[[55, 56]] = x[[56, 55]].copy()
        x[57:60] *= -1
        y[[1, 2]] = y[[2, 1]].copy()
    if flip_normal or flip_vertical_cross:
        grid = x[:49].reshape(7, 7, -1).copy()
        if flip_normal:
            grid = grid[:, ::-1]
            x[60:63] *= -1
        if flip_vertical_cross:
            grid = grid[::-1]
            x[63:66] *= -1
        x[:49] = grid.reshape(49, -1)
        if x.shape[0] == 84:
            if flip_normal:
                x[77:84] = x[77:84][::-1].copy()
            if flip_vertical_cross:
                x[70:77] = x[70:77][::-1].copy()


def augment_batch(x, y, valid, rng):
    for i in range(len(x)):
        is_vertical = bool(np.any(x[i, 66] > .5))
        change_frame(x[i], y[i], valid[i], reverse=rng.random() < .5,
                     flip_normal=rng.random() < .5,
                     flip_vertical_cross=is_vertical and rng.random() < .5)
