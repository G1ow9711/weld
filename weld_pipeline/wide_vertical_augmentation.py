"""Apply the same frame permutation to both near and wide distance grids."""
import numpy as np
from weld_pipeline.sequence_augmentation import change_frame
from weld_pipeline.vertical_frame_augmentation import swap_vertical_planes
from weld_pipeline.wide_vertical_features import CHANNELS, BASE_CHANNELS, OFFSETS


def change_wide_frame(x, y, valid, reverse=False, flip_normal=False, flip_vertical_cross=False, swap=False):
    if x.shape[0] != CHANNELS:
        raise ValueError('Wide vertical features required')
    extra = x[BASE_CHANNELS:].reshape(len(OFFSETS), len(OFFSETS), -1).copy()
    if swap:
        swap_vertical_planes(x[:BASE_CHANNELS], valid)
        extra = extra.transpose(1, 0, 2)
    change_frame(x[:BASE_CHANNELS], y, valid, reverse=reverse, flip_normal=flip_normal,
                 flip_vertical_cross=flip_vertical_cross)
    if reverse:
        extra = extra[:, :, ::-1]
    if flip_normal:
        extra = extra[:, ::-1]
    if flip_vertical_cross:
        extra = extra[::-1]
    x[BASE_CHANNELS:] = extra.reshape(CHANNELS - BASE_CHANNELS, -1).copy()


def augment_wide_frames(x, y, valid, rng):
    for i in range(len(x)):
        vertical = bool(np.any(x[i, 66] > .5))
        change_wide_frame(x[i], y[i], valid[i], reverse=rng.random() < .5,
                          flip_normal=rng.random() < .5,
                          flip_vertical_cross=vertical and rng.random() < .5,
                          swap=vertical and rng.random() < .5)
