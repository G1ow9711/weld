"""Frame transformations for 84 base plus 18 cross-surface extent features."""
import numpy as np
from weld_pipeline.sequence_augmentation import change_frame


def change_context_frame(x,y,valid,reverse=False,flip_normal=False,flip_vertical_cross=False):
    if x.shape[0]!=102:raise ValueError('Expected 102 features')
    change_frame(x[:84],y,valid,reverse,flip_normal,flip_vertical_cross)
    if reverse:x[84:]=x[84:,::-1].copy()
    for do_flip,start in [(flip_vertical_cross,84),(flip_normal,93)]:
        if do_flip:
            for channel in range(start,start+9,3):x[[channel,channel+1]]=x[[channel+1,channel]].copy()


def augment_context_batch(x,y,valid,rng):
    for i in range(len(x)):
        vertical=bool(np.any(x[i,66]>.5))
        change_context_frame(x[i],y[i],valid[i],reverse=rng.random()<.5,flip_normal=rng.random()<.5,
                             flip_vertical_cross=vertical and rng.random()<.5)
