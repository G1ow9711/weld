"""Four independently fitted point-profile heatmaps with one shared decoder."""
import numpy as np

from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.horizontal_patch_tree import patch_features

VERSION = 'horizontal_independent_patch_heatmaps_v1'
HEATMAP_NAMES = ('foreground', 'start', 'end', 'cut')


def infer_heatmaps(bundle, sequences, structural, mean, covariance):
    heads = bundle['heads']
    if set(heads) != set(HEATMAP_NAMES):
        raise ValueError('All four fitted heatmap heads required')
    probabilities = []
    for i, sequence in enumerate(sequences):
        values = np.zeros((len(sequence['features']), 7), np.float32)
        if sequence['type'] == 'horizontal' and sequence.get('curve_kind') != 'arc':
            for start in range(0, len(values), 4096):
                indices = np.arange(start, min(start + 4096, len(values)))
                features = patch_features(sequence, indices, structural[i], mean, covariance)
                for column, name in enumerate(HEATMAP_NAMES):
                    prediction = heads[name].predict(features)
                    if not np.isfinite(prediction).all():
                        raise ValueError('Nonfinite learned heatmap')
                    values[indices, column] = np.clip(prediction, 0., 1.)
        probabilities.append(values)
    return probabilities


def predict_straight(bundle, sequences, structural, mean, covariance):
    values = infer_heatmaps(bundle, sequences, structural, mean, covariance)
    return decode_intervals(sequences, values, threshold=.5, weld_penalty=2.,
                            mask_weight=.25, event_gap_mm=0.)
