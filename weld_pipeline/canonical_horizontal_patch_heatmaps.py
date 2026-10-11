"""One deterministic equivalent contact frame for a shared straight-H model."""
import numpy as np

from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.horizontal_independent_heatmaps import HEATMAP_NAMES
from weld_pipeline.horizontal_patch_tree import equivalent_frame, patch_features

VERSION = 'canonical_horizontal_patch_heatmaps_v1'


def canonical_frame(sequence, target=None):
    direction = np.asarray(sequence['direction'], float)
    first_normal = np.asarray(sequence['features'][len(sequence['features'])//2, 60:63], float)
    if (not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8 or
            not np.isfinite(first_normal).all() or np.linalg.norm(first_normal) < 1e-8):
        raise ValueError('Finite nondegenerate candidate tangent and contact normal required')
    reverse = bool(direction[np.argmax(np.abs(direction))] < 0)
    flip = bool(first_normal[np.argmax(np.abs(first_normal))] < 0)
    if target is None:
        target = np.zeros((len(sequence['features']), 7), np.float32)
    transformed, labels = equivalent_frame(sequence, target, reverse=reverse, flip=flip)
    return transformed, labels, {'reverse': reverse, 'flip': flip}


def restore_probability_order(values, transformation):
    result = np.asarray(values, np.float32).copy()
    if transformation['reverse']:
        result = result[::-1].copy()
        result[:, [1, 2]] = result[:, [2, 1]].copy()
    return result


def infer_heatmaps(bundle, sequences, structural, mean, covariance):
    if bundle['config']['version'] != VERSION or set(bundle['heads']) != set(HEATMAP_NAMES):
        raise ValueError('Expected all four canonical fitted heatmap heads')
    values = []
    for index, sequence in enumerate(sequences):
        probabilities = np.zeros((len(sequence['features']), 7), np.float32)
        if sequence['type'] == 'horizontal' and sequence.get('curve_kind') != 'arc':
            seq, _, transformation = canonical_frame(sequence)
            for start in range(0, len(probabilities), 4096):
                indices = np.arange(start, min(start+4096, len(probabilities)))
                x = patch_features(seq, indices, structural[index], mean, covariance)
                for column, name in enumerate(HEATMAP_NAMES):
                    y = bundle['heads'][name].predict(x)
                    if not np.isfinite(y).all():
                        raise ValueError('Nonfinite fitted canonical heatmap')
                    probabilities[indices, column] = np.clip(y, 0., 1.)
            probabilities = restore_probability_order(probabilities, transformation)
        values.append(probabilities)
    return values


def predict_straight(bundle, sequences, structural, mean, covariance):
    return decode_intervals(sequences, infer_heatmaps(bundle, sequences, structural, mean, covariance),
                            threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
