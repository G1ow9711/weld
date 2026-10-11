"""Frame-averaged pure local descriptors; no world-axis fingerprint input."""
import numpy as np

from weld_pipeline.candidate_tree_model import descriptor as base_descriptor
from weld_pipeline.sequence_augmentation import change_frame
from weld_pipeline.vertical_frame_augmentation import swap_vertical_planes

FEATURE_VERSION = 'local_candidate_equivalent_frames_v1'


def descriptor(sequence):
    original = base_descriptor(sequence)
    # Drop global tangent/normal coordinates, retaining local-grid geometry.
    return np.r_[original[0], original[10:]].astype(np.float32)


def frames(sequence):
    for swap in [False, True]:
        for first in [False, True]:
            for second in [False, True]:
                for reverse in [False, True]:
                    feature = np.asarray(sequence['features']).T.copy()
                    valid = np.ones((1, feature.shape[1]), np.float32)
                    target = np.zeros((7, feature.shape[1]), np.float32)
                    if swap:
                        swap_vertical_planes(feature, valid)
                    change_frame(feature, target, valid, reverse=reverse, flip_normal=first,
                                 flip_vertical_cross=second)
                    yield {**sequence, 'features': feature.T}, reverse


def predict_vertical(bundle, sequences):
    selected = [(i, seq) for i, seq in enumerate(sequences) if seq['type'] == 'vertical']
    if not selected:
        return [], []
    transformed = [[(descriptor(seq), reverse) for seq, reverse in frames(s)] for i, s in selected]
    x = np.stack([f for row in transformed for f, reverse in row])
    foreground = bundle['foreground'].predict_proba(x)[:, list(bundle['foreground'].classes_).index(1)]
    foreground = foreground.reshape(len(selected), 16).mean(axis=1)
    margin_samples = bundle['endpoints'].predict(x).reshape(len(selected), 16, 2)
    for j, row in enumerate(transformed):
        for k, (f, reverse) in enumerate(row):
            if reverse:
                margin_samples[j, k] = margin_samples[j, k][::-1].copy()
    margins = margin_samples.mean(axis=1)
    side = bundle['side_split'].predict_proba(x)
    side = side[:, list(bundle['side_split'].classes_).index(1)].reshape(len(selected), 16).mean(axis=1)
    records, predictions = [], []
    for j, (i, seq) in enumerate(selected):
        length = seq['sample_step_mm'] * (len(seq['xyz']) - 1)
        start, end = float(margins[j, 0] * length), float((1. - margins[j, 1]) * length)
        count = 3 if side[j] >= .5 else 1
        records.append({'sequence': i, 'foreground_probability': float(foreground[j]),
                        'side_split_probability': float(side[j]), 'predicted_segment_count': count,
                        'predicted_start_margin_mm': start, 'predicted_end_margin_mm': float(length - end)})
        if foreground[j] < bundle['foreground_threshold'] or not 0 <= start < end <= length or end - start <= 3.:
            continue
        cuts = start + np.asarray(bundle['split_fractions'][str(count)]) * (end - start)
        origin = np.asarray(seq['xyz'][0], dtype=float)
        direction = np.asarray(seq['direction'], dtype=float)
        segments = [{'kind': 'line', 'points': [(origin + a * direction).tolist(), (origin + b * direction).tolist()]}
                    for a, b in zip(cuts[:-1], cuts[1:])]
        predictions.append({'id': f'vertical_{len(predictions) + 1:04d}', 'type': 'vertical',
                            'segments': segments, 'confidence': float(foreground[j])})
    return predictions, records
