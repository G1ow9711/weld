"""Pure-point candidate descriptors and a trained vertical interval predictor.

This module has no reference loader. Identifiers, absolute XYZ, workpiece
names, and workpiece bounding extents are not model inputs.
"""
import numpy as np

FEATURE_VERSION = 'local_candidate_profile_v1'
LOCAL_CHANNELS = np.r_[0:49, 70:84]


def descriptor(sequence):
    feature = np.asarray(sequence['features'], dtype=np.float64)
    local = feature[:, LOCAL_CHANNELS]
    source = np.linspace(0., 1., len(local))
    positions = np.linspace(0., 1., 17)
    profile = np.column_stack([np.interp(positions, source, local[:, k]) for k in range(local.shape[1])])
    quantiles = np.quantile(local, [.05, .25, .5, .75, .95], axis=0)
    difference = np.abs(np.diff(local, axis=0))
    stats = np.r_[local.mean(axis=0), local.std(axis=0), difference.mean(axis=0), difference.max(axis=0)]
    length = float(sequence['sample_step_mm'] * (len(local) - 1))
    # Signed contact-plane axes orient the local distance grid. No origin or
    # workpiece-size lookup is admitted.
    axes = feature[len(feature) // 2, 57:66]
    return np.r_[length / 1000., axes, profile.ravel(), quantiles.ravel(), stats].astype(np.float32)


def predict_vertical(bundle, sequences):
    selected = [(i, seq) for i, seq in enumerate(sequences) if seq['type'] == 'vertical']
    if not selected:
        return [], []
    x = np.stack([descriptor(s) for i, s in selected])
    probs = bundle['foreground'].predict_proba(x)
    classes = list(bundle['foreground'].classes_)
    foreground = probs[:, classes.index(1)]
    margins = bundle['endpoints'].predict(x)
    counts = bundle['segments'].predict(x)
    records, predictions = [], []
    for j, (i, seq) in enumerate(selected):
        length = seq['sample_step_mm'] * (len(seq['xyz']) - 1)
        start, end = float(margins[j, 0]), float(length - margins[j, 1])
        count = int(counts[j])
        records.append({'sequence': i, 'foreground_probability': float(foreground[j]),
                        'predicted_start_margin_mm': start, 'predicted_end_margin_mm': float(margins[j, 1]),
                        'predicted_segment_count': count})
        if foreground[j] < bundle['foreground_threshold'] or not 0 <= start < end <= length or end - start <= 3.:
            continue
        fractions = bundle['split_fractions'][str(count)]
        cuts = start + np.asarray(fractions) * (end - start)
        origin = np.asarray(seq['xyz'][0], dtype=float)
        direction = np.asarray(seq['direction'], dtype=float)
        segments = [{'kind': 'line', 'points': [(origin + a * direction).tolist(), (origin + b * direction).tolist()]}
                    for a, b in zip(cuts[:-1], cuts[1:])]
        predictions.append({'id': f'vertical_{len(predictions) + 1:04d}', 'type': 'vertical',
                            'segments': segments, 'confidence': float(foreground[j])})
    return predictions, records
