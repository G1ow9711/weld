"""Learned point-derived candidate decisions; no STEP/reference loader."""
import numpy as np

from weld_pipeline.candidate_frame_tree_model import frames
from weld_pipeline.point_plate_context import contact_descriptor, endpoint_descriptor

VERSION = 'point_plate_candidate_v1'


def predict_vertical(bundle, sequences, structural, process):
    selected = [(i, s) for i, s in enumerate(sequences) if s['type'] == 'vertical']
    if not selected:
        return [], []
    xf, xe, xs = [], [], []
    for i, seq in selected:
        for transformed, reverse in frames(seq):
            xf.append(np.r_[structural[i], contact_descriptor(transformed)])
            xe.append(endpoint_descriptor(transformed))
        length = seq['sample_step_mm'] * (len(seq['xyz'])-1)
        xs.append(np.r_[structural[i], process[i], length/1000., 1.])
    foreground = bundle['foreground'].predict_proba(np.stack(xf))
    foreground = foreground[:, list(bundle['foreground'].classes_).index(1)].reshape(-1, 16).mean(1)
    margins = bundle['endpoints'].predict(np.stack(xe)).reshape(-1, 16, 2)
    for j, (i, seq) in enumerate(selected):
        for k, (transformed, reverse) in enumerate(frames(seq)):
            if reverse:
                margins[j, k] = margins[j, k][::-1].copy()
    margins = margins.mean(1)
    side = bundle['side_split'].predict_proba(np.stack(xs))[:, list(bundle['side_split'].classes_).index(1)]
    predictions, diagnostic = [], []
    for j, (i, seq) in enumerate(selected):
        length = seq['sample_step_mm'] * (len(seq['xyz'])-1)
        start, end = float(margins[j, 0]), float(length-margins[j, 1])
        count = 3 if side[j] >= .5 else 1
        diagnostic.append({'sequence': i, 'foreground_probability': float(foreground[j]),
                           'side_split_probability': float(side[j]), 'segment_count': count,
                           'margins_mm': margins[j].tolist()})
        if foreground[j] < .5 or not 0 <= start < end <= length or end-start <= 3.:
            continue
        cuts = start+np.array(bundle['split_fractions'][str(count)])*(end-start)
        origin, direction = np.asarray(seq['xyz'][0], float), np.asarray(seq['direction'], float)
        predictions.append({'id': f'plate_v_{len(predictions)+1:04d}', 'type': 'vertical',
                            'confidence': float(foreground[j]),
                            'segments': [{'kind': 'line', 'points': [(origin+a*direction).tolist(),
                                         (origin+b*direction).tolist()]} for a,b in zip(cuts[:-1], cuts[1:])]})
    return predictions, diagnostic
