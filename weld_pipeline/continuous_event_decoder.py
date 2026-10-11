"""Decode continuous typed events without deleting short welds or closing gaps.

Event positions are physical millimeters along a point-derived candidate.
This module accepts event hypotheses, never labels, names, or lookup tables.
Producing reliable event hypotheses remains the trained model's responsibility.
"""
import bisect

import numpy as np

from weld_pipeline.curved_sequences import arc_points

VERSION = 'continuous_typed_event_intervals_v1'


def validate_events(events, span, threshold):
    values = np.asarray(events, float)
    if values.size == 0:
        return np.empty((0, 2), float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError('Finite event position/confidence pairs required')
    if np.any((values[:, 1] < 0.) | (values[:, 1] > 1.)):
        raise ValueError('Event confidence must be a probability')
    values = values[(values[:, 1] >= threshold) & (values[:, 0] >= 0.) & (values[:, 0] <= span)]
    return values[np.argsort(values[:, 0], kind='stable')]


def select_event_intervals(foreground, step, events, threshold=.5, weld_penalty=2., mask_weight=.25):
    mask = np.asarray(foreground, float)
    if (mask.ndim != 1 or len(mask) < 2 or not np.isfinite(mask).all()
            or not np.isfinite(step) or step <= 0. or len(events) != 3):
        raise ValueError('Finite foreground, positive step and three typed event sets required')
    span = (len(mask)-1)*step
    starts, ends, cuts = (validate_events(e, span, threshold) for e in events)
    if not len(starts) or not len(ends):
        return [], cuts
    mask = np.clip(mask, 1e-4, 1.-1e-4)
    log_odds = np.log(mask)-np.log1p(-mask)
    prefix = np.r_[0., np.cumsum((log_odds[:-1]+log_odds[1:])*.5*step)]
    grid = np.arange(len(mask), dtype=float)*step
    logit = lambda p: float(np.log(np.clip(p, 1e-4, 1.-1e-4))-np.log1p(-np.clip(p, 1e-4, 1.-1e-4)))
    candidates = []
    for a, pa in starts:
        for b, pb in ends:
            if b <= a:
                continue
            # Sparse foreground samples are evidence, not a minimum length rule.
            # A sub-grid weld can have no sample in its interior.
            evidence = (np.interp(b, grid, prefix)-np.interp(a, grid, prefix))*mask_weight
            weight = evidence+logit(pa)+logit(pb)-weld_penalty
            if weight > 0.:
                candidates.append((float(b), float(a), float(weight), float(pa), float(pb)))
    candidates.sort(); end_positions = [c[0] for c in candidates]
    best = np.zeros(len(candidates)+1); previous = []; take = []
    for i, (b, a, weight, pa, pb) in enumerate(candidates):
        prior = bisect.bisect_right(end_positions, a, 0, i)
        selected = best[prior]+weight > best[i]
        best[i+1] = best[prior]+weight if selected else best[i]
        previous.append(prior); take.append(selected)
    result = []; i = len(candidates)
    while i:
        if take[i-1]:
            b, a, weight, pa, pb = candidates[i-1]
            result.append((a, b, pa, pb)); i = previous[i-1]
        else:
            i -= 1
    return result[::-1], cuts


def decode_event_sets(sequences, foregrounds, event_sets, threshold=.5, weld_penalty=2., mask_weight=.25):
    if len(sequences) != len(foregrounds) or len(sequences) != len(event_sets):
        raise ValueError('Each candidate needs foreground and typed continuous events')
    result = []
    for seq, foreground, events in zip(sequences, foregrounds, event_sets):
        step = float(seq['sample_step_mm'])
        intervals, internal = select_event_intervals(foreground, step, events, threshold, weld_penalty, mask_weight)
        for a, b, pa, pb in intervals:
            # Only exact duplicate coordinates are one event. No millimeter merge.
            divisions = np.unique(np.r_[a, internal[(internal[:, 0] > a) & (internal[:, 0] < b), 0], b])
            segments = []
            for u, v in zip(divisions[:-1], divisions[1:]):
                if seq.get('curve_kind') == 'arc':
                    locations = np.linspace(u, v, max(2, int(np.ceil(v-u))+1))
                    segments.append(dict(kind='arc', points=arc_points(seq, locations).tolist(), length_mm=float(v-u)))
                else:
                    q = np.asarray(seq['xyz'][0], float); direction = np.asarray(seq['direction'], float)
                    segments.append(dict(kind='line', points=[(q+u*direction).tolist(), (q+v*direction).tolist()]))
            result.append(dict(id=f'weld_{len(result)+1:04d}', type=seq['type'], segments=segments,
                               confidence=float(min(pa, pb))))
    return result
