"""Continuous event hypotheses from vote grids; no physical peak spacing rule."""
import numpy as np
from scipy.signal import find_peaks

from weld_pipeline.continuous_event_decoder import decode_event_sets

VERSION = 'continuous_vote_grid_events_v1'


def continuous_peaks(heat, step, threshold=.5):
    heat = np.asarray(heat, float)
    if heat.ndim != 1 or len(heat) < 2 or not np.isfinite(heat).all() or step <= 0.:
        raise ValueError('Finite sampled vote channel and positive step required')
    indices, _ = find_peaks(heat, height=threshold, distance=1, prominence=.05)
    result = []
    for i in indices:
        a, b, c = np.log(np.clip(heat[i-1:i+2], 1e-7, 1.))
        denominator = a-2.*b+c
        offset = np.clip(.5*(a-c)/denominator, -.5, .5) if abs(denominator) > 1e-8 else 0.
        amplitude = float(np.clip(np.exp(b-.25*(a-c)*offset), 1e-4, 1.-1e-4))
        result.append([float((i+offset)*step), amplitude])
    if heat[0] >= threshold and heat[0] > heat[1]:result.append([0., float(min(heat[0],1.-1e-4))])
    if heat[-1] >= threshold and heat[-1] > heat[-2]:result.append([(len(heat)-1)*step, float(min(heat[-1],1.-1e-4))])
    return sorted(result)


def decode_vote_grids(sequences, probabilities, threshold=.5, weld_penalty=2., mask_weight=.25):
    events = [[continuous_peaks(p[:, role], float(seq['sample_step_mm']), threshold) for role in (1,2,3)]
              for seq,p in zip(sequences, probabilities)]
    return decode_event_sets(sequences, [p[:,0] for p in probabilities], events, threshold, weld_penalty, mask_weight)
