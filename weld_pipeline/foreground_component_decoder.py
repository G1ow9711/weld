"""Development interval diagnostic: fixed foreground threshold, optional unchanged cut peaks."""
import numpy as np

from weld_pipeline.event_gap_decoder import event_peaks

VERSION = 'foreground_component_diagnostic_v1'


def crossing(values, left, right, threshold=.5):
    a, b = float(values[left]), float(values[right])
    if abs(b-a) < 1e-12:
        return (left+right)/2.
    return float(np.clip(left+(threshold-a)/(b-a), left, right))


def decode_foreground(sequences, probabilities, use_cuts=False):
    result = []
    for sequence, p in zip(sequences, probabilities):
        if sequence['type'] != 'horizontal' or sequence.get('curve_kind') == 'arc':
            continue
        values = np.asarray(p[:, 0], float)
        if not np.isfinite(p).all():
            raise ValueError('Finite foreground/cut predictions required')
        changes = np.diff(np.r_[False, values>.5, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1); stops = np.flatnonzero(changes == -1)
        cuts = [t for t, _ in event_peaks(p[:, 3], sequence['sample_step_mm'], .5)] if use_cuts else []
        for i, j in zip(starts, stops):
            a = crossing(values, i-1, i) if i > 0 else 0.
            b = crossing(values, j-1, j) if j < len(values) else float(len(values)-1)
            if b <= a:
                raise ValueError('Foreground interval has no physical length')
            # No minimum-length filter and no gap closing: short predictions remain scoreable.
            divisions = [a]+sorted(set(c for c in cuts if a+1e-7 < c < b-1e-7))+[b]
            origin = sequence['xyz'][0].astype(float)
            direction = sequence['direction']*sequence['sample_step_mm']
            segments = [dict(kind='line', points=[(origin+x*direction).tolist(), (origin+y*direction).tolist()])
                        for x, y in zip(divisions[:-1], divisions[1:])]
            result.append(dict(id=f'fg_{len(result)+1:05d}', type='horizontal', segments=segments,
                               confidence=float(values[i:j].mean())))
    return result
