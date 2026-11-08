"""Complete interval endpoints as input-grid indices and bounded local offsets."""
import numpy as np
from . import curveonly_01a1ac7c_interval_instances_v178 as previous

candidate_rows = previous.candidate_rows
geometry = previous.previous.geometry
reference = previous.previous.reference
input_curves = previous.input_curves
arclength = previous.arclength


def encode(row, start_mm, end_mm):
    a, b = float(start_mm), float(end_mm)
    length, step = row['length_mm'], row['grid_step_mm']
    closed, n = row['closed'], len(row['grid'])
    assert b > a and step > 0
    full = closed and abs(b-a-length) <= reference.PROJECTION_NUMERIC_MM
    if full:
        return dict(start_index=0, end_index=0, start_offset=0., end_offset=0., full_cycle=True)
    if not closed:
        assert a >= -reference.PROJECTION_NUMERIC_MM and b <= length+reference.PROJECTION_NUMERIC_MM
        a, b = max(a, 0.), min(b, length)
    values = np.array([a, b])/step
    nearest = np.floor(values+.5).astype(np.int64)
    offsets = values-nearest
    indices = nearest % n if closed else nearest
    assert np.all((indices >= 0) & (indices < n))
    assert np.abs(offsets).max() <= .5+1e-8
    return dict(start_index=int(indices[0]), end_index=int(indices[1]),
                start_offset=float(offsets[0]), end_offset=float(offsets[1]), full_cycle=False)


def decode(row, code):
    length, step = row['length_mm'], row['grid_step_mm']
    if code['full_cycle']:
        return (0., length) if row['closed'] else None
    index = np.array([code['start_index'], code['end_index']], np.int64)
    offset = np.array([code['start_offset'], code['end_offset']], float)
    assert np.all((index >= 0) & (index < len(row['grid'])))
    assert np.isfinite(offset).all() and np.abs(offset).max() <= .5+1e-7
    a, b = (index+offset)*step
    if row['closed']:
        a, b = a % length, b % length
        if b == a:
            return None
        if b < a:
            b += length
    else:
        a, b = float(np.clip(a, 0, length)), float(np.clip(b, 0, length))
    return (float(a), float(b)) if b > a else None


def proposals(packet, rows, codes):
    result = []
    for row, items in zip(rows, codes):
        ci = row['candidate']
        for i, code in enumerate(items):
            if code.get('confidence', 1.) <= .5:
                continue
            interval = decode(row, code)
            if interval is None:
                continue
            a, b = interval
            result.append(dict(candidate=ci,
                points=geometry.native_slice(np.asarray(packet['candidates'][ci]['points']), a, b).tolist(),
                weld_type='horizontal', confidence=float(code.get('confidence', 1.)),
                interval=dict(start_mm=a, end_mm=b, index=i, predicted_count=len(items))))
    return result


class CodeView:
    def __init__(self, rows, codes):
        self.rows, self.codes = rows, codes
        self.last_timing = None

    def predict(self, features, packet):
        self.last_timing = dict(paired_input_grid_endpoints_and_local_offsets=True,
                                TRAIN_teacher_not_a_learned_model=True)
        return proposals(packet, self.rows, self.codes)
