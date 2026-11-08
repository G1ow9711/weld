"""Continuous native weld-union range targets, in millimetres."""
import time
import numpy as np
from . import curveonly_01a1ac7c_section_local_v135 as section
from . import curveonly_01a1ac7c_native_reference_v154 as reference
from . import curveonly_01a1ac7c_local_intervals_v95 as pointwise
from . import curveonly_01a1ac7c_structured_intervals_v115 as sequence
from . import curveonly_01a1ac7c_set_v18 as base

CLIP_MM = 3.0  # Frozen geometric tolerance, not a tuned decision threshold.


def merge(intervals):
    result = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals):
        assert np.isfinite([a, b]).all() and b > a
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def signed_field(station, intervals, length, closed):
    """Positive inside the exact interval union, negative outside, zero at bounds.

    Replicate periodic intervals before merging, so the coordinate origin is
    not a boundary. Every strictly positive gap remains in the target.
    """
    s = np.asarray(station, float)
    assert s.ndim == 1 and np.isfinite(s).all() and length > 0
    assert np.all((s >= -1e-9) & (s <= length + 1e-9))
    windows = []
    for a, b in intervals:
        a, b = float(a), float(b)
        assert b > a and np.isfinite([a, b]).all()
        if closed:
            if b - a >= length:
                return np.full(len(s), CLIP_MM, np.float32)
            width = b - a
            a %= length
            windows.extend((a + turn * length, a + width + turn * length) for turn in (-1, 0, 1))
        else:
            windows.append((a, b))
    windows = merge(windows)
    if closed and any(b - a >= length for a, b in windows):
        return np.full(len(s), CLIP_MM, np.float32)
    result = np.full(len(s), -CLIP_MM)
    for a, b in windows:
        result = np.maximum(result, np.minimum(s - a, b - s))
    return np.clip(result, -CLIP_MM, CLIP_MM).astype(np.float32)


def case_targets(packet, chosen):
    values = np.full(packet['offsets'][-1], -CLIP_MM, np.float32)
    by = {}
    for target in chosen:
        by.setdefault(target['candidate'], []).append((target['start_mm'], target['end_mm']))
    for ci, windows in by.items():
        candidate = packet['candidates'][ci]
        points = np.asarray(candidate['points'])
        s, _ = reference.parameters(candidate)
        closed = bool(np.linalg.norm(points[0] - points[-1]) <= 1e-6)
        lo, hi = packet['offsets'][ci:ci + 2]
        values[lo:hi] = signed_field(s, windows, float(s[-1]), closed)
    return values


class DistanceUnary:
    """Expose an uncalibrated monotone distance score to existing decoders."""
    def __init__(self, regressor):
        self.regressor = regressor
        self.classes_ = np.array([0, 1], np.int64)

    def predict_proba(self, x):
        distance = np.asarray(self.regressor.predict(x), float)
        assert distance.shape == (len(x),) and np.isfinite(distance).all()
        positive = np.clip(.5 + distance / (2 * CLIP_MM), 0, 1)
        return np.column_stack([1 - positive, positive])


class NativeFieldCurveSet:
    """Keep frozen feature windows; decode ranges in verified native coordinates."""
    def __init__(self, classifier, prior=None):
        self.classifier=classifier
        self.prior=dict(prior)if prior is not None else None
        self.last_timing=self.last_states=None

    def predict(self, features, packet):
        start=time.perf_counter();encoded=section.encode(features,packet)
        probability=self.classifier.predict_proba(encoded)if len(encoded)else np.empty((0,2))
        proposals=[];states=[]
        for ci,candidate in enumerate(packet['candidates']):
            lo,hi=packet['offsets'][ci:ci+2];points=np.asarray(candidate['points'])
            station,receipt=reference.parameters(candidate);length=float(station[-1])
            closed=bool(np.linalg.norm(points[0]-points[-1])<=1e-6)
            if self.prior is None:
                runs=pointwise.intervals(station,probability[lo:hi],self.classifier.classes_,closed,length)
                detail={}
            else:
                runs,labels,objective=sequence.intervals(station,probability[lo:hi],closed,length,self.prior)
                detail=dict(changed_station_count=int((labels!=probability[lo:hi].argmax(1)).sum()),objective=objective)
            states.append(dict(candidate=ci,intervals=runs,native_reference=receipt,**detail))
            for j,q in enumerate(runs):
                proposals.append(dict(candidate=ci,points=base.native_slice(points,q['start_mm'],q['end_mm']).tolist(),
                    weld_type='horizontal',confidence=q['confidence'],
                    interval=dict(start_mm=q['start_mm'],end_mm=q['end_mm'],index=j,predicted_count=len(runs))))
        self.last_states=states
        self.last_timing=dict(cpu_seconds=time.perf_counter()-start,local_features=section.FEATURES,
            native_parameter_reference='INPUT_ONLY_EXACT_STATION_POLYLINE_PROJECTION_V154',
            old_feature_windows_unchanged=True,continuous_decode=self.prior is not None,prior=self.prior,
            no_minimum_length_or_gap_fill=True,input_curves=len(packet['candidates']),input_stations=len(features))
        return proposals


class ZeroCurveSet(NativeFieldCurveSet):
    def __init__(self,classifier):super().__init__(classifier,None)
