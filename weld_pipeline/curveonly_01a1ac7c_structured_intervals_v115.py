"""Arc-length Markov decoding of frozen local probabilities, with no length veto."""
import time
import numpy as np

from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_set_v18 as base
from . import curveonly_01a1ac7c_local_intervals_v95 as pointwise


def transition(delta, prior):
    a, b = prior['enter_per_mm'], prior['exit_per_mm']
    assert delta > 0 and a > 0 and b > 0
    q = -np.expm1(-(a + b) * delta)
    return np.array([[1 - a / (a + b) * q, a / (a + b) * q],
                     [b / (a + b) * q, 1 - b / (a + b) * q]])


def potentials(station, probability, closed, prior):
    s = np.asarray(station, float)
    p = np.asarray(probability, float)
    assert p.shape == (len(s), 2) and len(s) >= 2
    delta = np.diff(s)
    assert np.all(delta > 0) and abs(s[0]) < 1e-8
    if closed:
        assert len(s) >= 3
        p = p[:-1]
        width = (delta + np.roll(delta, 1)) / 2
    else:
        width = np.r_[delta[0] / 2, (delta[:-1] + delta[1:]) / 2, delta[-1] / 2]
    unary = np.log(np.clip(p, 1e-12, 1)) * (width / prior['reference_station_mm'])[:, None]
    edge = np.log(np.stack([transition(float(d), prior) for d in delta]))
    return unary, edge


def path_score(states, unary, edge, closed, prior):
    """Explicit objective used by independent brute-force controls."""
    z = np.asarray(states, int)
    value = float(unary[np.arange(len(z)), z].sum())
    value += float(edge[np.arange(len(z) - 1), z[:-1], z[1:]].sum())
    if closed:
        value += float(edge[-1, z[-1], z[0]])
    else:
        a, b = prior['enter_per_mm'], prior['exit_per_mm']
        value += float(np.log([b / (a + b), a / (a + b)])[z[0]])
    return value


def decode_states(station, probability, closed, prior):
    unary, edge = potentials(station, probability, closed, prior)
    n = len(unary)
    best, best_value = None, -np.inf
    for first in ((0, 1) if closed else (None,)):
        if closed:
            value = np.full(2, -np.inf)
            value[first] = unary[0, first]
        else:
            a, b = prior['enter_per_mm'], prior['exit_per_mm']
            value = np.log([b / (a + b), a / (a + b)]) + unary[0]
        previous = np.zeros((n, 2), np.int8)
        for i in range(1, n):
            proposal = value[:, None] + edge[i - 1]
            previous[i] = np.argmax(proposal, axis=0)
            value = proposal[previous[i], np.arange(2)] + unary[i]
        final = value + edge[-1, :, first] if closed else value
        last = int(np.argmax(final))
        if float(final[last]) > best_value:
            state = np.zeros(n, np.int8)
            state[-1] = last
            for i in range(n - 1, 0, -1):
                state[i - 1] = previous[i, state[i]]
            best, best_value = state, float(final[last])
    assert best is not None
    return (np.r_[best, best[0]] if closed else best), best_value


def intervals(station, probabilities, closed, length, prior):
    s = np.asarray(station, float)
    p = np.asarray(probabilities, float).copy()
    if closed:
        p[-1] = p[0]  # Same native endpoint convention as the frozen pointwise decoder.
    label, objective = decode_states(s, p, closed, prior)
    cuts = np.r_[0, np.flatnonzero(label[1:] != label[:-1]) + 1, len(label)]
    runs = []

    def crossing(a, b):
        y0, y1 = p[a, 1] - p[a, 0], p[b, 1] - p[b, 0]
        if y0 * y1 <= 0 and abs(y1 - y0) > 1e-12:
            return float(s[a] + (s[b] - s[a]) * np.clip(-y0 / (y1 - y0), 0, 1))
        return float((s[a] + s[b]) / 2)

    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if label[lo] == 0:
            continue
        a = 0. if lo == 0 else crossing(lo - 1, lo)
        b = length if hi == len(s) else crossing(hi - 1, hi)
        runs.append(dict(start_mm=a, end_mm=b, kind=1, confidence=float(p[lo:hi, 1].mean())))
    if closed and len(runs) > 1 and runs[0]['start_mm'] == 0. and runs[-1]['end_mm'] == length:
        first, last = runs[0], runs[-1]
        a, b = last['end_mm'] - last['start_mm'], first['end_mm']
        combined = dict(start_mm=last['start_mm'], end_mm=length + first['end_mm'], kind=1,
                        confidence=(a * last['confidence'] + b * first['confidence']) / max(a + b, 1e-12))
        runs = [*runs[1:-1], combined]
    return [q for q in runs if q['end_mm'] > q['start_mm'] + 1e-6], label, objective


class StructuredCurveSet:
    def __init__(self, classifier, prior):
        assert list(classifier.classes_) == [0, 1]
        self.classifier, self.prior = classifier, dict(prior)
        self.last_states = self.last_timing = None

    def predict(self, features, packet):
        tick = time.perf_counter()
        x = pointwise.encode(features, packet)
        prob = self.classifier.predict_proba(x) if len(x) else np.empty((0, 2))
        proposals, states = [], []
        for ci, candidate in enumerate(packet['candidates']):
            lo, hi = packet['offsets'][ci:ci + 2]
            points = np.asarray(candidate['points'])
            length = float(cad.arclength(points)[-1])
            closed = np.linalg.norm(points[0] - points[-1]) <= 1e-6
            runs, labels, objective = intervals(candidate['station_s_mm'], prob[lo:hi], closed, length, self.prior)
            raw = pointwise.intervals(candidate['station_s_mm'], prob[lo:hi], self.classifier.classes_, closed, length)
            states.append(dict(candidate=ci, intervals=runs, pointwise_intervals=raw,
                               changed_station_count=int((labels != prob[lo:hi].argmax(1)).sum()), objective=objective))
            for j, q in enumerate(runs):
                proposals.append(dict(candidate=ci, points=base.native_slice(points, q['start_mm'], q['end_mm']).tolist(),
                                      weld_type='horizontal', confidence=q['confidence'],
                                      interval=dict(start_mm=q['start_mm'], end_mm=q['end_mm'], index=j, predicted_count=len(runs))))
        self.last_states = states
        self.last_timing = dict(cpu_seconds=time.perf_counter() - tick, local_features=pointwise.FEATURES,
                                continuous_decode=True, prior=self.prior, no_minimum_length_or_gap_fill=True,
                                input_curves=len(packet['candidates']), input_stations=len(features))
        return proposals
