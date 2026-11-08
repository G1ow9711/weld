"""Score complete generated intervals without changing the frozen span generator.

Only input profiles, finite provenance and actual generator decisions are read.
The learned score is complete-proposal validity, not physical weldability.
"""
import copy
import math
import time
import numpy as np
from . import curveonly_01a1ac7c_ordered_spans_v235 as generator

MODES = ('quality_rank', 'quality_gate', 'quality_rank_gate')
SCALARS = ('continue_probability', 'continue_log_odds', 'full_cycle_probability',
    'full_cycle_selected', 'output_closed', 'input_closed', 'log_source_length',
    'log_interval_length', 'source_coverage', 'minimum_source_end_clearance',
    'maximum_source_end_clearance', 'source_primitive_count', 'range_primitive_count',
    'inverse_step_index', 'inverse_direction_emitted_count', 'minimum_pointer_peak',
    'maximum_pointer_peak', 'pointer_peak_log_product', 'next_continue_probability')
NAMES = list(SCALARS)+[prefix+':'+name for prefix in
    ('span_mean', 'span_std', 'span_min', 'span_max', 'endpoint_min', 'endpoint_max',
     'outside_mean') for name in generator.NAMES]


def metadata(row, state):
    """Restore one metadata record per original code, including reversed spans."""
    records, codes = [], []
    for reverse, key in ((False, 'forward_steps'), (True, 'reverse_steps')):
        steps = state[key]
        values, indices = [], []
        for i, step in enumerate(steps):
            if step.get('termination') == 'FULL_CYCLE':
                values.append(dict(a=0., b=row['length_mm'], full=True,
                    confidence=step['continue_probability']))
            elif 'predicted_start_mm' in step and step['predicted_end_mm'] > step['predicted_start_mm']:
                values.append(dict(a=step['predicted_start_mm'], b=step['predicted_end_mm'],
                    full=False, confidence=step['continue_probability']))
            else:
                continue
            indices.append(i)
        if reverse:
            values = generator.reverse_ranges(row, values)
        codes.extend(generator.intervals_to_codes(row, values))
        for i, value in zip(indices, values):
            step = steps[i]
            records.append(dict(step, full_selected=value['full'],
                emitted_count=len(values), reverse=reverse,
                next_continue_probability=steps[i+1]['continue_probability'] if i+1 < len(steps) else 0.))
    assert codes == state['codes'], 'Quality metadata must match the frozen actual codes exactly'
    return records


def samples(row, station):
    x = np.asarray(station, float)/row['grid_step_mm']
    n = len(row['grid'])
    x = x % n if row['closed'] else np.clip(x, 0, n-1)
    left = np.floor(x).astype(int)
    right = (left+1) % n if row['closed'] else np.minimum(left+1, n-1)
    fraction = (x-left)[:, None]
    return row['grid'][left]*(1-fraction)+row['grid'][right]*fraction


def features(rows, states, proposals, packet):
    length = [float(generator.arclength(c['points'])[-1]) for c in packet['candidates']]
    cache = {}
    result = []
    for proposal in proposals:
        provenance = proposal.get('finite_representation_provenance')
        ci = provenance['original_candidate'] if provenance else proposal['candidate']
        interval = provenance['original_interval'] if provenance else proposal['interval']
        row = rows[ci]
        assert row['candidate'] == ci and row['grid'].shape[1] == 111
        if ci not in cache:
            cache[ci] = metadata(row, states[ci])
        code_index = interval['index']
        meta, code = cache[ci][code_index], states[ci]['codes'][code_index]
        a, b = interval['start_mm'], interval['end_mm']
        L, step = row['length_mm'], row['grid_step_mm']
        expected = generator.codec.decode(row, code)
        assert expected == (a, b)
        if row['closed'] and code['full_cycle']:
            station = np.arange(len(row['grid']))*step
        else:
            station = np.r_[a, np.arange(math.ceil(a/step), math.floor(b/step)+1)*step, b]
            station = np.unique(station)
        observed = samples(row, station)
        ends = samples(row, [a, b])
        outside = samples(row, [a-step, b+step])
        clearances = [a/L, (L-b)/L] if not row['closed'] else [0., 0.]
        primitive = generator.previous.provenance.primitive_intervals
        # Primitive numbers never enter the learned features.
        source_parts = primitive(ci, 0., L, packet, length)
        range_parts = primitive(ci, a, b, packet, length)
        p = float(proposal['confidence'])
        clipped = min(1-1e-12, max(1e-12, p))
        peaks = [meta.get('start_peak_probability', 1.), meta.get('end_peak_probability', 1.)]
        points = np.asarray(proposal['points'])
        scalars = [p, math.log(clipped/(1-clipped)), meta['full_cycle_probability'],
            float(code['full_cycle']), float(np.linalg.norm(points[0]-points[-1]) < .1),
            float(row['closed']), math.log1p(L), math.log1p(b-a), (b-a)/L,
            min(clearances), max(clearances), len({q[0] for q in source_parts}),
            len({q[0] for q in range_parts}), 1/(1+meta['step']), 1/meta['emitted_count'],
            min(peaks), max(peaks), sum(math.log(max(1e-12, q)) for q in peaks),
            meta['next_continue_probability']]
        vector = np.concatenate([scalars, observed.mean(0), observed.std(0),
            observed.min(0), observed.max(0), ends.min(0), ends.max(0), outside.mean(0)])
        assert vector.shape == (len(NAMES),) and np.isfinite(vector).all()
        result.append(vector)
    return np.asarray(result, np.float64).reshape(len(proposals), len(NAMES))


def apply(proposals, probability, mode):
    assert mode in MODES and len(proposals) == len(probability)
    result = []
    records = []
    for i, (original, p) in enumerate(zip(proposals, probability)):
        p = float(p)
        assert 0 <= p <= 1
        keep = mode == 'quality_rank' or p > .5
        records.append(dict(raw_proposal=i, probability=p, passed=keep,
            generator_confidence=original['confidence']))
        if not keep:
            continue
        item = dict(original, generator_confidence=original['confidence'],
            complete_proposal_quality=p, quality_mode=mode)
        if mode != 'quality_gate':
            item['confidence'] = .5+.5*max(1e-12, p) if mode == 'quality_rank' else p
        result.append(item)
    return result, records


class StaticProposals:
    def __init__(self, proposals):
        self.proposals = proposals
        self.last_timing = dict(cached_exact_generator_proposal_replay=True)

    def predict(self, features, packet):
        return copy.deepcopy(self.proposals)


class ProposalQuality:
    def __init__(self, base, classifier, mode):
        assert mode in MODES
        self.base, self.classifier, self.mode = base, classifier, mode
        self.last_states = self.last_fields = self.last_full = self.last_timing = None
        self.last_quality = self.last_unfiltered_proposals = None

    def predict(self, observed, packet):
        start = time.perf_counter()
        raw = self.base.predict(observed, packet)
        rows = generator.candidate_rows(observed, packet)
        x = features(rows, self.base.last_states, raw, packet)
        probability = self.classifier.predict_proba(x)[:, 1] if len(x) else np.array([])
        result, self.last_quality = apply(raw, probability, self.mode)
        self.last_unfiltered_proposals = raw
        for key in ('last_states', 'last_fields', 'last_full', 'last_representation_normalization'):
            setattr(self, key, getattr(self.base, key))
        self.last_timing = dict(self.base.last_timing, complete_quality_mode=self.mode,
            quality_input_proposals=len(raw), quality_kept_proposals=len(result),
            quality_features=len(NAMES), frozen_generator_unchanged=True,
            cached_profile_to_proposal_seconds=time.perf_counter()-start)
        return result


def decode(observed, packet, model, threshold=.5):
    return generator.decode(observed, packet, model, threshold)


input_curves = generator.input_curves
