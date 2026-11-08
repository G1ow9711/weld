"""Soft source context for the actual native event interval/cut generator.

The local cell head discarded the existing network's global joint context.
Six cheap observations reuse its already required forward; they never impose
an old stopping gate, a name rule, a coordinate table or a minimum seam length.
"""
import numpy as np
from . import curveonly_01a1ac7c_event_cut_model_v308 as base

previous = base.previous
node_features, decode = base.node_features, base.decode
NODE_NAMES, HYPERPARAMETERS, CUT_PARAMETERS = base.NODE_NAMES, base.HYPERPARAMETERS, base.CUT_PARAMETERS
EXTRA_NAMES = ('prior_forward_initial_continuation_uncalibrated',
    'prior_reverse_initial_continuation_uncalibrated',
    'prior_max_executed_continuation_uncalibrated',
    'prior_max_full_cycle_score_uncalibrated', 'prior_has_executed_interval',
    'native_total_log_length_mm')
NAMES = base.NAMES+EXTRA_NAMES


def source_context(data, states):
    assert len(states) == len(data['offsets'])-1
    extra = np.zeros((len(data['cells']), len(EXTRA_NAMES)), np.float32)
    for ci, state in enumerate(states):
        assert state['candidate'] == ci
        forward, reverse = state['forward_steps'], state['reverse_steps']
        steps = forward+reverse
        lo, hi = data['offsets'][ci:ci+2]
        if hi == lo: continue
        values = [forward[0]['continue_probability'] if forward else 0.,
            reverse[0]['continue_probability'] if reverse else 0.,
            max([q['continue_probability'] for q in steps if q['continue_selected']], default=0.),
            max([float(q.get('full_cycle_probability') or 0.) for q in steps], default=0.),
            bool(state['codes']), np.log1p(data['cells'][lo]['length_mm'])]
        extra[lo:hi] = values
    assert np.isfinite(extra).all()
    return extra


class IntervalModel(base.IntervalModel):
    def predict(self, features, packet):
        physical = np.asarray(packet['event_interval_observations_v304'], np.float32)
        assert physical.shape[1] == len(base.NAMES) and self.backbone.last_states is not None
        extra = source_context(packet['event_interval_graph_v304'], self.backbone.last_states)
        augmented = np.column_stack([physical, extra]).astype(np.float32)
        result = super().predict(features, dict(packet, event_interval_observations_v304=augmented.tolist()))
        self.last_timing.update(cell_features=len(NAMES), reused_source_context_features=6,
            no_extra_neural_forward=True, no_hard_old_continuation_gate=True)
        self.last_diagnostics.update(source_context_names=list(EXTRA_NAMES),
            source_context_by_candidate=[extra[lo].tolist() if hi > lo else None
                for lo, hi in zip(packet['event_interval_graph_v304']['offsets'][:-1], packet['event_interval_graph_v304']['offsets'][1:])],
            source_scores_uncalibrated_not_new_weld_or_side_certificate=True)
        return result


def input_curves(record, model): return base.input_curves(record, model)
