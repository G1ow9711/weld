"""Analytic contract diagnostic: short welds and real small gaps are retained.

This reads no project labels and changes no frozen training decoder. It isolates
representation and decoder limits with exact event-center teachers.
"""
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import score_children
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.near_event_votes import vote_heatmaps

OUT = Path('weld_pipeline/reports/short_event_resolution_167_20261002')


def example(intervals, step):
    locations = np.arange(-20., intervals[-1][1]+20.+step/2, step)
    xyz = np.column_stack((locations, np.zeros((len(locations), 2))))
    sequence = dict(xyz=xyz, direction=np.array([1., 0., 0.]), type='horizontal', sample_step_mm=step)
    foreground = np.zeros(len(locations)); support = np.zeros((len(locations), 3)); offsets = np.zeros_like(support)
    for a, b in intervals:
        foreground[(locations >= a) & (locations <= b)] = 1.
    for channel, values in [(0, [a for a, _ in intervals]), (1, [b for _, b in intervals])]:
        delta = np.asarray(values)[None, :]-locations[:, None]
        nearest = np.argmin(np.abs(delta), axis=1)
        offset = delta[np.arange(len(locations)), nearest]
        support[:, channel] = abs(offset) <= 16.
        offsets[:, channel] = offset/16.
    probabilities = vote_heatmaps(foreground, support, offsets, step)
    predictions = decode_intervals([sequence], [probabilities], threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
    truth = [dict(id='analytic_'+str(i), type='horizontal', segments=[dict(kind='line', points=[[a, 0., 0.], [b, 0., 0.]])]) for i, (a, b) in enumerate(intervals)]
    score = score_children(predictions, truth)
    actual = [(s['points'][0][0], s['points'][-1][0]) for p in predictions for s in p['segments']]
    actual.sort(); gaps = [actual[i+1][0]-actual[i][1] for i in range(len(actual)-1)]
    return dict(intervals_mm=intervals, sample_step_mm=step, score=score, decoded_intervals_mm=actual,
        actual_gaps_mm=gaps, expected_gaps_mm=[intervals[i+1][0]-intervals[i][1] for i in range(len(intervals)-1)],
        pure_decoder_teacher_diagnostic=True, geometry_score_does_not_prove_process_gap_fidelity=True)


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed analytic short/gap audit')
    rows = []
    for step in (.25, 2.):
        for length in (1., 2.5, 3., 3.1, 6.2680269):
            rows.append(example([(0., length)], step))
        for gap in (.5, 1., 2.):
            rows.append(example([(0., 10.), (10.+gap, 20.+gap)], step))
    OUT.mkdir(parents=True)
    receipt = dict(rows=rows, sources={p:sha256(p) for p in [__file__,
        'weld_pipeline/event_gap_decoder.py', 'weld_pipeline/near_event_votes.py', 'weld_pipeline/constrained_decoder.py']},
        actual_customer_minimum_length_not_frozen=True, original_labels_or_weights_changed=False,
        frozen_166_decoder_unchanged=True, analytic_cases_are_not_original_measured_dataset=True,
        full_contract_pass=False)
    (OUT/'summary.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    for row in rows:
        score = row['score']
        print('SHORT_GAP_RESOLUTION', row['sample_step_mm'], row['intervals_mm'],
              'TP', score['valid_children'], '/', score['truth_count'], 'GAPS', row['actual_gaps_mm'], flush=True)


if __name__ == '__main__':
    main()
