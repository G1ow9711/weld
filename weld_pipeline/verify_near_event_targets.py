"""Full original 16-source target representation proof before near-vote training."""
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals, event_peaks
from weld_pipeline.near_event_votes import dense_targets, vote_heatmaps, RADIUS_MM, SIGMA_MM, TAIL_MM, VOTE_SCALE

OUT = Path('weld_pipeline/reports/near_event_target_proof_156_20261001')


def physical_cases():
    cases = []; step = 2.; locations = np.arange(121)*step
    events_by_case = [('empty', []), ('subsample', [37.3]), ('near_boundary', [2.7, 235.1]),
                      ('near_pair', [67.3, 73.568]), ('multiple', [33.2, 88.7, 165.1, 217.8])]
    for label, events in events_by_case:
        target = np.zeros((len(locations), 4), np.float32)
        for event in events:
            target[:, 1] = np.maximum(target[:, 1], np.exp(-.5*((locations-event)/1.5)**2))
        dense = dense_targets(target, step)
        result = vote_heatmaps(dense[:, 0], dense[:, 1:4], dense[:, 4:7], step)
        got = [p*step for p, _ in event_peaks(result[:, 1], step, .5)]
        error = max((min(abs(e-g) for g in got) if got else float('inf') for e in events), default=0.)
        reversed_target = target[::-1].copy(); reversed_target[:, [1, 2]] = reversed_target[:, [2, 1]]
        rev = dense_targets(reversed_target, step)
        rp = vote_heatmaps(rev[:, 0], rev[:, 1:4], rev[:, 4:7], step)
        expected = result[::-1].copy(); expected[:, [1, 2]] = expected[:, [2, 1]]
        reverse_error = float(np.abs(rp-expected).max())
        passed = len(got) == len(events) and error <= 3. and reverse_error <= 2e-6
        cases.append(dict(case=label, events_mm=events, predicted_events_mm=got,
                          max_error_mm=error, reverse_max_error=reverse_error, passed=passed))
    # Distinct start/end roles and short adjacent child tails stay distinct.
    for label, bounds, cuts in [('short_6_268', (53.3, 59.568), []),
                                ('adjacent_short_tails', (45.3, 86.2), [51.568, 80.2])]:
        target = np.zeros((len(locations), 4), np.float32)
        target[:, 0] = ((locations >= bounds[0]) & (locations <= bounds[1]))
        for channel, events in [(1, [bounds[0]]), (2, [bounds[1]]), (3, cuts)]:
            for event in events:
                target[:, channel] = np.maximum(target[:, channel], np.exp(-.5*((locations-event)/1.5)**2))
        dense = dense_targets(target, step)
        p = vote_heatmaps(dense[:, 0], dense[:, 1:4], dense[:, 4:7], step)
        seq = dict(type='horizontal', xyz=np.c_[locations, np.zeros((len(locations), 2))],
                   direction=np.array([1., 0., 0.]), sample_step_mm=step)
        predictions = decode_intervals([seq], [p], threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
        points = [bounds[0], *cuts, bounds[1]]
        truth = [dict(type='horizontal', segments=[dict(kind='line', points=[[a, 0., 0.], [b, 0., 0.]])
                  for a, b in zip(points[:-1], points[1:])])]
        score = score_children(predictions, truth)
        passed = score['valid_children'] == len(points)-1 and score['false_predictions'] == 0
        cases.append(dict(case=label, bounds_mm=bounds, cuts_mm=cuts, score=score, passed=passed))
    return cases


def main():
    if OUT.exists():
        raise ValueError('Preserve near-event target proof')
    OUT.mkdir(parents=True)
    cases = physical_cases()
    (OUT/'physical.json').write_text(json.dumps(cases, indent=2), encoding='utf8')
    if not all(c['passed'] for c in cases):
        (OUT/'failed.json').write_text(json.dumps(dict(stage='physical', cases=cases,
            training_allowed=False, full_contract_pass=False), indent=2), encoding='utf8')
        raise ValueError('Near-vote physical proof failed; training disallowed')
    cfg_path = Path('weld_pipeline/data/joint_relation_v1/config.json')
    cfg = json.loads(cfg_path.read_text('utf8')); names = cfg['folds'][0]['train_real']
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8')); scores = {}; sources = {}
    for name in names:
        root = Path(split['piece_roots'][name]); piece = load_piece(root, verify=True)
        sources[name] = dict(cache_manifest_sha256=sha256(root/'manifest.json'),
                             train_manifest_sha256=sha256(Path('weld_pipeline/data/joint_relation_v1')/name/'manifest.json'))
        probabilities = []
        for sequence, target in zip(piece['sequences'], piece['targets']):
            p = np.zeros((len(target), 7), np.float32)
            if sequence['type'] == 'horizontal' and sequence.get('curve_kind') != 'arc':
                _, labels, transform = canonical_frame(sequence, target)
                dense = dense_targets(labels, sequence['sample_step_mm'])
                p = restore_probability_order(vote_heatmaps(dense[:, 0], dense[:, 1:4],
                    dense[:, 4:7], sequence['sample_step_mm']), transform)
            probabilities.append(p)
        predictions = decode_intervals(piece['sequences'], probabilities,
            threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
        truth = [w for w in piece['truth'] if w['type'] == 'horizontal'
                 and all(s['kind'] == 'line' for s in w['segments'])]
        score = score_children(predictions, truth); scores[name] = score
        (OUT/(name+'.json')).write_text(json.dumps(dict(score=score, predictions=predictions),
            indent=2), encoding='utf8')
        print('NEAR_EVENT_TARGET', name, score['valid_children'], '/', score['truth_count'],
              'FP', score['false_predictions'], flush=True)
    result = aggregate(scores)
    passed = result['truth_count'] == 1221 and result['valid_children'] == 1221 and result['false_predictions'] == 0
    complete = dict(physical_cases=cases, train_sources=names, original_sources=sources,
        target_representation=result, all_target_checks_passed=passed, training_allowed=passed,
        parameters=dict(radius_mm=RADIUS_MM, sigma_mm=SIGMA_MM, tail_mm=TAIL_MM, vote_scale=VOTE_SCALE),
        source_sha256={p:sha256(p) for p in [__file__, 'weld_pipeline/near_event_votes.py',
                        'weld_pipeline/event_gap_decoder.py', 'weld_pipeline/contract_child_evaluation.py']},
        data_config_sha256=sha256(cfg_path), full_fit_split_sha256=sha256(split_path),
        held_sources_read=False, observed_check_read=False, no_gradient_training=True,
        proof_scope='Original 16 FIT H-line children and specified physical cases only',
        minimum_contract_length_not_frozen=True, full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete, indent=2, ensure_ascii=False), encoding='utf8')
    print('NEAR_EVENT_TARGET_PROOF_COMPLETE', passed, result['valid_children'], result['false_predictions'], flush=True)
    if not passed:
        raise ValueError('Near-vote full-denominator representation failed; training disallowed')


if __name__ == '__main__':
    main()
