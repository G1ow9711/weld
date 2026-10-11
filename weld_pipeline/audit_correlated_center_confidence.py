"""Fixed raw heads: isolate confidence pooling without changing event geometry."""
import gc
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline import audit_bounded_phase_centers_v2 as bounded_proof
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.continuous_center_modes import center_modes as original_modes
from weld_pipeline.continuous_center_modes_v2 import center_modes as mean_modes, restore_event_order
from weld_pipeline.continuous_event_decoder import decode_event_sets
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.train_forest_event_centers import load_cached

RAW = Path('weld_pipeline/reports/deep_physical_head_diagnosis_173_20261002')
RUN = Path('weld_pipeline/runs/deep_physical_events_170_v3/relative_context')
OUT = Path('weld_pipeline/reports/correlated_center_confidence_176_20261002')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def sigmoid(values):
    return 1./(1.+np.exp(-np.clip(np.asarray(values, float), -60., 60.)))


def main():
    if OUT.exists():
        raise ValueError('Preserve confidence pooling intervention')
    config = json.loads(Path('weld_pipeline/runs/deep_physical_events_170_v3/config.json').read_text('utf8'))
    old = json.loads((RAW/'complete.json').read_text('utf8'))
    if not old['full_16_frozen_learned_outputs_exact']:
        raise ValueError('Full original raw head consistency required')
    OUT.mkdir(parents=True)
    source_paths = [__file__, 'weld_pipeline/continuous_center_modes_v2.py',
        'weld_pipeline/continuous_center_modes.py', 'weld_pipeline/continuous_event_decoder.py',
        'weld_pipeline/audit_bounded_phase_centers_v2.py', 'weld_pipeline/bounded_event_phase_v2.py',
        'weld_pipeline/bounded_event_phase.py', 'weld_pipeline/contract_child_evaluation.py']
    dump(OUT/'protocol.json', dict(sources=config['train'], input_checkpoint_sha256=sha256(RUN/'final.pt'),
        raw_diagnosis_complete_sha256=sha256(RAW/'complete.json'),
        same_role_position_groups_unchanged=True, confidence='mean predicted support per correlated query group',
        no_minimum_segment_length_or_opposite_role_merge=True,
        source_sha256={p: sha256(p) for p in source_paths}, actual_trained_models=0,
        full_contract_pass=False))
    analytic_rows = json.loads(Path('weld_pipeline/reports/bounded_phase_centers_175v2_20261002/analytic.json').read_text('utf8'))['rows']
    bounded_proof.center_modes = mean_modes
    analytic = [bounded_proof.analytic(r['intervals_mm'], r['sample_step_mm'], r['cuts_mm']) for r in analytic_rows]
    dump(OUT/'analytic.json', analytic)
    if not all(r['topology_and_coordinates_retained'] for r in analytic):
        raise ValueError('Mean confidence lost a short/gap/cut target')
    variants = {v: {} for v in ('learned_original_or', 'learned_correlated_mean', 'teacher_correlated_mean')}
    receipts = {}
    for name in config['train']:
        data, near, known, piece = load_cached(name)
        with np.load(RAW/(name+'_raw_heads.npz'), allow_pickle=False) as archive:
            raw, lengths = archive['raw'], archive['lengths']
        if not np.array_equal(lengths, [len(s['xyz']) for s in piece['sequences']]):
            raise ValueError('Canonical raw rows changed')
        prefixes = np.r_[0, np.cumsum(lengths)]
        foregrounds = {v: [np.zeros(n, np.float32) for n in lengths] for v in variants}
        events = {v: [[[], [], []] for n in lengths] for v in variants}
        for entry in data['entries']:
            if entry['type'] != 'horizontal':
                continue
            index = entry['sequence']
            seq, _, tf = canonical_frame(piece['sequences'][index])
            rr = raw[prefixes[index]:prefixes[index+1]]
            yy = np.asarray(near[entry['start']:entry['stop']], float)
            kk = np.asarray(known[entry['start']:entry['stop']])
            step = float(seq['sample_step_mm']); span = (len(rr)-1)*step
            original = original_modes(sigmoid(rr[:, 1:4]), rr[:, 4:7], step)
            averaged = mean_modes(sigmoid(rr[:, 1:4]), rr[:, 4:7], step)
            if any([p for p, c in a] != [p for p, c in b] for a, b in zip(original, averaged)):
                raise ValueError('Confidence intervention changed mode coordinates')
            teacher = mean_modes(np.where(kk[:, 1:4]>0, yy[:, 1:4], 0.), yy[:, 4:7], step)
            for variant, fg, hypotheses in [('learned_original_or', sigmoid(rr[:, 0]), original),
                    ('learned_correlated_mean', sigmoid(rr[:, 0]), averaged),
                    ('teacher_correlated_mean', np.clip(yy[:, 0], 0., 1.), teacher)]:
                foregrounds[variant][index], events[variant][index] = restore_event_order(fg, hypotheses, span, tf)
        expected = json.loads((RUN/'FIT'/(name+'.json')).read_text('utf8'))
        if events['learned_original_or'] != expected['event_sets']:
            raise ValueError('Original frozen mode baseline differs')
        truth = [w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        results = {}
        for variant in variants:
            predictions = decode_event_sets(piece['sequences'], foregrounds[variant], events[variant])
            if variant == 'learned_original_or' and predictions != expected['predictions']:
                raise ValueError('Original frozen child baseline differs')
            score = score_children(predictions, truth); variants[variant][name] = score
            results[variant] = dict(score=score, predictions=predictions)
            print('CORRELATED_CONFIDENCE', name, variant, score['valid_children'], '/', score['truth_count'],
                'FP', score['false_predictions'], flush=True)
        dump(OUT/(name+'.json'), dict(variants=results, baseline_exact=True,
            raw_heads_sha256=sha256(RAW/(name+'_raw_heads.npz')), full_contract_pass=False))
        receipts[name] = sha256(OUT/(name+'.json'))
        del data, near, known, piece, raw, foregrounds, events, expected; gc.collect()
    totals = {v: aggregate(scores) for v, scores in variants.items()}
    teacher = totals['teacher_correlated_mean']
    passed = teacher['truth_count']==1221 and teacher['valid_children']==1221 and teacher['false_predictions']==0
    dump(OUT/('complete.json' if passed else 'failure.json'), dict(variants=totals,
        all_16_original_baselines_exact=True, all_1221_teacher_and_analytic_targets_represented=passed,
        analytic_cases=len(analytic), piece_receipts=receipts, actual_trained_models=0,
        held_or_observed_read=False, default_predictor_replaced=False, full_contract_pass=False))
    if not passed:
        raise ValueError('Full mean confidence representation gate failed')


if __name__ == '__main__':
    main()
