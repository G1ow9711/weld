"""Offline causal interventions on the frozen 166 auxiliary model.

Already-read development references are teachers only in this diagnostic.
Neither model weights nor runtime inference are changed.
"""
import gc
import itertools
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.forest_event_centers import HEADS, build_inputs, combine_features, centers_to_offsets, proximity_features
from weld_pipeline.near_event_votes import dense_targets, vote_heatmaps

RUN = Path('weld_pipeline/runs/forest_event_centers_seed166')
BASE = Path('weld_pipeline/data/joint_relation_v1')
OUT = Path('weld_pipeline/reports/forest_center_head_interventions_168_20261002')
FACTORS = ('foreground', 'support', 'centers')
VARIANTS = {('+'.join(name for name, flag in zip(FACTORS, flags) if flag) or 'learned'): flags
            for flags in itertools.product((False, True), repeat=3)}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def apply_teachers(raw, near, original_known, step, flags):
    foreground = np.clip(raw[:, 0], 0., 1.).copy()
    support = np.clip(raw[:, 1:4], 0., 1.).copy()
    offsets = centers_to_offsets(raw[:, 4:7], len(raw), step)
    if flags[0]:
        known = near[:, 0] >= 0
        foreground[known] = near[known, 0]
    if flags[1]:
        support[original_known] = near[:, 1:4][original_known]
    if flags[2]:
        # A negative signed offset, including -1, is a valid center target.
        known_centers = original_known & (near[:, 1:4] > .5)
        offsets[known_centers] = near[:, 4:7][known_centers]
    return foreground, support, offsets


def preflight():
    raw = np.zeros((4, 7)); raw[:, 4:7] = .75
    near = np.zeros((4, 7)); near[:, 1:4] = 1.; near[:, 4:7] = -1.
    known = np.ones((4, 3), bool); known[2, 1] = False
    near[2, 0] = -1.
    f, s, o = apply_teachers(raw, near, known, 2., (True, True, True))
    old = centers_to_offsets(raw[:, 4:7], 4, 2.)
    assert np.all(o[known] == -1.) and o[2, 1] == old[2, 1]
    assert f[2] == raw[2, 0] and s[2, 1] == raw[2, 2]
    return {'signed_negative_center_and_explicit_unknown_masks_passed': True}


def summaries(errors, binary):
    result = {}
    for channel, key in enumerate(HEADS):
        if channel < 4:
            row = binary[key]
            result[key] = {**row, 'sample_metrics_are_not_child_acceptance': True}
        else:
            values = np.concatenate(errors[key]) if errors[key] else np.empty(0)
            result[key] = dict(known_supported_samples=int(len(values)),
                center_error_mm_median=float(np.median(values)) if len(values) else None,
                center_error_mm_p95=float(np.quantile(values, .95)) if len(values) else None,
                center_error_mm_max=float(values.max()) if len(values) else None,
                center_within_3mm_fraction=float(np.mean(values <= 3.)) if len(values) else None,
                sample_metrics_are_not_child_acceptance=True)
    return result


def main():
    if OUT.exists():
        raise ValueError('Preserve the fixed diagnostic; use a new version after failure')
    checkpoint = RUN/'candidate.joblib'
    complete = json.loads((RUN/'complete.json').read_text('utf8'))
    config = json.loads((RUN/'config.json').read_text('utf8'))
    frozen = sha256(checkpoint)
    if frozen != complete['checkpoint_sha256'] or not complete['held_or_observed_read']:
        raise ValueError('Expected frozen 166 and already-read development sources')
    for path, digest in config['source_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen 166 source changed: '+path)
    OUT.mkdir(parents=True); snap = OUT/'source_snapshot'; snap.mkdir()
    shutil.copy2(__file__, snap/Path(__file__).name)
    protocol = dict(checkpoint_sha256=frozen, sources=config['held'], factors=list(FACTORS),
        variants={k:list(v) for k,v in VARIANTS.items()}, preflight=preflight(),
        expected_truth_children=222, references_are_already_read_development_data=True,
        teacher_used_only_for_offline_diagnosis=True, model_weights_changed=False,
        unknown_regions_retain_learned_values=True, source_sha256={__file__:sha256(__file__)},
        full_contract_pass=False)
    dump(OUT/'protocol.json', protocol)
    bundle = joblib.load(checkpoint); scores = {v:{} for v in VARIANTS}; receipts = {}
    all_errors = {k:[] for k in HEADS[4:]}
    all_binary = {k:dict(known_samples=0, true_positive=0, false_negative=0, false_positive=0, true_negative=0) for k in HEADS[:4]}
    began = time.perf_counter()
    for name in config['held']:
        manifest_path = BASE/name/'manifest.json'
        manifest = json.loads(manifest_path.read_text('utf8'))
        piece = load_piece(manifest['source_cache_root'], verify=True)
        input_path = Path(piece['record']['input']); input_hash = sha256(input_path)
        sequences, data = build_inputs(read_xyz_pcd(input_path)*1000.)
        if len(sequences) != len(piece['sequences']):
            raise ValueError('Raw sequence count differs from fixed development source')
        probabilities = {v:[np.zeros((len(s['xyz']), 7), np.float32) for s in sequences] for v in VARIANTS}
        raw_values = [np.zeros((len(s['xyz']), 7), float) for s in sequences]
        errors = {k:[] for k in HEADS[4:]}
        binary = {k:dict(known_samples=0, true_positive=0, false_negative=0, false_positive=0, true_negative=0) for k in HEADS[:4]}
        for entry in data['entries']:
            index = entry['sequence']
            if entry['type'] != 'horizontal' or sequences[index].get('curve_kind') == 'arc':
                continue
            seq, _, transform = canonical_frame(sequences[index])
            labeled, original, source_transform = canonical_frame(piece['sequences'][index], piece['targets'][index])
            if (transform != entry['transform'] or source_transform != transform
                    or not np.array_equal(seq['xyz'], labeled['xyz'])):
                raise ValueError('Teacher geometry differs from the fixed raw candidate')
            size = len(seq['xyz']); step = float(seq['sample_step_mm']); raw = raw_values[index]
            for a in range(0, size, 8192):
                b = min(size, a+8192); sl = slice(entry['start']+a, entry['start']+b)
                x = combine_features(np.c_[data['local'][sl], data['point_context'][sl]],
                    data['nodes'][entry['node']], proximity_features(seq, data['geometry'], data['planes'], np.arange(a,b)))
                for channel, key in enumerate(HEADS):
                    raw[a:b, channel] = bundle['heads'][key].predict(x)
            near = dense_targets(original, step); original_known = original[:, 1:4] >= 0
            offsets = centers_to_offsets(raw[:, 4:7], size, step)
            for channel, key in enumerate(HEADS):
                if channel < 4:
                    known = near[:, 0] >= 0 if channel == 0 else original_known[:, channel-1]
                    y = near[:, channel] > .5; pred = raw[:, channel] >= .5
                    row = binary[key]; row['known_samples'] += int(known.sum())
                    for metric, mask in [('true_positive', y & pred), ('false_negative', y & ~pred),
                                         ('false_positive', ~y & pred), ('true_negative', ~y & ~pred)]:
                        row[metric] += int((known & mask).sum())
                else:
                    role = channel-4
                    known = original_known[:, role] & (near[:, role+1] > .5)
                    error = np.abs(offsets[:, role]-near[:, channel])*16.
                    errors[key].append(error[known]); all_errors[key].append(error[known])
            for variant, flags in VARIANTS.items():
                foreground, support, adjusted = apply_teachers(raw, near, original_known, step, flags)
                probabilities[variant][index] = restore_probability_order(vote_heatmaps(foreground, support, adjusted, step), transform)
        baseline = np.concatenate(probabilities['learned']); lengths = np.asarray([len(p) for p in probabilities['learned']])
        with np.load(RUN/'HELD'/(name+'_heatmaps.npz'), allow_pickle=False) as archive:
            baseline_exact = np.array_equal(lengths, archive['lengths']) and np.array_equal(baseline, archive['probabilities'])
        if not baseline_exact:
            raise ValueError('Intervention baseline differs from frozen 166 HEATMAPS: '+name)
        reference = json.loads((RUN/'HELD'/(name+'.json')).read_text('utf8'))
        truth = [w for w in piece['truth'] if w['type'] == 'horizontal' and all(s['kind'] == 'line' for s in w['segments'])]
        results = {}
        for variant, p in probabilities.items():
            predictions = decode_intervals(sequences, p, threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
            score = score_children(predictions, truth); scores[variant][name] = score
            if variant == 'learned' and predictions != reference['predictions']:
                raise ValueError('Intervention baseline child output differs: '+name)
            results[variant] = dict(score=score, predictions=predictions, teacher_is_not_runtime=True)
        for key, row in binary.items():
            for metric, count in row.items():all_binary[key][metric] += count
        np.savez_compressed(OUT/(name+'_raw_heads.npz'), raw=np.concatenate(raw_values), lengths=lengths)
        receipt = dict(input_sha256=input_hash, base_manifest_sha256=sha256(manifest_path),
            checkpoint_sha256=frozen, learned_heatmaps_and_predictions_exact=True,
            head_samples=summaries(errors, binary), interventions=results, full_contract_pass=False)
        if sha256(input_path) != input_hash or sha256(checkpoint) != frozen:
            raise ValueError('Original source or checkpoint changed during diagnosis')
        dump(OUT/(name+'.json'), receipt); receipts[name] = sha256(OUT/(name+'.json'))
        print('HEAD_INTERVENTIONS', name, {v:(r['score']['valid_children'], r['score']['false_predictions']) for v,r in results.items()}, flush=True)
        del piece, sequences, data, probabilities, raw_values, baseline; gc.collect()
    totals = {v:aggregate(s) for v,s in scores.items()}
    if any(s['truth_count'] != 222 for s in totals.values()):
        raise ValueError('Complete held child denominator changed')
    dump(OUT/'complete.json', dict(protocol_sha256=sha256(OUT/'protocol.json'),
        checkpoint_sha256=frozen, variants=totals, head_samples=summaries(all_errors, all_binary),
        piece_receipts=receipts, learned_baseline_matches_all_nine_frozen_outputs=True,
        actual_trained_models=0, model_adopted=False, original_labels_or_weights_changed=False,
        elapsed_seconds=time.perf_counter()-began, full_contract_pass=False))
    print('HEAD_INTERVENTIONS_COMPLETE', {v:(s['valid_children'],s['false_predictions']) for v,s in totals.items()}, flush=True)


if __name__ == '__main__':
    main()
