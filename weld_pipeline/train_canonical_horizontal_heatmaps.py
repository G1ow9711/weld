"""Fixed FIT-only canonical sampling repair; check read requires FIT eligibility."""
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import VERSION, canonical_frame, predict_straight
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.horizontal_independent_heatmaps import HEATMAP_NAMES
from weld_pipeline.horizontal_patch_tree import FEATURE_COUNT, patch_features
from weld_pipeline.train_horizontal_independent_heatmaps import collision_audit
from weld_pipeline.train_horizontal_patch_tree import context
from weld_pipeline.typed_contact_model import infer_probabilities


def select_rows(target, previous, step, rng):
    known = np.all(target[:, :4] >= 0, axis=1)
    stride = max(1, int(round(16. / step)))
    phase = int(rng.integers(stride))
    uniform = np.arange(phase, len(target), stride)
    groups = [uniform, np.flatnonzero(known & (target[:, 1:4].max(1) > .001)),
              np.flatnonzero(known & (np.abs(previous-target[:, :4]).max(1) > .15))]
    foreground = target[:, 0] >= .5
    boundaries = np.flatnonzero(np.diff(foreground.astype(np.int8)))
    radius = int(np.ceil(16. / step))
    if len(boundaries):
        neighbourhood = boundaries[:, None] + np.arange(-radius, radius+2)[None, :]
        groups.append(np.clip(neighbourhood.ravel(), 0, len(target)-1))
    for label, maximum in [(False, 80), (True, 48)]:
        candidates = np.flatnonzero(known & (foreground == label))
        if len(candidates):
            groups.append(rng.choice(candidates, min(maximum, len(candidates)), replace=False))
    indices = np.unique(np.concatenate(groups)).astype(np.int64)
    return indices[known[indices]]


def evaluate(bundle, curve_model, split, names, out, role):
    scores = {}
    for name in names:
        piece = load_piece(split['piece_roots'][name], verify=True)
        structure, mean, covariance = context(split, name)
        predictions = predict_straight(bundle, piece['sequences'], structure, mean, covariance)
        arcs = [sequence for sequence in piece['sequences'] if sequence.get('curve_kind') == 'arc']
        if arcs:
            predictions.extend(decode_intervals(arcs, infer_probabilities(curve_model, arcs),
                threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.))
        for i, prediction in enumerate(predictions):
            prediction['id'] = 'horizontal_' + str(i+1)
        score = score_children(predictions, [w for w in piece['truth'] if w['type'] == 'horizontal'])
        scores[name] = score
        (out / (role+'_'+name+'.json')).write_text(json.dumps(dict(predictions=predictions, score=score), indent=2), encoding='utf8')
        print(role, name, score['valid_children'], '/', score['truth_count'], 'FP', score['false_predictions'], flush=True)
    result = aggregate(scores)
    result.update(role=role, independent_acceptance=False, full_contract_pass=False)
    (out / (role+'_summary.json')).write_text(json.dumps(result, indent=2), encoding='utf8')
    return result


def main():
    started = time.perf_counter()
    torch.set_num_threads(2)
    out = Path('weld_pipeline/runs/canonical_horizontal_heatmaps_seed141')
    if out.exists():
        raise ValueError('Fixed trial exists; preserve it')
    out.mkdir(parents=True)
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    names = [n for n in split['train'] if '__' not in n]
    diagnostic = Path('weld_pipeline/reports/feature_replay_139_20261001')
    replay = json.loads((diagnostic / 'summary.json').read_text('utf8'))
    if not replay['exact_replay'] or replay['config']['pieces'] != names:
        raise ValueError('Completed exact FIT replay required')
    source_paths = [__file__, 'weld_pipeline/canonical_horizontal_patch_heatmaps.py',
        'weld_pipeline/horizontal_patch_tree.py', 'weld_pipeline/sequence_augmentation.py',
        'weld_pipeline/event_gap_decoder.py', 'weld_pipeline/contract_child_evaluation.py',
        'weld_pipeline/train_horizontal_patch_tree.py']
    config = dict(version=VERSION, seed=141, trees_per_head=64, max_features=.25,
        max_depth=None, min_samples_leaf=1, feature_count=FEATURE_COUNT, train=names,
        observed_check=split['validation'], split_sha256=sha256(split_path),
        sources={path: sha256(path) for path in source_paths},
        replay_sha256=sha256(diagnostic / 'summary.json'), source_balance=True,
        sampling=dict(grid_mm=16., background_per_candidate=80, foreground_per_candidate=48,
                      all_event_target_above=.001, foreground_transition_neighbourhood_mm=16.,
                      all_prior_FIT_heatmap_errors_above=.15),
        equivalent_frame='largest_tangent_and_first_contact_normal_component_positive',
        decoder=dict(threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.),
        exclusions=['part_name_as_input', 'absolute_xyz', 'absolute_process_planes', 'runtime_STEP', 'runtime_truth'],
        selection='Single fixed trial. FIT geometry gates, TP>=1432, FP<=50 before observed-check read.',
        full_contract_pass=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    archive = out / 'source_snapshot'; archive.mkdir()
    for path in source_paths:
        shutil.copy2(path, archive / Path(path).name)
    rng = np.random.default_rng(141)
    plans, counts, priors, total = {}, {}, {}, 0
    for name in names:
        piece = load_piece(split['piece_roots'][name], verify=True)
        heatmap_path = diagnostic / (name+'_heatmaps.npz')
        priors[name] = sha256(heatmap_path)
        with np.load(heatmap_path, allow_pickle=False) as arrays:
            previous, lengths = arrays['probabilities'], arrays['lengths']
        if not np.array_equal(lengths, [len(sequence['features']) for sequence in piece['sequences']]):
            raise ValueError('FIT hard-point candidate order differs: '+name)
        per_sequence, offset, count = {}, 0, 0
        for index, (seq, target) in enumerate(zip(piece['sequences'], piece['targets'])):
            length = len(target)
            if seq['type'] == 'horizontal' and seq.get('curve_kind') != 'arc':
                indices = select_rows(target, previous[offset:offset+length], seq['sample_step_mm'], rng)
                per_sequence[index] = indices
                count += len(indices)
            offset += length
        plans[name], counts[name] = per_sequence, count
        total += count
        np.savez_compressed(out / (name+'_sample_indices.npz'), **{str(i): v for i, v in per_sequence.items()})
        print('CANONICAL_SAMPLES', name, count, flush=True)
    x_path = out / 'training_features.npy'
    x = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.float32, shape=(total, FEATURE_COUNT))
    y = np.zeros((total, 4), np.float32)
    weights = np.zeros(total, float)
    source_rows, row = {}, 0
    for name in names:
        first = row
        piece = load_piece(split['piece_roots'][name], verify=True)
        structural, mean, covariance = context(split, name)
        for index, original_indices in plans[name].items():
            original, target = piece['sequences'][index], piece['targets'][index]
            sequence, labels, transformation = canonical_frame(original, target)
            indices = len(labels)-1-original_indices if transformation['reverse'] else original_indices
            for start in range(0, len(indices), 8192):
                block = indices[start:start+8192]
                stop = row+len(block)
                x[row:stop] = patch_features(sequence, block, structural[index], mean, covariance)
                y[row:stop] = labels[block, :4]
                weights[row:stop] = 1. / counts[name]
                row = stop
        source_rows[name] = dict(first_row=first, stop_row=row, rows=row-first,
                                  cache_manifest_sha256=piece['cache_manifest_sha256'])
        if row-first != counts[name]:
            raise ValueError('Prepared rows differ')
        print('CANONICAL_PREPARED', name, row-first, flush=True)
    if row != total or np.any(y < 0) or not np.isfinite(y).all():
        raise ValueError('Supervised row preparation invalid')
    x.flush()
    np.savez(out / 'training_targets_weights.npz', targets=y, weights=weights)
    del plans
    audit = collision_audit(x, y)
    sampling = dict(rows=total, sources=source_rows, collision_audit=audit,
                    features_sha256=sha256(x_path),
                    targets_weights_sha256=sha256(out / 'training_targets_weights.npz'),
                    previous_FIT_heatmaps_sha256=priors, observed_check_used=False)
    (out / 'sampling_receipt.json').write_text(json.dumps(sampling, indent=2, ensure_ascii=False), encoding='utf8')
    print('CANONICAL_COLLISIONS', audit, 'TOTAL', total, flush=True)
    if audit['exact_feature_groups_with_conflicting_heatmaps']:
        raise ValueError('Conflicting canonical sample heatmaps require representation review')
    heads = {}
    for column, name in enumerate(HEATMAP_NAMES):
        print('CANONICAL_HEAD_START', name, 'ROWS', total, flush=True)
        head = ExtraTreesRegressor(n_estimators=64, max_features=.25, max_depth=None,
                                   min_samples_leaf=1, n_jobs=2, random_state=141+column)
        head.fit(x, y[:, column], sample_weight=weights)
        heads[name] = head
        joblib.dump(head, out / (name+'.joblib'))
        print('CANONICAL_HEAD_SAVED', name, round(time.perf_counter()-started, 2), flush=True)
    bundle = dict(heads=heads, config=config)
    joblib.dump(bundle, out / 'heads.joblib')
    del x, y, weights
    curve_path = Path('weld_pipeline/runs/contract_full_fit_horizontal_seed128/best.pt')
    curve = from_checkpoint(torch.load(curve_path, map_location='cpu')).cuda().eval()
    fit = evaluate(bundle, curve, split, names, out, 'FIT')
    eligible = bool(fit['geometry_core_gates_pass'] and fit['valid_children'] >= 1432 and fit['false_predictions'] <= 50)
    (out / 'fit_adoption_decision.json').write_text(json.dumps(dict(eligible_on_fit=eligible, fit=fit,
        check_used_for_decision=False, checkpoint_frozen_before_check=True), indent=2), encoding='utf8')
    print('CANONICAL_FIT_DECISION', eligible, fit['valid_children'], fit['false_predictions'], flush=True)
    check = evaluate(bundle, curve, split, split['validation'], out, 'OBSERVED_CHECK') if eligible else None
    result = dict(rows=total, fit_summary=fit, observed_check_summary=check,
                  observed_check_status='FIXED_POST_FIT_OBSERVED_NOT_BLIND' if eligible else 'NOT_RUN_FIT_GATE_FAILED',
                  eligible_on_fit=eligible, elapsed_seconds=time.perf_counter()-started,
                  heads_sha256=sha256(out / 'heads.joblib'), retained_v4_unchanged=True,
                  full_contract_pass=False)
    (out / 'complete.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print('CANONICAL_COMPLETE', fit['valid_children'], 'CHECK',
          check['valid_children'] if check else 'NOT_RUN', 'FULL_CONTRACT_NOT_ACHIEVED', flush=True)


if __name__ == '__main__':
    main()
