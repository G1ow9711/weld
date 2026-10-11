"""A preregistered capacity trial; FIT decides eligibility before check reads."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.horizontal_independent_heatmaps import VERSION, HEATMAP_NAMES, predict_straight
from weld_pipeline.horizontal_patch_tree import FEATURE_COUNT, equivalent_frame, patch_features, training_indices
from weld_pipeline.train_horizontal_patch_tree import context
from weld_pipeline.typed_contact_model import infer_probabilities


def evaluate(bundle, curve_model, split, names, out, role):
    scores = {}
    for name in names:
        piece = load_piece(split['piece_roots'][name], verify=True)
        structure, mean, covariance = context(split, name)
        predictions = predict_straight(bundle, piece['sequences'], structure, mean, covariance)
        arcs = [s for s in piece['sequences'] if s.get('curve_kind') == 'arc']
        if arcs:
            predictions.extend(decode_intervals(arcs, infer_probabilities(curve_model, arcs),
                threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.))
        for i, prediction in enumerate(predictions):
            prediction['id'] = 'horizontal_' + str(i + 1)
        score = score_children(predictions, [w for w in piece['truth'] if w['type'] == 'horizontal'])
        scores[name] = score
        (out / (role + '_' + name + '.json')).write_text(
            json.dumps({'predictions': predictions, 'score': score}, indent=2), encoding='utf8')
        print(role, name, score['valid_children'], '/', score['truth_count'],
              'FP', score['false_predictions'], flush=True)
    summary = aggregate(scores)
    summary.update(role=role, independent_acceptance=False, full_contract_pass=False)
    (out / (role + '_summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf8')
    return summary


def collision_audit(features, labels):
    seen = {}; conflicting = set(); duplicate_rows = 0
    for feature, target in zip(features, labels):
        key = hashlib.blake2b(feature.tobytes(), digest_size=16).digest()
        if key in seen:
            duplicate_rows += 1
            if not np.array_equal(seen[key], target):
                conflicting.add(key)
        else:
            seen[key] = target.copy()
    return {'exact_feature_duplicate_rows': duplicate_rows,
            'exact_feature_groups_with_conflicting_heatmaps': len(conflicting),
            'unique_profiles': len(seen), 'rounded_or_near_collisions_not_tested': True}


def main():
    torch.set_num_threads(3)
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    names = [n for n in split['train'] if '__' not in n]
    out = Path('weld_pipeline/runs/horizontal_independent_heatmaps_seed139')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise ValueError('Frozen run already exists; resume its saved rows/heads explicitly')
    sources = [__file__, 'weld_pipeline/horizontal_independent_heatmaps.py',
        'weld_pipeline/horizontal_patch_tree.py', 'weld_pipeline/train_horizontal_patch_tree.py',
        'weld_pipeline/event_gap_decoder.py', 'weld_pipeline/constrained_decoder.py',
        'weld_pipeline/sequence_augmentation.py', 'weld_pipeline/contract_child_evaluation.py']
    config = {'version': VERSION, 'seed': 139, 'trees_per_head': 64,
        'max_features': .25, 'max_depth': None, 'min_samples_leaf': 1,
        'features': FEATURE_COUNT, 'train': names, 'observed_check': split['validation'],
        'split_sha256': sha256(split_path), 'sources': {p: sha256(p) for p in sources},
        'source_balance': True, 'equivalent_frames_per_sequence': 2,
        'sampling_policy': 'unchanged136_known_rows',
        'decoder': {'threshold': .5, 'weld_penalty': 2., 'mask_weight': .25},
        'selection_policy': 'Fixed single trial; FIT eligibility written before check labels read.',
        'input_exclusions': ['part_name', 'part_id', 'absolute_xyz', 'absolute_process_planes',
                             'runtime_step', 'runtime_truth'],
        'role': 'FIT_NOT_INDEPENDENT', 'full_contract_pass': False}
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    archive = out / 'source_snapshot'; archive.mkdir(exist_ok=True)
    for source in sources:
        shutil.copy2(source, archive / Path(source).name)
    started = time.perf_counter(); rng = np.random.default_rng(139)
    x, y, weights, counts = [], [], [], {}
    for name in names:
        piece = load_piece(split['piece_roots'][name], verify=True)
        structure, mean, covariance = context(split, name)
        first = len(x)
        for index, (sequence, target) in enumerate(zip(piece['sequences'], piece['targets'])):
            if sequence['type'] != 'horizontal' or sequence.get('curve_kind') == 'arc':
                continue
            for reverse in (False, True):
                seq, label = equivalent_frame(sequence, target, reverse=reverse, flip=bool(rng.integers(2)))
                indices = training_indices(label, rng)
                if not len(indices):
                    continue
                x.append(patch_features(seq, indices, structure[index], mean, covariance))
                y.append(label[indices, :4].copy())
        size = sum(len(block) for block in x[first:]); counts[name] = size
        if size:
            weights.extend(np.full(len(block), 1. / size, float) for block in x[first:])
        print('SAMPLES', name, size, flush=True)
    xx, yy, ww = np.concatenate(x), np.concatenate(y), np.concatenate(weights)
    del x, y, weights
    if np.any(yy < 0) or not np.isfinite(xx).all():
        raise ValueError('Unknown or invalid supervised rows')
    np.save(out / 'training_features.npy', xx)
    np.savez(out / 'training_targets_weights.npz', targets=yy, weights=ww)
    audit = collision_audit(xx, yy)
    (out / 'sampling_receipt.json').write_text(json.dumps({'counts': counts, 'rows': len(xx),
        'collision_audit': audit, 'features_sha256': sha256(out / 'training_features.npy'),
        'targets_weights_sha256': sha256(out / 'training_targets_weights.npz')}, indent=2), encoding='utf8')
    print('COLLISION_AUDIT', audit, flush=True)
    heads = {}
    for column, name in enumerate(HEATMAP_NAMES):
        head = ExtraTreesRegressor(n_estimators=64, max_features=.25, max_depth=None,
            min_samples_leaf=1, n_jobs=3, random_state=139 + column)
        print('HEAD_START', name, xx.shape, flush=True)
        head.fit(xx, yy[:, column], sample_weight=ww)
        heads[name] = head
        joblib.dump(head, out / (name + '.joblib'))
        print('HEAD_SAVED', name, round(time.perf_counter() - started, 2), flush=True)
    bundle = {'heads': heads, 'config': config}
    joblib.dump(bundle, out / 'heads.joblib')
    del xx, yy, ww
    curve = from_checkpoint(torch.load('weld_pipeline/runs/contract_full_fit_horizontal_seed128/best.pt',
                                      map_location='cpu')).cuda().eval()
    fit = evaluate(bundle, curve, split, names, out, 'FIT')
    previous = json.loads(Path('weld_pipeline/runs/contract_full_fit_horizontal_seed128/complete.json').read_text('utf8'))['fit_summary']
    eligible = (fit['geometry_core_gates_pass'] and fit['valid_children'] >= previous['valid_children']
                and fit['false_predictions'] <= previous['false_predictions'])
    (out / 'fit_adoption_decision.json').write_text(json.dumps({'eligible_on_fit': eligible,
        'fit': fit, 'prior': previous, 'check_used_for_decision': False}, indent=2), encoding='utf8')
    print('FIT_DECISION', eligible, fit['valid_children'], fit['false_predictions'], flush=True)
    check = evaluate(bundle, curve, split, split['validation'], out, 'OBSERVED_CHECK')
    receipt = {'elapsed_seconds': time.perf_counter() - started, 'rows': sum(counts.values()),
        'fit_summary': fit, 'observed_check_summary': check, 'eligible_on_fit': eligible,
        'heads_sha256': sha256(out / 'heads.joblib'), 'full_contract_pass': False}
    (out / 'complete.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print('INDEPENDENT_HEADS_COMPLETE', fit['valid_children'], check['valid_children'],
          'FULL_CONTRACT_NOT_ACHIEVED', flush=True)


if __name__ == '__main__':
    main()
