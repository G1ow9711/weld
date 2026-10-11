"""Fixed shared auxiliary event-center heads, full unchanged 1221-child FIT gate."""
import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.forest_event_centers import (VERSION, HEADS, FEATURE_COUNT, proximity_features,
    combine_features, centers_to_offsets, infer, build_inputs)
from weld_pipeline.near_event_votes import vote_heatmaps

BASE = Path('weld_pipeline/data/joint_relation_v1')
CONTEXT = Path('weld_pipeline/data/joint_workpiece_context_pilot154')
TARGET = Path('weld_pipeline/data/near_event_pilot156')
GEOMETRY = Path('weld_pipeline/reports/small_finite_surface_events_165v3_20261002')
RAW_PROOF = Path('weld_pipeline/reports/small_finite_surface_runtime_165v3_serialized_20261002/complete.json')
DATA = Path('weld_pipeline/data/forest_event_centers_pilot166')
OUT = Path('weld_pipeline/runs/forest_event_centers_seed166')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def load_cached(name, verify=True):
    root = BASE/name; manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf8'))
    context_path = CONTEXT/name/'manifest.json'
    context = json.loads(context_path.read_text('utf8'))
    near = json.loads((TARGET/'manifest.json').read_text('utf8'))[name]
    evidence_path = GEOMETRY/(name+'.json')
    evidence = json.loads(evidence_path.read_text('utf8'))
    parent_path = Path('weld_pipeline/data/expanded_surface_v3')/name/'manifest.json'
    parent = json.loads(parent_path.read_text('utf8'))
    if (manifest['graph_uses_reference_labels'] or context['context_labels_used']
            or not context['canonical_query_order_exact'] or context['base_manifest_sha256'] != sha256(manifest_path)
            or near['source_manifest_sha256'] != sha256(manifest_path)
            or evidence['parent_manifest_sha256'] != sha256(parent_path)):
        raise ValueError('Frozen point-only feature or target identity differs')
    if verify:
        for p, digest in [(TARGET/(name+'_targets.npy'), near['targets_sha256']),
                          (TARGET/(name+'_known.npy'), near['known_sha256'])]:
            if sha256(p) != digest:
                raise ValueError('Unchanged near supervision required')
        for folder, files in [(root, manifest['files']), (CONTEXT/name, context['files'])]:
            for file, digest in files.items():
                if sha256(folder/file) != digest:
                    raise ValueError('Prepared point-derived feature changed')
        if sha256(parent['record']['input']) != evidence['input_sha256']:
            raise ValueError('Original point input changed')
    with np.load(root/'graph.npz', allow_pickle=False) as archive:
        nodes = archive['nodes'].copy()
    data = dict(local=np.load(root/'features.npy', mmap_mode='r', allow_pickle=False),
        point_context=np.load(CONTEXT/name/'point_context.npy', mmap_mode='r', allow_pickle=False),
        nodes=np.c_[nodes, np.load(CONTEXT/name/'node_context.npy', allow_pickle=False)],
        entries=manifest['entries'], geometry=parent['record']['geometry'], planes=evidence['planes'])
    target = np.load(TARGET/(name+'_targets.npy'), mmap_mode='r', allow_pickle=False)
    known = np.load(TARGET/(name+'_known.npy'), mmap_mode='r', allow_pickle=False)
    if len(data['local']) != len(target) or known.shape != target.shape or data['nodes'].shape[1] != 255:
        raise ValueError('Canonical prepared arrays differ')
    piece = load_piece(manifest['source_cache_root'], verify=verify)
    return data, target, known, piece


def prepare(config):
    if DATA.exists():
        raise ValueError('Preserve fixed sampled capacity inputs')
    DATA.mkdir(parents=True); rng = np.random.default_rng(166); manifests = {}; oracle_scores = {}
    dump(DATA/'config.json', config)
    for name in config['train']:
        data, target, known, piece = load_cached(name)
        horizontal = np.zeros(len(target), bool); event_features = {}; hard = []; truth_votes = [np.zeros((len(s['xyz']), 7), np.float32) for s in piece['sequences']]
        max_inverse_error = 0.
        for entry in data['entries']:
            if entry['type'] != 'horizontal':
                continue
            sequence, _, transform = canonical_frame(piece['sequences'][entry['sequence']])
            if transform != entry['transform']:
                raise ValueError('Canonical raw geometry and cached target order differ')
            a, b = entry['start'], entry['stop']; horizontal[a:b] = True
            event = proximity_features(sequence, data['geometry'], data['planes'])
            event_features[entry['sequence']] = event
            mask = np.all(known[a:b, 1:4] > 0, axis=1) & np.all(target[a:b, 1:4] < .5, axis=1) & (event[:, 8] > 0)
            hard.extend((a+np.flatnonzero(mask)).tolist())
            y = np.asarray(target[a:b], float); k = np.asarray(known[a:b], float)
            step = float(sequence['sample_step_mm']); span = (len(sequence['xyz'])-1)*step
            centers = (np.arange(len(y))[:, None]*step+y[:, 4:7]*16.)/span
            offset = centers_to_offsets(centers, len(y), step)
            if np.any(k[:, 4:7] > 0):
                max_inverse_error = max(max_inverse_error, float(np.abs(offset-y[:, 4:7])[k[:, 4:7] > 0].max()*16.))
            votes = vote_heatmaps(np.clip(y[:, 0], 0., 1.), np.where(k[:, 1:4] > 0, y[:, 1:4], 0.), offset, step)
            truth_votes[entry['sequence']] = restore_probability_order(votes, transform)
        near = horizontal & np.any((target[:, 1:4] > .5) & (known[:, 1:4] > 0), axis=1)
        selected = [np.flatnonzero(near)]
        for positive in (False, True):
            pool = np.flatnonzero(horizontal & (known[:, 0] > 0) & ((target[:, 0] > .5) == positive))
            selected.append(rng.choice(pool, min(4000, len(pool)), replace=False))
        hard = np.asarray(hard, np.int64)
        selected.append(rng.choice(hard, min(2000, len(hard)), replace=False))
        indices = np.unique(np.concatenate(selected)); x = np.empty((len(indices), FEATURE_COUNT), np.float32)
        y = np.asarray(target[indices], np.float32).copy(); k = np.asarray(known[indices], np.float32).copy()
        filled = np.zeros(len(indices), bool)
        for entry in data['entries']:
            if entry['type'] != 'horizontal':
                continue
            rows = np.flatnonzero((indices >= entry['start']) & (indices < entry['stop']))
            if not len(rows):
                continue
            local_indices = indices[rows]-entry['start']; source_indices = indices[rows]
            local = np.c_[data['local'][source_indices], data['point_context'][source_indices]]
            x[rows] = combine_features(local, data['nodes'][entry['node']], event_features[entry['sequence']][local_indices])
            step = piece['sequences'][entry['sequence']]['sample_step_mm']; span = (entry['stop']-entry['start']-1)*step
            y[rows, 4:7] = (local_indices[:, None]*step+y[rows, 4:7].astype(float)*16.)/span
            filled[rows] = True
        if not filled.all() or not np.isfinite(x).all() or max_inverse_error > 1e-6:
            raise ValueError('Shared center representation lost rows or numerical precision')
        root = DATA/name; root.mkdir()
        for file, array in [('features.npy', x), ('targets.npy', y), ('known.npy', k), ('sample_indices.npy', indices)]:
            np.save(root/file, array)
        reference = [w for w in piece['truth'] if w['type'] == 'horizontal' and all(s['kind'] == 'line' for s in w['segments'])]
        predictions = decode_intervals(piece['sequences'], truth_votes, threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
        score = score_children(predictions, reference); oracle_scores[name] = score
        manifest = dict(piece=name, sampled_rows=len(indices), all_event_support_rows=int(near.sum()),
            targets_are_original_near_targets_in_center_coordinates=True, max_center_inverse_error_mm=max_inverse_error,
            base_manifest_sha256=sha256(BASE/name/'manifest.json'), context_manifest_sha256=sha256(CONTEXT/name/'manifest.json'),
            near_target_sha256=sha256(TARGET/(name+'_targets.npy')), near_known_sha256=sha256(TARGET/(name+'_known.npy')),
            geometry_receipt_sha256=sha256(GEOMETRY/(name+'.json')),
            files={file:sha256(root/file) for file in ['features.npy', 'targets.npy', 'known.npy', 'sample_indices.npy']},
            geometry_reference_is_not_a_runtime_input=True, oracle_score=score, full_contract_pass=False)
        dump(root/'manifest.json', manifest); manifests[name] = sha256(root/'manifest.json')
        print('FOREST_CENTERS_PREPARED', name, len(indices), 'ORACLE', score['valid_children'], '/', score['truth_count'], 'FP', score['false_predictions'], flush=True)
        del data, target, known, piece, event_features, truth_votes, x, y, k; gc.collect()
    oracle = aggregate(oracle_scores)
    if oracle['truth_count'] != 1221 or oracle['valid_children'] != 1221 or oracle['false_predictions'] != 0:
        raise ValueError('All 1221 unchanged children must be exactly representable before training')
    dump(DATA/'complete.json', dict(config_sha256=sha256(DATA/'config.json'), manifests=manifests,
        oracle_score=oracle, training_allowed=True, original_targets_unchanged=True, full_contract_pass=False))


def evaluate(bundle, names, split_name, runtime=False):
    root = OUT/split_name; root.mkdir(); scores = {}
    for name in names:
        if runtime:
            manifest = json.loads((BASE/name/'manifest.json').read_text('utf8'))
            piece = load_piece(manifest['source_cache_root'], verify=True)
            sequences, data = build_inputs(read_xyz_pcd(piece['record']['input'])*1000.)
        else:
            data, _, _, piece = load_cached(name)
            sequences = piece['sequences']
        probabilities = infer(bundle, data, sequences)
        predictions = decode_intervals(sequences, probabilities, threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.)
        truth = [w for w in piece['truth'] if w['type'] == 'horizontal' and all(s['kind'] == 'line' for s in w['segments'])]
        score = score_children(predictions, truth); scores[name] = score
        dump(root/(name+'.json'), dict(score=score, predictions=predictions, original_truth_unchanged=True, full_contract_pass=False))
        np.savez_compressed(root/(name+'_heatmaps.npz'), probabilities=np.concatenate(probabilities), lengths=np.asarray([len(p) for p in probabilities]))
        print('FOREST_CENTERS_SCORE', split_name, name, score['valid_children'], '/', score['truth_count'], 'FP', score['false_predictions'], flush=True)
        del data, piece, sequences, probabilities; gc.collect()
    result = aggregate(scores); dump(root/'summary.json', result)
    return result


def main():
    if OUT.exists():
        raise ValueError('Preserve frozen auxiliary center trial')
    raw = json.loads(RAW_PROOF.read_text('utf8')); geometric = json.loads((GEOMETRY/'complete.json').read_text('utf8'))
    old = json.loads(Path('weld_pipeline/runs/near_event_capacity_156/config.json').read_text('utf8'))
    context = json.loads((CONTEXT/'config.json').read_text('utf8'))
    if (not raw['all_checks_passed'] or old['train'] != context['pieces']
            or set(old['train']) & set(old['held']) or geometric['total']['truth_children'] != 1221):
        raise ValueError('Frozen raw proof and complete unchanged source scope required')
    for path, digest in raw['sources'].items():
        if sha256(path) != digest:
            raise ValueError('Raw proof source changed')
    sources = [__file__, 'weld_pipeline/forest_event_centers.py', 'weld_pipeline/small_finite_surface_events_v3.py',
        'weld_pipeline/near_event_votes.py', 'weld_pipeline/event_gap_decoder.py', 'weld_pipeline/contract_child_evaluation.py',
        'weld_pipeline/canonical_horizontal_patch_heatmaps.py']
    config = dict(version=VERSION, train=old['train'], held=old['held'], random_seed=166,
        heads=list(HEADS), feature_count=FEATURE_COUNT, n_estimators=128, max_features=.5, min_samples_leaf=1,
        max_depth=None, n_jobs=2, all_event_support_rows=True, sampled_foreground_rows_per_source_class=4000,
        sampled_physical_event_negative_rows_per_source=2000, explicit_known_masks_preserved=True,
        source_and_class_balanced_weights=True, event_radius_mm=16., center_target='candidate-relative fraction',
        event_sigma_mm=1.5, event_tail_mm=9., vote_scale=4., foreground_threshold=.5,
        weld_penalty=2., mask_weight=.25, event_gap_mm=0.,
        source_sha256={p:sha256(p) for p in sources}, raw_proof_sha256=sha256(RAW_PROOF),
        geometry_complete_sha256=sha256(GEOMETRY/'complete.json'),
        fitting_and_oracle_scope='all 1221 H-line children from the same 16 sources',
        auxiliary_candidate_not_a_complete_deep_learning_delivery=True,
        existing_deep_learning_predictor_and_weights_preserved=True,
        held_gate='Read the 9 held sources only after FIT TP/G >95% and FP/G <=5%; no OBS tuning.',
        no_runtime_lookup_by_name_ID_or_absolute_coordinate=True, attributes_vertical_and_curves_still_required=True,
        full_contract_pass=False)
    OUT.mkdir(parents=True); dump(OUT/'config.json', config); snap = OUT/'source_snapshot'; snap.mkdir()
    for source in sources:
        shutil.copy2(source, snap/Path(source).name)
    began = time.perf_counter(); prepare(config)
    xs = []; ys = []; ks = []; source_ranges = {}; row = 0
    for name in config['train']:
        root = DATA/name; manifest = json.loads((root/'manifest.json').read_text('utf8'))
        for file, digest in manifest['files'].items():
            if sha256(root/file) != digest:
                raise ValueError('Fixed capacity sampling changed')
        x = np.load(root/'features.npy', allow_pickle=False); xs.append(x)
        ys.append(np.load(root/'targets.npy', allow_pickle=False)); ks.append(np.load(root/'known.npy', allow_pickle=False))
        source_ranges[name] = row, row+len(x); row += len(x)
    x = np.concatenate(xs); y = np.concatenate(ys); known = np.concatenate(ks)
    del xs, ys, ks; gc.collect(); heads = {}; receipts = {}
    for channel, key in enumerate(HEADS):
        weight = np.zeros(row, float)
        for name, (a, b) in source_ranges.items():
            valid = known[a:b, channel] > 0
            if channel < 4:
                for positive in (False, True):
                    indices = np.flatnonzero(valid & ((y[a:b, channel] > .5) == positive))+a
                    if len(indices):
                        weight[indices] = 1./(2.*len(indices))
            else:
                indices = np.flatnonzero(valid)+a
                if len(indices):
                    weight[indices] = 1./len(indices)
        indices = np.flatnonzero(weight > 0)
        if not len(indices):
            raise ValueError('A required shared head lacks known supervision: '+key)
        model = ExtraTreesRegressor(n_estimators=128, max_features=.5, min_samples_leaf=1,
            max_depth=None, n_jobs=2, random_state=166+channel)
        print('FOREST_CENTERS_TRAINING', key, 'ROWS', len(indices), 'FEATURES', x.shape[1], flush=True)
        model.fit(x[indices], y[indices, channel], sample_weight=weight[indices])
        checkpoint = OUT/(key+'.joblib'); joblib.dump(model, checkpoint)
        heads[key] = model; receipts[key] = dict(checkpoint_sha256=sha256(checkpoint), known_training_rows=len(indices), trees=len(model.estimators_))
        dump(OUT/(key+'_receipt.json'), receipts[key]); print('FOREST_CENTERS_HEAD_COMPLETE', key, flush=True)
    bundle = dict(config=config, heads=heads); checkpoint = OUT/'candidate.joblib'; joblib.dump(bundle, checkpoint)
    del x, y, known; gc.collect()
    fit = evaluate(bundle, config['train'], 'FIT')
    if fit['truth_count'] != 1221:
        raise ValueError('Full FIT child denominator changed')
    eligible = fit['valid_children']/fit['truth_count'] > .95 and fit['false_predictions']/fit['truth_count'] <= .05
    held = evaluate(bundle, config['held'], 'HELD', runtime=True) if eligible else None
    dump(OUT/'complete.json', dict(config_sha256=sha256(OUT/'config.json'), checkpoint_sha256=sha256(checkpoint),
        actual_trained_shared_candidates=1, actual_trained_heads=7, head_receipts=receipts,
        fit=fit, eligible_on_fit=eligible, held=held, held_or_observed_read=held is not None,
        default_predictor_replaced=False, elapsed_seconds=time.perf_counter()-began,
        input_data_complete_sha256=sha256(DATA/'complete.json'), original_labels_unchanged=True, full_contract_pass=False))
    print('FOREST_CENTERS_COMPLETE', 'FIT', fit['valid_children'], '/', fit['truth_count'],
          'FP', fit['false_predictions'], 'ELIGIBLE_ON_FIT', eligible, flush=True)


if __name__ == '__main__':
    main()
