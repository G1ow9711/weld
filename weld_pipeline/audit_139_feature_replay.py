"""Replay frozen FIT sampling and separate fitting, orientation and grid errors."""
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.horizontal_independent_heatmaps import HEATMAP_NAMES
from weld_pipeline.horizontal_patch_tree import equivalent_frame, patch_features, training_indices
from weld_pipeline.train_horizontal_patch_tree import context


def new_stats():
    return dict(rows=0, absolute_sum=[0.] * 4, square_sum=[0.] * 4,
                max_absolute=[0.] * 4, threshold_errors=[0] * 4,
                target_positive=[0] * 4, false_positive=[0] * 4,
                false_negative=[0] * 4)


def accumulate(stats, predicted, target):
    if not len(target):
        return
    error = np.abs(np.asarray(predicted, float) - target)
    expected, actual = target >= .5, predicted >= .5
    stats['rows'] += len(target)
    for column in range(4):
        stats['absolute_sum'][column] += float(error[:, column].sum())
        stats['square_sum'][column] += float(np.square(error[:, column]).sum())
        stats['max_absolute'][column] = max(stats['max_absolute'][column], float(error[:, column].max()))
        stats['threshold_errors'][column] += int((expected[:, column] != actual[:, column]).sum())
        stats['target_positive'][column] += int(expected[:, column].sum())
        stats['false_positive'][column] += int((actual[:, column] & ~expected[:, column]).sum())
        stats['false_negative'][column] += int((~actual[:, column] & expected[:, column]).sum())


def finish_stats(stats):
    size = stats['rows']
    return {**stats, 'heads': list(HEATMAP_NAMES),
            'mae': [v / size for v in stats['absolute_sum']] if size else None,
            'rmse': [float(np.sqrt(v / size)) for v in stats['square_sum']] if size else None,
            'threshold_error_fraction': [v / size for v in stats['threshold_errors']] if size else None}


def predict(heads, features):
    return np.column_stack([heads[name].predict(features) for name in HEATMAP_NAMES]).astype(np.float32)


def main():
    started = time.perf_counter()
    run = Path('weld_pipeline/runs/horizontal_independent_heatmaps_seed139')
    out = Path('weld_pipeline/reports/feature_replay_139_20261001')
    if out.exists():
        raise ValueError('Diagnostic directory exists; do not overwrite')
    out.mkdir(parents=True)
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    frozen = json.loads((run / 'config.json').read_text('utf8'))
    receipt = json.loads((run / 'sampling_receipt.json').read_text('utf8'))
    names = [name for name in split['train'] if '__' not in name]
    if names != frozen['train'] or sha256(split_path) != frozen['split_sha256']:
        raise ValueError('Training order or split changed')
    for source, expected in frozen['sources'].items():
        if sha256(source) != expected:
            raise ValueError('Frozen source differs: ' + source)
    for filename, key in [('training_features.npy', 'features_sha256'),
                          ('training_targets_weights.npz', 'targets_weights_sha256')]:
        if sha256(run / filename) != receipt[key]:
            raise ValueError('Frozen rows differ: ' + filename)
    config = dict(role='FIT_DIAGNOSTIC_NOT_INDEPENDENT_ACCEPTANCE', pieces=names,
                  model_sha256=sha256(run / 'heads.joblib'), source_sha256=sha256(__file__),
                  saved_features_sha256=receipt['features_sha256'],
                  saved_targets_sha256=receipt['targets_weights_sha256'],
                  check_truth_read=False, full_contract_pass=False,
                  decoder=dict(threshold=.5, weld_penalty=2., mask_weight=.25, event_gap_mm=0.))
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    shutil.copy2(__file__, out / Path(__file__).name)
    xx = np.load(run / 'training_features.npy', mmap_mode='r')
    with np.load(run / 'training_targets_weights.npz', allow_pickle=False) as saved:
        yy, ww = saved['targets'], saved['weights']
    heads = joblib.load(run / 'heads.joblib')['heads']
    for head in heads.values():
        head.n_jobs = 2
    rng = np.random.default_rng(139)
    row = 0
    totals = {name: new_stats() for name in ['saved_sampled_rows', 'canonical_sampled_rows',
                                             'canonical_unsampled_rows']}
    scores, pieces, mismatches = {}, {}, []
    geometry_known_excluded = 0
    original_frame_rows = 0
    for name in names:
        piece = load_piece(split['piece_roots'][name], verify=True)
        structure, mean, covariance = context(split, name)
        first_row = row
        local = {key: new_stats() for key in totals}
        sequence_rows, values, mismatched_features, mismatched_targets = [], [], 0, 0
        for sequence_index, (sequence, target) in enumerate(zip(piece['sequences'], piece['targets'])):
            length = len(target)
            probabilities = np.zeros((length, 7), dtype=np.float32)
            values.append(probabilities)
            if sequence['type'] != 'horizontal' or sequence.get('curve_kind') == 'arc':
                continue
            selected_original = np.zeros(length, dtype=bool)
            frames = []
            geometry_known = np.all(target[:, :4] >= 0, axis=1)
            known_all = np.all(target >= 0, axis=1)
            geometry_known_excluded += int((geometry_known & ~known_all).sum())
            for reverse in (False, True):
                flip = bool(rng.integers(2))
                seq, labels = equivalent_frame(sequence, target, reverse=reverse, flip=flip)
                indices = training_indices(labels, rng)
                if not len(indices):
                    continue
                features = patch_features(seq, indices, structure[sequence_index], mean, covariance)
                expected = labels[indices, :4]
                count = len(indices)
                saved_features = xx[row:row + count]
                saved_labels = yy[row:row + count]
                feature_diff = np.any(features != saved_features, axis=1)
                target_diff = np.any(expected != saved_labels, axis=1)
                mismatched_features += int(feature_diff.sum())
                mismatched_targets += int(target_diff.sum())
                if feature_diff.any() or target_diff.any():
                    mismatches.append(dict(piece=name, sequence=sequence_index, reverse=reverse, flip=flip,
                        row=row, feature_rows=int(feature_diff.sum()), target_rows=int(target_diff.sum()),
                        maximum_feature_difference=float(np.abs(features-saved_features).max())))
                fitted = predict(heads, saved_features)
                accumulate(local['saved_sampled_rows'], fitted, saved_labels)
                original_indices = length - 1 - indices if reverse else indices
                selected_original[original_indices] = True
                original_frame_rows += int(not reverse and not flip) * count
                frames.append(dict(reverse=reverse, flip=flip, count=count, first_saved_row=row))
                row += count
            for first in range(0, length, 4096):
                idx = np.arange(first, min(first + 4096, length))
                features = patch_features(sequence, idx, structure[sequence_index], mean, covariance)
                probabilities[idx, :4] = predict(heads, features)
            for group, mask in [('canonical_sampled_rows', geometry_known & selected_original),
                                ('canonical_unsampled_rows', geometry_known & ~selected_original)]:
                accumulate(local[group], probabilities[mask, :4], target[mask, :4])
            sequence_rows.append(dict(sequence_index=sequence_index, candidate_id=sequence.get('candidate_id'),
                length=length, sampled_original_rows=int(selected_original.sum()), frames=frames,
                foreground_truth=bool(np.any(target[:, 0] >= .5))))
        count = row-first_row
        if count != receipt['counts'][name] or not np.array_equal(ww[first_row:row], np.full(count, 1./count)):
            raise ValueError('Source count or weights changed: ' + name)
        # The frozen curved head is unchanged; retain its archived predictions,
        # while every straight prediction above is freshly rebuilt from model inputs.
        archived = json.loads((run / ('FIT_' + name + '.json')).read_text('utf8'))
        predictions = decode_intervals(piece['sequences'], values, **config['decoder'])
        predictions.extend(p for p in archived['predictions'] if any(s['kind'] == 'arc' for s in p['segments']))
        for i, prediction in enumerate(predictions):
            prediction['id'] = 'horizontal_' + str(i + 1)
        score = score_children(predictions, [w for w in piece['truth'] if w['type'] == 'horizontal'])
        previous = archived['score']
        if any(score[k] != previous[k] for k in ['truth_count', 'prediction_count', 'valid_children', 'false_predictions']):
            raise ValueError('Recreated inference score differs: ' + name)
        scores[name] = score
        for key in totals:
            for field in totals[key]:
                if field == 'rows':
                    totals[key][field] += local[key][field]
                elif field == 'max_absolute':
                    totals[key][field] = [max(a, b) for a, b in zip(totals[key][field], local[key][field])]
                else:
                    totals[key][field] = [a + b for a, b in zip(totals[key][field], local[key][field])]
        info = dict(rows=count, mismatched_feature_rows=mismatched_features,
                    mismatched_target_rows=mismatched_targets, score=score,
                    statistics={key: finish_stats(stat) for key, stat in local.items()}, sequences=sequence_rows)
        pieces[name] = info
        (out / (name + '.json')).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf8')
        np.savez_compressed(out / (name + '_heatmaps.npz'),
            probabilities=np.concatenate([v[:, :4] for v in values]),
            lengths=np.asarray([len(v) for v in values]),
            sequences=np.asarray([v['sequence_index'] for v in sequence_rows]))
        print('REPLAY', name, count, 'X_DIFF', mismatched_features, 'Y_DIFF', mismatched_targets,
              'SAVED_MAE', finish_stats(local['saved_sampled_rows'])['mae'],
              'TP', score['valid_children'], 'FP', score['false_predictions'], flush=True)
    if row != len(xx):
        raise ValueError('Final reconstructed row count differs')
    summary = dict(config=config, rows=row, feature_dtype=str(xx.dtype),
                   exact_replay=not mismatches, mismatches=mismatches,
                   geometry_known_rows_excluded_by_process_unknown=geometry_known_excluded,
                   original_orientation_training_rows=original_frame_rows,
                   statistics={key: finish_stats(stat) for key, stat in totals.items()},
                   geometry=aggregate(scores), elapsed_seconds=time.perf_counter()-started,
                   curved_predictions_from_frozen_archive=True, full_contract_pass=False)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print('REPLAY_COMPLETE', 'EXACT', not mismatches, 'ROWS', row,
          'TP', summary['geometry']['valid_children'], 'FP', summary['geometry']['false_predictions'], flush=True)


if __name__ == '__main__':
    main()
