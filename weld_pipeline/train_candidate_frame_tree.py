"""Candidate diagnostic with equivalent local frames and shared split head."""
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from training.labels import sha256
from weld_pipeline.candidate_frame_tree_model import FEATURE_VERSION, descriptor, frames, predict_vertical
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.train_candidate_tree import mapped_references, supervision


def main():
    out = Path('weld_pipeline/runs/contract_candidate_frame_tree_seed125')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'model.joblib').exists():
        raise ValueError('Run already completed')
    split_path = Path('weld_pipeline/data/contract_child_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    train = {n: load_piece(split['piece_roots'][n], verify=True) for n in split['train']}
    raw = []
    for name, piece in train.items():
        for row in supervision(piece):
            row.update(name=name, sequence_data=piece['sequences'][row['sequence']])
            raw.append(row)
    def source(name):
        return next((n for n in split['train'] if '__' not in n and (name == n or name.startswith(n + '__') or name.startswith(n + '_f'))), name)
    source_counts = Counter((source(r['name']), r['foreground']) for r in raw)
    inputs, foreground_labels, weights, positive_indices, endpoint_labels = [], [], [], [], []
    for row in raw:
        seq = row['sequence_data']
        length = seq['sample_step_mm'] * (len(seq['xyz']) - 1)
        for transformed, reverse in frames(seq):
            inputs.append(descriptor(transformed))
            foreground_labels.append(row['foreground'])
            weights.append(1. / (16 * source_counts[source(row['name']), row['foreground']]))
            if row['foreground']:
                positive_indices.append(len(inputs) - 1)
                margins = np.array(row['margins']) / length
                endpoint_labels.append(margins[::-1] if reverse else margins)
    x, y, weight = np.stack(inputs), np.array(foreground_labels), np.array(weights)
    positive_indices = np.array(positive_indices)
    side_x, side_y, side_groups = [], [], []
    for name, piece in train.items():
        refs = mapped_references(piece)
        for i, welds in refs.items():
            if len(welds) != 1:
                continue
            w, seq = welds[0], piece['sequences'][i]
            if 'IsSplitedForDualRobot (Any)' not in w['metadata']:
                continue
            side = int(bool(w['metadata']['IsSplitedForDualRobot (Any)']))
            options = frames(seq) if seq['type'] == 'vertical' else [(seq, False)]
            for transformed, reverse in options:
                side_x.append(descriptor(transformed))
                side_y.append(side)
                side_groups.append((source(name), side))
    side_counts = Counter(side_groups)
    side_weights = np.array([1. / side_counts[g] for g in side_groups])
    side_x, side_y = np.stack(side_x), np.array(side_y)
    positives = [r for r in raw if r['foreground']]
    for r in positives:
        expected = 3 if r['split_flag'] else 1
        if r['segment_count'] != expected:
            raise ValueError('Unsupported vertical split flag/count correspondence in TRAIN')
    fractions = {str(c): np.median(np.array([r['fractions'] for r in positives if r['segment_count'] == c]), axis=0).tolist()
                 for c in sorted({r['segment_count'] for r in positives})}
    paths = [__file__, 'weld_pipeline/candidate_frame_tree_model.py', 'weld_pipeline/candidate_tree_model.py',
             'weld_pipeline/train_candidate_tree.py', 'weld_pipeline/sequence_augmentation.py',
             'weld_pipeline/vertical_frame_augmentation.py', 'weld_pipeline/contract_child_evaluation.py']
    config = {'version': FEATURE_VERSION, 'seed': 125, 'trees': 256, 'split_sha256': sha256(split_path),
              'feature_policy': 'Point-local distance profiles and length; all world axes/positions/names/extents excluded.',
              'augmentation': '16 exact equivalent vertical frames; source-balanced weights; unknown negatives excluded.',
              'split_head': 'Shared supervised parent Any flag from original TRAIN H/V and existing augmentations; no child Any broadcast.',
              'topology_policy': 'TRAIN vertical flag/count relationship verified (false=1,true=3); relative cuts fitted to TRAIN only. Development diagnostic, not a proven universal process rule.',
              'foreground_threshold': .5, 'side_threshold': .5, 'split_fractions': fractions,
              'sources': {p: sha256(p) for p in paths}, 'train': split['train'], 'validation': split['validation']}
    archive = out / 'source_snapshot'
    archive.mkdir(exist_ok=True)
    for path in paths:
        shutil.copy2(path, archive / Path(path).name)
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    start = time.perf_counter()
    foreground = ExtraTreesClassifier(n_estimators=256, max_features=.5, class_weight='balanced', n_jobs=4, random_state=125)
    endpoints = ExtraTreesRegressor(n_estimators=256, max_features=.5, n_jobs=4, random_state=125)
    side_split = ExtraTreesClassifier(n_estimators=256, max_features=.5, class_weight='balanced', n_jobs=4, random_state=125)
    foreground.fit(x, y, sample_weight=weight)
    endpoints.fit(x[positive_indices], np.array(endpoint_labels), sample_weight=weight[positive_indices])
    side_split.fit(side_x, side_y, sample_weight=side_weights)
    bundle = {'foreground': foreground, 'endpoints': endpoints, 'side_split': side_split,
              'split_fractions': fractions, 'foreground_threshold': .5, 'config': config}
    joblib.dump(bundle, out / 'model.joblib')
    print('TRAINED', len(x), len(positive_indices), len(side_x), 'seconds', round(time.perf_counter()-start, 3), flush=True)
    for role, names in [('TRAIN', [n for n in split['train'] if '__' not in n]), ('DEV', split['validation'])]:
        scores = {}
        for name in names:
            piece = train[name] if name in train else load_piece(split['piece_roots'][name], verify=True)
            pred, diagnostic = predict_vertical(bundle, piece['sequences'])
            truth = [w for w in piece['truth'] if w['type'] == 'vertical']
            score = score_children(pred, truth)
            scores[name] = score
            (out / (role + '_' + name + '.json')).write_text(json.dumps({'predictions': pred, 'diagnostic': diagnostic,
                'score': score}, ensure_ascii=False, indent=2), encoding='utf8')
            if truth or pred:
                print(role, name, json.dumps({k: score[k] for k in ['truth_count', 'prediction_count', 'valid_children', 'false_predictions']}), flush=True)
        summary = aggregate(scores)
        (out / (role + '_summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf8')
        print(role, json.dumps(summary), flush=True)
    (out / 'complete.json').write_text(json.dumps({'elapsed_seconds': time.perf_counter()-start,
        'full_contract_pass': False, 'baseline_replaced': False}), encoding='utf8')


if __name__ == '__main__':
    main()
