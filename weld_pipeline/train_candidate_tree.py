"""TRAIN-only candidate classification/regression diagnostic, fixed decoder.

References are used for supervision and evaluation only. The predictor in
candidate_tree_model receives feature sequences plus a trained bundle.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from training.labels import sha256
from weld_pipeline.candidate_tree_model import FEATURE_VERSION, descriptor, predict_vertical
from weld_pipeline.contract_child_evaluation import aggregate, score_children
from weld_pipeline.curved_cache import load_piece


def mapped_references(piece):
    metadata = piece['record']['mapping']
    mapping = metadata.get('line_mapping', metadata)['weld_to_sequence']
    ids = [i for i, s in enumerate(piece['sequences']) if s.get('curve_kind') != 'arc']
    refs = [w for w in piece['truth'] if all(s['kind'] == 'line' for s in w['segments'])]
    by_sequence = {}
    for wi, si in mapping.items():
        by_sequence.setdefault(ids[int(si)], []).append(refs[int(wi)])
    return by_sequence


def supervision(piece):
    refs = mapped_references(piece)
    rows = []
    for i, (seq, target) in enumerate(zip(piece['sequences'], piece['targets'])):
        if seq['type'] != 'vertical':
            continue
        known = target[:, 0] >= 0
        positive = bool((target[:, 0] > .5).any())
        if not positive and not known.all():
            continue
        row = {'sequence': i, 'x': descriptor(seq), 'foreground': int(positive)}
        if positive:
            welds = refs.get(i, [])
            if len(welds) != 1:
                raise ValueError('Multiple or missing vertical reference parents on a candidate')
            w = welds[0]
            points = np.array([p for seg in w['segments'] for p in [seg['points'][0], seg['points'][-1]]])
            position = (points - seq['xyz'][0]) @ seq['direction']
            start, end = float(position.min()), float(position.max())
            length = (len(seq['xyz']) - 1) * seq['sample_step_mm']
            boundaries = np.unique(np.round((position - start) / (end - start), 6))
            row.update(margins=[start, length - end], segment_count=len(w['segments']),
                       fractions=boundaries.tolist(), split_flag=w['metadata'].get('IsSplitedForDualRobot (Any)'))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='weld_pipeline/runs/contract_candidate_tree_seed124')
    ap.add_argument('--seed', type=int, default=124)
    ap.add_argument('--trees', type=int, default=384)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'model.joblib').exists():
        raise ValueError('Completed run already exists')
    split_path = Path('weld_pipeline/data/contract_child_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    train = {n: load_piece(split['piece_roots'][n], verify=True) for n in split['train']}
    supervision_rows, coverage = [], []
    for name, piece in train.items():
        rows = supervision(piece)
        counts = Counter(r['foreground'] for r in rows)
        coverage.append({'name': name, 'known_candidates': len(rows), 'foreground_counts': dict(counts)})
        # Equal mass for each source group and class inside that group. Partial
        # positive context is retained; unknown candidates are not negatives.
        for row in rows:
            row.update(name=name, weight=1. / max(1, counts[row['foreground']]))
            supervision_rows.append(row)
    x = np.stack([r['x'] for r in supervision_rows])
    y = np.array([r['foreground'] for r in supervision_rows])
    weights = np.array([r['weight'] for r in supervision_rows])
    positives = [r for r in supervision_rows if r['foreground']]
    xp = np.stack([r['x'] for r in positives])
    yp = np.array([r['margins'] for r in positives])
    count = np.array([r['segment_count'] for r in positives])
    positive_weights = np.array([r['weight'] for r in positives])
    fractions = {}
    for c in sorted(set(count)):
        samples = np.array([r['fractions'] for r in positives if r['segment_count'] == c])
        if samples.shape[1] != c + 1:
            raise ValueError('Noncontiguous vertical child supervision')
        # A single global learned relative-cut template per topology. This is
        # fitted solely on TRAIN, with no object ID/XYZ lookup or DEV geometry.
        fractions[str(c)] = np.median(samples, axis=0).tolist()
    config = {'version': FEATURE_VERSION, 'seed': args.seed, 'trees': args.trees,
              'split_sha256': sha256(split_path), 'train': split['train'], 'validation': split['validation'],
              'feature_policy': 'Local 63-channel point-distance profiles, sequence length and contact axes only; no global position, extent, identifiers or label features.',
              'supervision_policy': 'Original TRAIN + existing rigid/partial augmentation; unknown negative candidates excluded.',
              'foreground_threshold': .5, 'split_fractions': fractions, 'sklearn_version': sklearn.__version__,
              'scope': 'Development diagnostic; single real TRAIN vertical-positive source, not blind acceptance.',
              'sources': {p: sha256(p) for p in [__file__, 'weld_pipeline/candidate_tree_model.py']},
              'manifests': {n: sha256(Path(split['piece_roots'][n]) / 'manifest.json') for n in split['train']}}
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    (out / 'training_coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding='utf8')
    start = time.perf_counter()
    foreground = ExtraTreesClassifier(n_estimators=args.trees, max_features=.5, min_samples_leaf=1,
                                      class_weight='balanced', n_jobs=4, random_state=args.seed)
    endpoints = ExtraTreesRegressor(n_estimators=args.trees, max_features=.5, n_jobs=4, random_state=args.seed)
    segments = ExtraTreesClassifier(n_estimators=args.trees, max_features=.5, class_weight='balanced',
                                    n_jobs=4, random_state=args.seed)
    foreground.fit(x, y, sample_weight=weights)
    endpoints.fit(xp, yp, sample_weight=positive_weights)
    segments.fit(xp, count, sample_weight=positive_weights)
    bundle = {'foreground': foreground, 'endpoints': endpoints, 'segments': segments,
              'split_fractions': fractions, 'foreground_threshold': .5, 'config': config}
    joblib.dump(bundle, out / 'model.joblib')
    print('TRAINED', len(supervision_rows), len(positives), 'seconds', round(time.perf_counter() - start, 3), flush=True)
    for role, names in [('TRAIN', [n for n in split['train'] if '__' not in n]), ('DEV', split['validation'])]:
        scores = {}
        for name in names:
            piece = train[name] if name in train else load_piece(split['piece_roots'][name], verify=True)
            pred, diagnostic = predict_vertical(bundle, piece['sequences'])
            truth = [w for w in piece['truth'] if w['type'] == 'vertical']
            score = score_children(pred, truth)
            scores[name] = score
            result = {'predictions': pred, 'diagnostic': diagnostic, 'score': score}
            (out / (role + '_' + name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
            if truth or pred:
                print(role, name, json.dumps({k: score[k] for k in ['truth_count', 'prediction_count', 'valid_children', 'false_predictions']}), flush=True)
        summary = aggregate(scores)
        (out / (role + '_summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf8')
        print(role, json.dumps(summary), flush=True)
    (out / 'complete.json').write_text(json.dumps({'elapsed_seconds': time.perf_counter() - start,
        'full_contract_pass': False, 'baseline_replaced': False}), encoding='utf8')


if __name__ == '__main__':
    main()
