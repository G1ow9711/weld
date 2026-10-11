"""One fixed FIT-only local tree trial; checks never tune or select its head."""
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
from weld_pipeline.horizontal_patch_tree import (VERSION, FEATURE_COUNT, OFFSETS_MM,
    equivalent_frame, patch_features, training_indices, predict_straight)
from weld_pipeline.typed_contact_model import infer_probabilities


def context(split, name):
    structure_path = Path(split['structure_roots'][name])/(name+'.npz')
    stats_path = Path('weld_pipeline/data/invariant_horizontal_context_v1')/(name+'.npz')
    with np.load(structure_path) as values:
        structural = values['structural'].copy()
    with np.load(stats_path) as values:
        mean, covariance = values['mean'].copy(), values['covariance'].copy()
    return structural, mean, covariance


def evaluate(bundle, curve_model, split, names, out, role):
    scores = {}
    for name in names:
        piece = load_piece(split['piece_roots'][name], verify=True)
        structure, mean, covariance = context(split, name)
        pred = predict_straight(bundle, piece['sequences'], structure, mean, covariance)
        arcs = [s for s in piece['sequences'] if s.get('curve_kind') == 'arc']
        if arcs:
            pred.extend(decode_intervals(arcs, infer_probabilities(curve_model, arcs), threshold=.5,
                weld_penalty=2., mask_weight=.25, event_gap_mm=0.))
        for i, item in enumerate(pred):
            item['id'] = 'horizontal_'+str(i+1)
        score = score_children(pred, [w for w in piece['truth'] if w['type'] == 'horizontal'])
        scores[name] = score
        (out/(role+'_'+name+'.json')).write_text(json.dumps({'predictions':pred, 'score':score}, indent=2), encoding='utf8')
        print(role, name, score['valid_children'], '/', score['truth_count'], 'FP', score['false_predictions'], flush=True)
    summary = aggregate(scores)
    summary.update(role=role, independent_acceptance=False, full_contract_pass=False)
    (out/(role+'_summary.json')).write_text(json.dumps(summary, indent=2), encoding='utf8')
    return summary


def main():
    torch.set_num_threads(3)
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    names = [n for n in split['train'] if '__' not in n]
    out = Path('weld_pipeline/runs/horizontal_patch_tree_seed136')
    out.mkdir(parents=True, exist_ok=True)
    if (out/'heads.joblib').exists():
        raise ValueError('Existing fixed tree trial')
    paths = [__file__, 'weld_pipeline/horizontal_patch_tree.py', 'weld_pipeline/event_gap_decoder.py',
        'weld_pipeline/constrained_decoder.py', 'weld_pipeline/sequence_augmentation.py',
        'weld_pipeline/contract_child_evaluation.py']
    config = {'version':VERSION, 'seed':136, 'trees':128, 'max_features':.25, 'max_depth':22,
        'min_samples_leaf':2, 'features':FEATURE_COUNT, 'offsets_mm':OFFSETS_MM.tolist(),
        'train_pieces':names, 'untrained_check_pieces':split['validation'], 'split_sha256':sha256(split_path),
        'source_hashes':{p:sha256(p) for p in paths},
        'data_hashes':{n:sha256(Path(split['piece_roots'][n])/'manifest.json') for n in names},
        'context_hashes':{n:{'structure':sha256(Path(split['structure_roots'][n])/(n+'.npz')),
            'statistics':sha256(Path('weld_pipeline/data/invariant_horizontal_context_v1')/(n+'.npz'))} for n in names},
        'source_balance':True, 'equivalent_frames_per_sequence':2, 'decoder':{'threshold':.5, 'weld_penalty':2., 'mask_weight':.25},
        'selection_policy':'One fixed tree fit; FIT comparison against128 before reading check labels. No check-based tuning.',
        'role':'FIT_NOT_INDEPENDENT', 'full_contract_pass':False}
    (out/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf8')
    archive = out/'source_snapshot'; archive.mkdir(exist_ok=True)
    for p in paths:
        shutil.copy2(p, archive/Path(p).name)
    started = time.perf_counter(); rng = np.random.default_rng(136)
    x, y, w, counts = [], [], [], {}
    for name in names:
        piece = load_piece(split['piece_roots'][name], verify=True)
        structural, mean, covariance = context(split, name)
        first = len(x)
        for si, (seq, target) in enumerate(zip(piece['sequences'], piece['targets'])):
            if seq['type'] != 'horizontal' or seq.get('curve_kind') == 'arc':
                continue
            for reverse in [False, True]:
                transformed, labels = equivalent_frame(seq, target, reverse=reverse, flip=bool(rng.integers(2)))
                indices = training_indices(labels, rng)
                if not len(indices):
                    continue
                x.append(patch_features(transformed, indices, structural[si], mean, covariance))
                y.append(labels[indices, :4].copy())
        size = sum(len(block) for block in x[first:])
        if size:
            w.extend([np.full(len(block), 1./size, float) for block in x[first:]])
        counts[name] = size
        print('TRAIN_SAMPLES', name, size, flush=True)
    xx, yy, weights = np.concatenate(x), np.concatenate(y), np.concatenate(w)
    if np.any(yy < 0) or not np.isfinite(xx).all():
        raise ValueError('Unknown or invalid values entered supervision')
    del x, y, w
    head = ExtraTreesRegressor(n_estimators=128, max_features=.25, max_depth=22,
        min_samples_leaf=2, n_jobs=4, random_state=136)
    print('PATCH_FIT_START', xx.shape, flush=True)
    head.fit(xx, yy, sample_weight=weights)
    bundle = {'heatmaps':head, 'config':config}
    joblib.dump(bundle, out/'heads.joblib')
    print('PATCH_WEIGHTS_WRITTEN', round(time.perf_counter()-started, 2), flush=True)
    del xx, yy, weights
    model = from_checkpoint(torch.load('weld_pipeline/runs/contract_full_fit_horizontal_seed128/best.pt', map_location='cpu')).cuda().eval()
    fit = evaluate(bundle, model, split, names, out, 'FIT')
    previous = json.loads(Path('weld_pipeline/runs/contract_full_fit_horizontal_seed128/complete.json').read_text('utf8'))['fit_summary']
    eligible = (fit['geometry_core_gates_pass'] and fit['valid_children'] >= previous['valid_children']
        and fit['false_predictions'] <= previous['false_predictions'])
    (out/'fit_adoption_decision.json').write_text(json.dumps({'eligible_on_fit':eligible, 'fit':fit,
        'prior':previous, 'check_used_for_decision':False}, indent=2), encoding='utf8')
    print('FIT_DECISION', eligible, fit['valid_children'], fit['false_predictions'], flush=True)
    check = evaluate(bundle, model, split, split['validation'], out, 'UNTRAINED_CHECK')
    report = {'sample_counts':counts, 'training_rows':sum(counts.values()), 'elapsed_seconds':time.perf_counter()-started,
        'fit_summary':fit, 'untrained_check_summary':check, 'eligible_on_fit':eligible,
        'head_sha256':sha256(out/'heads.joblib'), 'full_contract_pass':False}
    (out/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print('PATCH_TREE_COMPLETE', fit['valid_children'], check['valid_children'], 'FULL_CONTRACT_NOT_ACHIEVED', flush=True)


if __name__ == '__main__':
    main()
