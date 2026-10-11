"""Frozen run376 plus a geometric compatibility log-odds correction.

All labels are TRAIN-only complete-parent compatibility. Inference reads only
the frozen geometric features, model files, and the current selected set.
"""
import argparse
import pickle
import shutil
import time
import traceback

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, check_resources
from weld_pipeline.sjg_incompatibility_head_1216 import IncompatibilityScorer
from weld_pipeline.sjg_incompatibility_worker_1217 import all_data
from weld_pipeline.sjg_incompatibility_continuation_1219 import control
from weld_pipeline.sjg_duplicate_head_1209 import DuplicateRelations
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS

SCHEMA = 'frozen376-U160-geometric-compatibility-logodds-v1'


def deduction(probability):
    q = np.asarray(probability, dtype=np.float64)
    assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1))
    p = np.clip(q, 1e-6, 1. - 1e-6)
    return np.maximum(0., np.log(p) - np.log1p(-p))


def probability(model, u):
    assert u.ndim == 2 and u.shape[1] == 160 and np.isfinite(u).all()
    assert model.classes_.tolist() == [0, 1]
    return model.predict_proba(u)[:, 1]


def controls(dest):
    # Existing full-state restore and one nonempty-prefix bitwise comparison.
    control(dest)
    d = deduction(np.array([0., .25, .5, .75, 1.]))
    assert np.array_equal(d[:3], np.zeros(3)) and np.isfinite(d).all()
    assert abs(d[3] - np.log(3.)) < 1e-14 and d[4] > d[3]
    write(dest/'precision_control.json', dict(status='FROZEN376_AND_DEDUCTION_CONTROL_PASSED',
        correction_zero_exact_for_probability_at_or_below_half=True,
        no_candidate_deletion=True, model_updates=0))


def train(dest, limit):
    start = time.perf_counter()
    assert read(dest/'precision_control.json')['status'] == 'FROZEN376_AND_DEDUCTION_CONTROL_PASSED'
    old = OUT/'run376'
    frozen_files = ('selector.npz', 'base_selector.npz', 'frozen_residual.npz',
                    'frozen_duplicate.npz', 'selector_normalization.npz')
    bindings = {name: digest(old/name) for name in frozen_files}
    for name in frozen_files:
        shutil.copy2(old/name, dest/name)
    with np.load(dest/'selector_normalization.npz') as z:
        data = all_data(dest, z['mean'], z['scale'])
    xs, ys, weights = [], [], []
    for key in KEYS:
        u, y = data[key]
        known = y >= 0
        classes = np.unique(y[known])
        w = np.zeros(len(y))
        for value in classes:
            mask = y == value
            w[mask] = 1. / (len(KEYS) * len(classes) * int(mask.sum()))
        xs.append(u[known]); ys.append(y[known]); weights.append(w[known])
    x = np.concatenate(xs); y = np.concatenate(ys); w = np.concatenate(weights)
    assert x.shape == (3899, 160) and np.isfinite(x).all()
    assert np.isclose(w.sum(), 1.) and set(y.tolist()) == {0, 1}
    # Fixed before this trial; no test-directed threshold or tree selection.
    model = ExtraTreesClassifier(n_estimators=16, max_depth=16,
        min_samples_leaf=2, max_features=.75, bootstrap=False,
        random_state=1031, n_jobs=2, warm_start=True)
    trace = []
    for batch in range(1, 9):
        check_resources()
        if time.perf_counter() - start >= limit:
            raise TimeoutError('Precision forest training budget exhausted')
        model.set_params(n_estimators=batch * 16)
        model.fit(x, y, sample_weight=w)
        assert len(model.estimators_) == batch * 16
        checkpoint = dict(schema=SCHEMA, model=model, frozen_files=bindings,
            trees=len(model.estimators_), registered_trees=128,
            input_channels=160, TRAIN_sources=list(KEYS), DEV_reads=0)
        atomic_binary_write(dest/'precision_resume.pkl', lambda stream: pickle.dump(checkpoint, stream, protocol=5))
        trace.append(dict(batch=batch, trees=len(model.estimators_), seconds=time.perf_counter()-start))
        write(dest/'precision_progress.json', dict(status='PARTIAL_TREES_NOT_FINAL', trace=trace,
            checkpoint_sha256=digest(dest/'precision_resume.pkl')))
        print('PRECISION_TREE_BATCH', batch, '/8', 'TREES', len(model.estimators_), '/128', flush=True)
    # Canonical serial inference makes accumulation and reload comparisons exact.
    model.set_params(n_jobs=1, warm_start=False)
    payload = dict(schema=SCHEMA, model=model, frozen_files=bindings, trees=128,
        input_channels=160, TRAIN_sources=list(KEYS), DEV_reads=0)
    atomic_binary_write(dest/'precision_model.pkl', lambda stream: pickle.dump(payload, stream, protocol=5))
    with (dest/'precision_model.pkl').open('rb') as stream:
        loaded = pickle.load(stream)
    p = probability(model, x); reloaded = probability(loaded['model'], x)
    assert np.array_equal(p, reloaded)
    assert np.array_equal(probability(loaded['model'], x[::-1])[::-1], p)
    groups = []
    for key, (u, labels) in data.items():
        q = probability(loaded['model'], u); d = deduction(q)
        groups.append(dict(source=key, compatible=int(np.sum(labels==0)), incompatible=int(np.sum(labels==1)),
            compatible_positive_deduction=int(np.sum((labels==0)&(d>0))),
            incompatible_zero_deduction=int(np.sum((labels==1)&(d==0))),
            maximum_compatible_deduction=float(d[labels==0].max()) if np.any(labels==0) else None,
            not_extraction_metric=True))
    write(dest/'precision_classification.json', dict(cases=groups, reload_bitwise_equal=True,
        row_permutation_equivariant=True, probability_calibration_not_established=True))
    for name, sha in bindings.items():
        assert digest(old/name) == digest(dest/name) == sha
    source = read(old/'selector_training.json')
    write(dest/'selector_training.json', dict(status='FROZEN376_WITH_FINAL_PRECISION_FOREST',
        updates=112, new_updates=0, new_precision_trees=128, forest_fit_batches=8,
        base_Adam_step=560, frozen_residual_Adam_step=112, frozen_duplicate_Adam_step=168,
        all_previous_parameters_and_moments_frozen=True, source_run='run376',
        selector_sha256=bindings['selector.npz'], normalization_sha256=bindings['selector_normalization.npz'],
        frozen_files=bindings, source_checkpoint_sha256=digest(old/'selector_resume_checkpoint.npz'),
        precision_model_sha256=digest(dest/'precision_model.pkl'), trace=trace,
        bindings=source['bindings'], seconds=time.perf_counter()-start,
        DEV_reads=0, GT_inference_reads=0, only_final_weights_evaluated=True,
        fitted_probability_is_not_calibrated=True, geometry_labels_only=True, full_contract_pass=False))
    print('PRECISION_FOREST_TRAINED', groups, flush=True)


def select(dest, key, limit):
    start = time.perf_counter(); folder = dest/f'test_{key}'
    training = read(dest/'selector_training.json')
    for name, sha in training['frozen_files'].items():
        assert digest(dest/name) == sha
    assert digest(dest/'precision_model.pkl') == training['precision_model_sha256']
    with (dest/'precision_model.pkl').open('rb') as stream:
        correction = pickle.load(stream)
    assert correction['schema'] == SCHEMA and correction['frozen_files'] == training['frozen_files']
    model = IncompatibilityScorer.load(dest/'selector.npz', dest/'frozen_duplicate.npz',
        dest/'frozen_residual.npz', dest/'base_selector.npz')
    with np.load(dest/'selector_normalization.npz') as z:
        u = (np.load(folder/'unary.npy') - z['mean']) / z['scale']
    logits, _ = model.logits(u)
    q = probability(correction['model'], u); penalty = deduction(q)
    parents, points = load_parents(folder)
    context = DuplicateRelations(parents, read(folder/'parent_metadata.json'), model.frozen)
    for step in range(len(parents)+1):
        check_resources()
        if time.perf_counter()-start > min(120., limit):
            raise TimeoutError('Selection development time exhausted')
        scores = model.forward(u, context.features(), context.selected, logits)
        scores[:-1] -= penalty
        chosen = model.choose_next(scores, context.selected)
        if chosen is None:
            break
        context.add(chosen)
    else:
        raise RuntimeError('Missing STOP after all candidates')
    indices = list(context.selected)
    del context, parents
    points._mmap.close()
    original = read(folder/'decoded.json'); result = dict(original)
    result['parents'] = [original['parents'][i] for i in indices]
    result['total_prediction_children'] = sum(p['child_count'] for p in result['parents']) + sum(v['counted_false_children'] for v in original['invalid_predictions'])
    result.update(runtime_model='FROZEN376_WITH_GEOMETRIC_PRECISION_RESIDUAL', GT_reads=0)
    write(folder/'final_decoded.json', result)
    atomic_binary_write(folder/'precision_scores.npz', lambda stream: np.savez(stream, probability=q, deduction=penalty))
    write(folder/'final_prediction.json', dict(status='LABEL_FREE_PRECISION_PREDICTION_FROZEN', source=key,
        selected_candidate_indices=indices, candidate_count=len(original['parents']),
        decoded_sha256=digest(folder/'final_decoded.json'), selector_sha256=training['selector_sha256'],
        precision_model_sha256=training['precision_model_sha256'], GT_reads=0, STEP_reads=0,
        invalid_ledger_unchanged=True, inference_classifier_evaluations=1, seconds=time.perf_counter()-start))


def aggregate(dest):
    results = [read(dest/f'test_{k}/evaluation.json') for k in ('21','BM79A_native_local','10')]
    requirements = {'overall': (213,203,10), 'Hline': (84,80,4), 'Vline': (126,120,6), 'HArc': (3,3,0)}
    groups = {}; passed = True
    for name, (g, tp_min, fp_max) in requirements.items():
        total = {field: sum(r['score']['categories'][name][field] for r in results) for field in ('G','TP','FP','predicted')}
        assert total['G'] == g
        total.update(FN=g-total['TP'], TP_min=tp_min, FP_max=fp_max,
            extraction_correct_rate_percent=100*total['TP']/g, FP_over_G_percent=100*total['FP']/g,
            passed=total['TP']>=tp_min and total['FP']<=fp_max)
        groups[name] = total; passed &= total['passed']
    maxima = [r['report']['matched_maximum_geometry_error_mm'] for r in results if r['report']['matched_maximum_geometry_error_mm'] is not None]
    passed &= bool(maxima) and max(maxima)<=3.
    report = dict(status='THREE_CASE_GEOMETRY_CONTINUATION_EVALUATED', groups=groups,
        geometric_continuation_pass=bool(passed), reports=[r['report'] for r in results],
        matched_maximum_geometry_error_mm=max(maxima) if maxima else None,
        fully_matched_parents=sum(r['report']['fully_matched_parents'] for r in results), original_parents=126,
        baseline_run='run376', baseline_TP=206, baseline_FP=13,
        regression_TP_delta=groups['overall']['TP']-206, FP_delta=groups['overall']['FP']-13,
        historical_run367_TP=198, historical_run367_FP=41,
        best_baseline_count_preserved=groups['overall']['TP']>=206,
        geometry_gate_is_not_full_contract=True, full_contract_pass=False)
    write(dest/'three_case_final_summary.json', report)
    print('FINAL_THREE_CASE_METRICS', report, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', required=True)
    ap.add_argument('--stage', required=True); ap.add_argument('--source', default='')
    ap.add_argument('--limit-seconds', type=float, required=True)
    args = ap.parse_args(); dest = OUT/args.run; start = time.perf_counter()
    try:
        if args.stage == 'precision_control': controls(dest)
        elif args.stage == 'precision_train': train(dest, args.limit_seconds)
        elif args.stage == 'select': select(dest, args.source, args.limit_seconds)
        elif args.stage == 'aggregate': aggregate(dest)
        else: raise ValueError(args.stage)
        write(dest/f'stage_{args.stage}_{args.source}_complete.json', dict(status='STAGE_COMPLETE',
            seconds=time.perf_counter()-start, worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{args.stage}_{args.source}_failure.json', dict(error=traceback.format_exc(), seconds=time.perf_counter()-start))
        raise


if __name__ == '__main__':
    main()
