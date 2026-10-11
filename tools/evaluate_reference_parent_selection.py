"""Freeze nullable whole-parent predictions, then score against original D3."""
import argparse
import pickle
import time
from pathlib import Path
import numpy as np
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set, set_model_utility
from weld_pipeline.server_bound_parent_pool import load_parent_pool
from weld_pipeline.server_array_cached_d3_evaluation import score_native_parents_cached
from weld_pipeline.server_offline_parent_set_reference import evaluate_reference_subset
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'server_rebuild/expansion_v1/reference_parent_selection_v1'


def native_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: native_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(scope, residual_folder=None):
    dataset = BASE/(scope+'_dataset'); receipt = read(dataset/'receipt.json')
    modelpath = BASE/'model_v1/model.pkl'; trained = read(modelpath.parent/'receipt.json')
    assert digest(modelpath) == trained['model_sha256']
    assert trained['input_receipt_sha256'] == digest(BASE/'train_dataset/receipt.json')
    protocol = read(modelpath.parent/'protocol.json')
    assert digest(ROOT/'weld_pipeline/server_reference_parent_selection.py') == protocol['runtime_sha256']
    with modelpath.open('rb') as stream:
        model = pickle.load(stream)
    # Deterministic tree summation at the decision boundary.
    model['unary'].set_params(n_jobs=1); model['pair'].set_params(n_jobs=1)
    residual_model = None
    residual_sha = None
    if residual_folder is not None:
        import torch
        from weld_pipeline.server_structured_parent_residual import StructuredParentResidual, predict_scores
        residual_folder = Path(residual_folder)
        rr = read(residual_folder/'receipt.json'); rp = read(residual_folder/'protocol.json')
        assert rr['status'] == 'TRAINED_NOT_PROMOTED'
        assert digest(residual_folder/'last.pt') == rr['checkpoint_sha256']
        assert digest(residual_folder/'protocol.json') == rr['protocol_sha256']
        assert rp['base_model_sha256'] == digest(modelpath)
        for name, sha in rp['modules'].items():
            assert digest(ROOT/'weld_pipeline'/f'{name}.py') == sha
        checkpoint = torch.load(residual_folder/'last.pt', map_location='cpu', weights_only=False)
        assert checkpoint['protocol_sha256'] == rr['protocol_sha256']
        state = checkpoint['state_dict']
        residual_model = StructuredParentResidual(*[state[k] for k in ['parent_mean', 'parent_std', 'pair_mean', 'pair_std']])
        residual_model.load_state_dict(state); residual_model.cuda().eval()
        residual_sha = rr['checkpoint_sha256']
    output = (BASE if residual_folder is None else residual_folder)/(scope+'_outputs'); output.mkdir(exist_ok=False)
    reports = []
    for row in receipt['cases']:
        start = time.perf_counter(); key = row['key']; out = output/key; out.mkdir()
        for item in list(row['files'].values())+[row['geometry']]:
            assert digest(ROOT/item['path']) == item['sha256']
        geometry = read(ROOT/row['geometry']['path'])
        pool = geometry if row['geometry']['kind'] == 'native' else load_parent_pool(ROOT, geometry)[0]
        x = np.load(ROOT/row['files']['features.npy']['path'])
        px = np.load(ROOT/row['files']['pair_features.npy']['path'])
        pairs = np.load(ROOT/row['files']['pairs.npy']['path'])
        counts = np.load(ROOT/row['files']['child_counts.npy']['path'])
        pc = read(ROOT/row['files']['parent_children.json']['path'])
        if residual_model is None:
            unary = model['unary'].predict(x)*counts if len(x) else np.empty(0)
            penalty = np.clip(model['pair'].predict(px), 0., 1.) * 2*np.minimum(counts[pairs[:, 0]], counts[pairs[:, 1]]) if len(px) else np.empty(0)
        else:
            baseout = BASE/(scope+'_outputs')/key
            basefreeze = read(baseout/'prediction_freeze.json')
            assert digest(baseout/'model_scores.npz') == basefreeze['scores_sha256']
            assert basefreeze['model_sha256'] == digest(modelpath)
            with np.load(baseout/'model_scores.npz') as z:
                baseu = z['unary']; basep = z['penalty']
            unary, penalty = predict_scores(residual_model, x, px, counts, pairs, baseu, basep)
        np.savez_compressed(out/'model_scores.npz', unary=unary, penalty=penalty)
        chosen = {}
        # Same unary weights/whole parents for both arms; isolate soft competition.
        arms = [('unary_only', np.zeros(len(pairs))), ('soft_set', penalty)] if residual_model is None else [('soft_set', penalty)]
        for arm, cost in arms:
            selected = [int(i) for i in greedy_nullable_set(unary, pairs, cost, pc)]
            prediction = dict(parents=[pool['parents'][i] for i in selected], invalid_predictions=pool['invalid_predictions'],
                              total_prediction_children=int(counts[selected].sum()) + sum(v['counted_false_children'] for v in pool['invalid_predictions']))
            write(out/(arm+'_prediction.json'), native_json(prediction))
            chosen[arm] = dict(selected_parent_indices=selected, selected_parents=len(selected),
                               predicted_children=prediction['total_prediction_children'],
                               model_utility=set_model_utility(unary, pairs, cost, selected),
                               prediction_sha256=digest(out/(arm+'_prediction.json')))
        write(out/'prediction_freeze.json', dict(key=key, arms=chosen, model_sha256=digest(modelpath),
              input_receipt_sha256=digest(dataset/key/'receipt.json'), scores_sha256=digest(out/'model_scores.npz'),
              GT_reads=0, STEP_reads=0, same_whole_parent_geometry=True, runtime_sha256=protocol['runtime_sha256'],
              residual_checkpoint_sha256=residual_sha))
        print('FROZEN', scope, key, {a:r['predicted_children'] for a,r in chosen.items()}, flush=True)
        labels = ROOT/'server_rebuild'/('expansion_v1/labels' if key.startswith('new_') else 'labels_v1')/key
        lr = read(labels/'receipt.json'); assert lr['role'] in ['TRAIN', 'EXPANSION_TRAIN']
        targetpath = labels/'targets.json'; assert digest(targetpath) == lr['targets_sha256']
        target = read(targetpath); scores = {}
        for arm, frozen in chosen.items():
            path = out/(arm+'_prediction.json'); assert digest(path) == frozen['prediction_sha256']
            scores[arm] = score_native_parents_cached(read(path), target, [False]*len(target['parents']))
            if scope == 'train':
                item = row['reference_table']; assert digest(ROOT/item['path']) == item['sha256']
                with np.load(ROOT/item['path']) as z:
                    table = {n:z[n] for n in z.files}
                exact = evaluate_reference_subset(table, frozen['selected_parent_indices'])
                assert all(scores[arm]['categories']['overall'][k] == exact[k] for k in ['TP', 'FP', 'predicted'])
        result = dict(key=key, scope=scope, scores=scores, input_parents=len(pool['parents']),
                      prediction_freeze_sha256=digest(out/'prediction_freeze.json'), target_sha256=digest(targetpath),
                      independent_acceptance=False, protected_DEV_reads=0, seconds=time.perf_counter()-start)
        write(out/'evaluation.json', result)
        reports.append(dict(key=key, arms={arm:score['categories'] for arm,score in scores.items()}, seconds=result['seconds']))
        print('SCORED', key, {a:s['categories']['overall'] for a,s in scores.items()}, flush=True)
    totals = {}
    for row in reports:
        for arm, categories in row['arms'].items():
            total = totals.setdefault(arm, {})
            for category, values in categories.items():
                entry = total.setdefault(category, {k:0 for k in values})
                for name, value in values.items():
                    entry[name] += value
    write(output/'summary.json', dict(status='EVALUATED_NOT_PROMOTED', cases=reports, totals=totals,
          scope=scope, model_sha256=digest(modelpath), residual_checkpoint_sha256=residual_sha, dataset_receipt_sha256=digest(dataset/'receipt.json'),
          not_a_raw_XYZ_latency_benchmark=True, independent_acceptance=False, original_run446_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', choices=['train', 'evaluation'], required=True)
    parser.add_argument('--residual-folder', type=Path)
    arguments = parser.parse_args()
    main(arguments.scope, arguments.residual_folder)
