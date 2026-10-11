"""Fit whole-parent reference utility and soft reference competition.

Unmatched annotated-reference cost is supervised here; it is not a physical
absence label. This pairwise approximation is not full higher-order set loss.
"""
from pathlib import Path
import pickle
import time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'server_rebuild/expansion_v1/reference_parent_selection_v1'
SETTINGS = dict(n_estimators=96, max_depth=18, min_samples_leaf=2,
                max_features=.75, bootstrap=False, random_state=1031, n_jobs=8)


def main():
    start = time.perf_counter(); inputs = BASE/'train_dataset'
    receipt = read(inputs/'receipt.json'); assert receipt['status'] == 'READY'
    assert digest(inputs/'protocol.json') == receipt['protocol_sha256']
    protocol = read(inputs/'protocol.json')
    for name, sha in protocol['modules'].items():
        assert digest(ROOT/'weld_pipeline'/f'{name}.py') == sha
    destination = BASE/'model_v1'; destination.mkdir(exist_ok=False)
    write(destination/'protocol.json', dict(settings=SETTINGS, keys=[r['key'] for r in receipt['cases']],
          input_receipt_sha256=digest(inputs/'receipt.json'), equal_case_weight=True,
          within_case_weights='Uniform natural candidate prevalence, no class rebalance',
          unary_target='(2 * best individually matched reference children - child_count) / child_count',
          pair_target='Exact original two-parent D3 loss versus independent TPs / (2 * min(child_counts))',
          solver='Positive-marginal greedy; exact identical child reuse excluded; soft learned pair costs',
          null_output=True, thresholds_tuned=False, reference_policy_not_physical_absence=True,
          full_higher_order_set_loss=False, frozen_original_run446=True, DEV_reads=0,
          source_sha256=digest(__file__), runtime_sha256=digest(ROOT/'weld_pipeline/server_reference_parent_selection.py')))
    xs, ys, ws, pxs, pys, pws = [], [], [], [], [], []
    summaries = []
    for row in receipt['cases']:
        for item in list(row['files'].values()) + [row['targets'], row['reference_table']]:
            assert digest(ROOT/item['path']) == item['sha256']
        x = np.load(ROOT/row['files']['features.npy']['path'])
        px = np.load(ROOT/row['files']['pair_features.npy']['path'])
        with np.load(ROOT/row['targets']['path']) as targets:
            y = targets['unary']; py = targets['penalties']
        assert x.shape == (len(y), 1458) and px.shape == (len(py), 29)
        assert np.isfinite(x).all() and np.isfinite(px).all()
        xs.append(x); ys.append(y); ws.append(np.full(len(y), 1./len(y)))
        if len(py):
            pxs.append(px); pys.append(py); pws.append(np.full(len(py), 1./len(py)))
        summaries.append(dict(key=row['key'], parents=len(y), pairs=len(py),
                              positive_unary=int((y > 0).sum()), positive_pair_cost=int((py > 0).sum())))
    print('FIT_UNARY', sum(len(x) for x in xs), flush=True)
    unary = ExtraTreesRegressor(**SETTINGS).fit(np.concatenate(xs), np.concatenate(ys),
                                               sample_weight=np.concatenate(ws))
    print('FIT_COMPETITION', sum(len(x) for x in pxs), flush=True)
    pair = ExtraTreesRegressor(**SETTINGS).fit(np.concatenate(pxs), np.concatenate(pys),
                                              sample_weight=np.concatenate(pws))
    model = dict(unary=unary, pair=pair, parent_columns=1458, pair_columns=29)
    with (destination/'model.pkl').open('wb') as stream:
        pickle.dump(model, stream, protocol=5)
    with (destination/'model.pkl').open('rb') as stream:
        loaded = pickle.load(stream)
    for name, data in [('unary', xs), ('pair', pxs)]:
        sample = np.concatenate([x[:min(32, len(x))] for x in data])
        # Parallel tree reduction can differ by roundoff; persisted model must
        # reproduce sequential predictions exactly, including reordered rows.
        model[name].set_params(n_jobs=1); loaded[name].set_params(n_jobs=1)
        assert np.array_equal(model[name].predict(sample), loaded[name].predict(sample))
        assert np.array_equal(loaded[name].predict(sample), loaded[name].predict(sample[::-1])[::-1])
    write(destination/'receipt.json', dict(status='TRAINED_NOT_PROMOTED', cases=summaries, fit_calls=2,
          input_receipt_sha256=digest(inputs/'receipt.json'), protocol_sha256=digest(destination/'protocol.json'),
          model_sha256=digest(destination/'model.pkl'), original_run446_unchanged=True,
          DEV_reads=0, seconds=time.perf_counter()-start))
    print('REFERENCE_PARENT_MODEL_READY', round(time.perf_counter()-start, 2), flush=True)


if __name__ == '__main__':
    main()
