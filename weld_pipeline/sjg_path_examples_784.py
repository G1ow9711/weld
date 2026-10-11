"""Parent labels for frozen free predictions, retaining false children as negatives."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from weld_pipeline.sjg_geometry_776 import read, digest
from weld_pipeline.sjg_prepare_776 import ROOT, OUT
from weld_pipeline.sjg_path_781 import free_children, candidate_features, train_examples as clean_examples
from weld_pipeline.sjg_interval_782 import predictions_from_rows


def frozen_prediction(si, contacts):
    folder = OUT/'run08'
    binding = next(r for r in read(folder/'candidate_freeze.json')['outputs'] if r['source'] == si)
    file = folder/binding['path']; assert digest(file) == binding['sha256']
    with np.load(file, allow_pickle=False) as data: rows = data['rows']
    selection = read(folder/f'source_{si:02d}_selection.json')['selected_candidate_ids']
    return predictions_from_rows(rows, selection, contacts)


def free_examples(si, contacts, h, inventory):
    predictions = frozen_prediction(si, contacts)
    children = free_children(predictions, contacts)
    # Features are completely determined before label access.
    pairs, x = candidate_features(children, contacts, h)
    coverage = read(OUT/f'source_{si:02d}/coverage_v2.json')['assignments']
    target_path = ROOT/inventory[si]['target']['target_path']
    assert digest(target_path) == inventory[si]['target']['target_sha256']
    target = read(target_path)
    labels = {}
    for ci in {c['contact'] for c in children}:
        ids = [i for i, c in enumerate(children) if c['contact'] == ci]
        truth = [a for a in coverage if a['contact'] == ci]
        if not truth: continue
        actual = np.asarray([[children[i]['lo'], children[i]['hi']] for i in ids])
        expected = np.asarray([[a['lo'], a['hi']] for a in truth])
        error = np.abs(actual[:, None, :]-expected[None, :, :]).max(2)
        cost = np.c_[np.where(error <= 3., error, 1e6), np.full((len(ids), len(ids)), 1e4)]
        ii, jj = linear_sum_assignment(cost)
        for i, j in zip(ii, jj):
            if j < len(truth) and error[i, j] <= 3.: labels[ids[i]] = truth[j]
    y = np.zeros((len(pairs), 2), np.float32)
    for k, (i, j) in enumerate(pairs):
        a, b = labels.get(int(i)), labels.get(int(j))
        if a is None or b is None: continue
        linked = a['parent'] == b['parent'] and abs(a['child']-b['child']) == 1
        shared = linked and target['parents'][a['parent']]['gaps'][min(a['child'], b['child'])] == 'SHARED'
        y[k] = [float(linked), float(shared)]
    return x, y


def train_examples(si, contacts, h, inventory):
    clean_x, clean_y = clean_examples(si, contacts, h, inventory)
    free_x, free_y = free_examples(si, contacts, h, inventory)
    return np.concatenate([clean_x, free_x]), np.concatenate([clean_y, free_y])
