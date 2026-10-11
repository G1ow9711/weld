"""Post-test stage attribution, with all original denominators preserved."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from weld_pipeline.sjg_geometry_776 import read, write, digest
from weld_pipeline.sjg_prepare_776 import OUT, TRAIN, VALID
from weld_pipeline.sjg_interval_782 import training_labels


def main():
    out = OUT/'run08'; rows = []
    for si in TRAIN+VALID:
        file = out/f'source_{si:02d}_candidates.npz'
        binding = next(r for r in read(out/'candidate_freeze.json')['outputs'] if r['source'] == si)
        assert digest(file) == binding['sha256']
        with np.load(file, allow_pickle=False) as data: candidates = data['rows']
        selection = read(out/f'source_{si:02d}_selection.json')['selected_candidate_ids']
        truth = read(OUT/f'source_{si:02d}/coverage_v2.json')['assignments']
        _, support = training_labels(si, candidates)
        selected = candidates[selection]; correct = 0
        for ci in np.unique(selected[:, 0]):
            indices = np.flatnonzero(selected[:, 0] == ci)
            targets = [t for t in truth if t['contact'] == ci]
            if not targets: continue
            target_q = np.array([[t['lo'], t['hi']] for t in targets])
            error = np.abs(selected[indices, None, 3:5]-target_q[None, :, :]).max(2)
            cost = np.where(error <= 3, error, 1e6)
            cost = np.c_[cost, np.full((len(indices), len(indices)), 1e4)]
            ii, jj = linear_sum_assignment(cost)
            correct += int(sum(j < len(targets) and error[i, j] <= 3 for i, j in zip(ii, jj)))
        strict = read(out/f'source_{si:02d}_contract_score.json')['categories']['overall']
        row = dict(source=si, split='TRAIN' if si in TRAIN else 'DEVELOPMENT_HOLDOUT',
                   candidate_supported=support['candidate_supported'], G=support['original_G'],
                   emitted=len(selection), same_contact_correct_before_parent=correct,
                   strict_parent_TP=strict['TP'])
        rows.append(row); print(row, flush=True)
    write(out/'stage_error_audit.json', dict(rows=rows,
          holdout_used_only_for_posthoc_audit=True, no_new_optimizer_updates=True,
          warning='Candidate support is a representation diagnostic, not neural accuracy'))


if __name__ == '__main__': main()
