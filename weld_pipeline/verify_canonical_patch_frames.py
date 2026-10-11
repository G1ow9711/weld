"""Full-feature equality and endpoint orientation under equivalent frames."""
import json
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.horizontal_patch_tree import equivalent_frame, patch_features
from weld_pipeline.train_horizontal_patch_tree import context


def main():
    split = json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    rows = []
    for name in ['224-GR24A1', '207-FR79M', '203-TT1A']:
        piece = load_piece(split['piece_roots'][name], verify=True)
        structural, mean, covariance = context(split, name)
        candidates = [i for i, sequence in enumerate(piece['sequences'])
                      if sequence['type'] == 'horizontal' and sequence.get('curve_kind') != 'arc']
        for index in candidates[::max(1, len(candidates)//8)]:
            original, target = piece['sequences'][index], piece['targets'][index]
            seq, labels, _ = canonical_frame(original, target)
            indices = np.unique(np.linspace(0, len(labels)-1, 19).astype(int))
            expected = patch_features(seq, indices, structural[index], mean, covariance)
            for reverse in (False, True):
                for flip in (False, True):
                    changed, changed_labels = equivalent_frame(original, target, reverse=reverse, flip=flip)
                    canonical, got, transformation = canonical_frame(changed, changed_labels)
                    actual = patch_features(canonical, indices, structural[index], mean, covariance)
                    if not np.array_equal(actual, expected) or not np.array_equal(got, labels):
                        raise ValueError('Equivalent canonical features/labels differ')
                    restored = restore_probability_order(got, transformation)
                    if not np.array_equal(restored, changed_labels):
                        raise ValueError('Start/end probability order not restored')
                    rows.append(dict(piece=name, sequence=index, reverse=reverse, flip=flip, passed=True))
    out = Path('weld_pipeline/reports/canonical_patch_frames_20261001.json')
    if out.exists():
        raise ValueError('Physical equality receipt exists')
    out.write_text(json.dumps(dict(cases=rows, all_passed=True, full_521_feature_equality=True,
        canonical_source_sha256=sha256('weld_pipeline/canonical_horizontal_patch_heatmaps.py'),
        verifier_sha256=sha256(__file__), check_truth_read=False, full_contract_pass=False), indent=2), encoding='utf8')
    print('CANONICAL_FRAMES_EXACT', len(rows), 'FEATURES_521_AND_ENDPOINT_ORDER', flush=True)


if __name__ == '__main__':
    main()
