"""Physical-origin and unknown-supervision checks on an actual candidate."""
import json
from pathlib import Path

import numpy as np

from weld_pipeline.curved_cache import load_piece
from weld_pipeline.horizontal_patch_tree import equivalent_frame, patch_features, training_indices
from weld_pipeline.train_horizontal_patch_tree import context


def main():
    split = json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    name = split['train'][0]
    piece = load_piece(split['piece_roots'][name], verify=True)
    structural, mean, covariance = context(split, name)
    i = next(i for i, (s, y) in enumerate(zip(piece['sequences'], piece['targets']))
        if s['type']=='horizontal' and (y[:,0]>.5).any())
    sequence, target = piece['sequences'][i], piece['targets'][i]
    indices = np.arange(0, len(sequence['features']), max(1, len(sequence['features'])//32))
    expected = patch_features(sequence, indices, structural[i], mean, covariance)
    shift = np.array([1234.5, -743.25, 87.])
    feature = np.array(sequence['features'], copy=True)
    feature[:,67:70] += shift/10000.
    feature[:,49:52] += shift/1000.
    translated = {**sequence, 'xyz':np.asarray(sequence['xyz'],float)+shift, 'features':feature,
        'name':'irrelevant_probe_name', 'candidate_id':'irrelevant_probe_id'}
    got = patch_features(translated, indices, structural[i], mean+shift, covariance)
    np.testing.assert_allclose(got, expected, atol=2e-7, rtol=2e-7)
    np.testing.assert_array_equal(sequence['features'], piece['sequences'][i]['features'])
    print('PASS physical translation and metadata exclusion')
    transformed, reversed_target = equivalent_frame(sequence, target, reverse=True)
    np.testing.assert_array_equal(transformed['xyz'], sequence['xyz'][::-1])
    np.testing.assert_array_equal(reversed_target[:,1], target[::-1,2])
    np.testing.assert_array_equal(reversed_target[:,2], target[::-1,1])
    unknown = np.array(target, copy=True)
    unknown[::3] = -1.
    samples = training_indices(unknown, np.random.default_rng(136))
    assert not np.any(samples % 3 == 0)
    assert len(training_indices(np.full_like(target,-1.), np.random.default_rng(136))) == 0
    print('PASS directed endpoint reversal and unknown supervision exclusion')


if __name__ == '__main__':
    main()
