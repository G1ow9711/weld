"""Finite-input equivalence only; stored component integrity hashes stay exact."""
import copy
import numpy as np
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_manifest


def finite_manifest(model):
    import sklearn
    assert sklearn.__version__ == '1.5.2'
    assert type(model).__name__ in ('ExtraTreesClassifier', 'ExtraTreesRegressor')
    assert model.n_estimators==64 and model.max_depth==32 and model.min_samples_leaf==1
    assert model.max_features=='sqrt' and not model.bootstrap and model.n_jobs==1 and model.random_state==1031
    expected=('left_child','right_child','feature','threshold','impurity','n_node_samples','weighted_n_node_samples','missing_go_to_left')
    for tree in model.estimators_:
        assert tree.splitter == 'random' and not tree._get_tags()['allow_nan']
        assert tree.tree_.__getstate__()['nodes'].dtype.names==expected
    manifest = forest_state_manifest(model)
    for tree in manifest['estimators']:
        # sklearn 1.5.2 node_split_random does not initialize this field.
        # It is unreachable for finite inputs, and NaN is rejected by this API.
        del tree['tree_state']['nodes']['fields']['missing_go_to_left']
    return manifest


def verify_finite_layout(a, b, data):
    assert data.dtype==np.float32 and np.isfinite(data).all()
    ma, mb = finite_manifest(a), finite_manifest(b)
    assert ma == mb, 'Finite predictive tree state differs between C/F layouts'
    assert np.array_equal(a.predict(data), b.predict(data))
    assert np.array_equal(a.apply(data), b.apply(data))
    if hasattr(a, 'predict_proba'):
        assert np.array_equal(a.predict_proba(data), b.predict_proba(data))
    invalid = data[:1].copy(); invalid[0, 0] = np.nan
    for model in (a, b):
        try:
            model.predict(invalid)
        except ValueError as exc:
            assert 'NaN' in str(exc)
        else:
            raise AssertionError('NaN accepted: missing-value branch cannot be ignored')
    # Mutate only a disposable control copy, never either fitted control model.
    control=copy.deepcopy(b);tree = control.estimators_[0]
    nodes = tree.tree_.__getstate__()['nodes']
    saved = nodes['missing_go_to_left'].copy()
    try:
        nodes['missing_go_to_left'] ^= np.uint8(1)
        assert finite_manifest(control) == mb
        assert np.array_equal(a.predict(data), control.predict(data))
    finally:
        nodes['missing_go_to_left'][:] = saved
    controls = []
    for field, array in [('threshold', tree.tree_.threshold), ('leaf_value', tree.tree_.value),('structure',tree.tree_.children_left)]:
        index = int(np.flatnonzero(tree.tree_.children_left < 0)[0]) if field == 'leaf_value' else 0
        old = np.array(array[index], copy=True)
        try:
            array[index] = old + 1
            assert finite_manifest(control) != mb, 'Comparator ignored ' + field
        finally:
            array[index] = old
        controls.append(field)
    seed = tree.random_state
    try:
        tree.random_state = seed + 1
        assert finite_manifest(control) != mb, 'Comparator ignored random seed'
    finally:
        tree.random_state = seed
    assert finite_manifest(control) == mb and finite_manifest(b)==mb
    return dict(finite_predictive_fields_exact=True, predictions_exact=True, leaf_indices_exact=True,nan_rejected=True,
                excluded_field='missing_go_to_left', exclusion_scope='sklearn1.5.2 random ExtraTrees finite inputs only',
                mutation_controls=controls + ['random_state'], stored_component_hash_unchanged=True)
