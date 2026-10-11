"""Relation inheritance and complete-parent/null/arc controls, without labels."""
import math
from pathlib import Path

import numpy as np
import torch

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.contract_curve_geometry import CircularArc, sample_arc, verify_sampled_arc
from weld_pipeline.server_full_prediction_runtime import compact_segment, proposal_token
from weld_pipeline.server_predicted_parent_pool import (directed_state, child_metadata,
        assemble_parent, decoded_subset, FrozenRun446Priors)
from weld_pipeline.server_ordered_parent_set_model import OrderedParentSetModel, parent_batch, predict_set_scores
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set
from weld_pipeline.sjg_child_parent_supervision_1255 import singleton, state_geometry, link_features
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
from weld_pipeline.sjg_local_runtime_904 import write, digest
from weld_pipeline.world_set_runtime_285 import line_segment

ROOT = Path(__file__).resolve().parents[1]


def child(segment, profile):
    represented = compact_segment(segment)
    return dict(represented, weld_type='horizontal', predicted_side='UNKNOWN', reference_score=.6,
                profile=profile, branch='carrier', query_position_mm=0.,
                face_pairs=[[0, 1]], side_known=False, physical_sides=['UNKNOWN'],
                frame_origins_mm=represented['endpoints_mm'], raw_export_frame_axes=[np.eye(3).tolist()] * 2,
                endpoint_attributes={}, frame_business_semantics_certified=False)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(1031)
    arc = CircularArc([0., 0., 0.], 10., [0., 0., 1.], [1., 0., 0.], math.pi / 2)
    children = [child(line_segment([-10., 0., 0.], [10., 0., 0.]), 0),
                child(sample_arc(arc), 1),
                child(line_segment([0., 10.2, 0.], [0., 30., 0.]), 2)]
    reference_state = state_geometry(ParentRelations([singleton(dict(c, points=c['endpoints_mm'])) for c in children],
                                         [child_metadata(c) for c in children]))
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    expected = link_features(reference_state, edges * 2)
    actual = link_features(directed_state(children), edges)
    assert np.array_equal(actual, expected)
    priors = FrozenRun446Priors(ROOT)
    _, probability = priors.link_probabilities(children, edges)
    assert probability.shape == (3, 3) and np.isfinite(probability).all()
    joint = assemble_parent(children, [1, 2], [1])
    assert joint['gaps'] == ['SHARED']
    assert np.allclose(joint['segments'][0]['endpoints_mm'][1], joint['segments'][1]['endpoints_mm'][0], atol=1e-10, rtol=0)
    mixed = decoded_subset([joint], [0])
    verify_sampled_arc(mixed['parents'][0]['segments'][0])
    assert mixed['total_prediction_children'] == 2
    gap = assemble_parent(children, [1, 2], [2])
    assert gap['gaps'] == ['POSITIVE_GAP'] and not np.array_equal(gap['segments'][0]['endpoints_mm'][1], gap['segments'][1]['endpoints_mm'][0])
    full_circle = CircularArc([0., 0., 0.], 10., [0., 0., 1.], [1., 0., 0.], 2 * math.pi, True)
    circle = assemble_parent([child(sample_arc(full_circle), 3)], [0], [])
    assert circle['is_closed']
    empty = decoded_subset([joint, gap, circle], [])
    assert empty['total_prediction_children'] == 0 and empty['invalid_predictions'] == []
    # Complete tokens explicitly distinguish gap classes and parent closure.
    pool = dict(parents=[joint, gap])
    vectors = np.array([proposal_token(dict(c, points=c['endpoints_mm']), np.zeros(128), np.zeros(3), np.eye(3), c['reference_score']) for c in children])
    proposals = dict(child_tokens=vectors, query_xyz=np.zeros((3, 3)), query_bases=np.array([np.eye(3)] * 3))
    tokens, valid = parent_batch(pool, proposals, [0, 1], 'cpu')
    assert tokens.shape == (2, 2, 151) and valid.all()
    assert torch.equal(tokens[0, 1, 148:150], torch.tensor([1., 0.]))
    assert torch.equal(tokens[1, 1, 148:150], torch.tensor([0., 1.]))
    pair_x, duplicate = priors.duplicate_probabilities(pool['parents'], np.array([[0, 1]]))
    assert pair_x.shape == (1, 32) and np.isfinite(pair_x).all()
    model = OrderedParentSetModel().eval()
    unary, penalties = predict_set_scores(model, pool, proposals, np.array([[0, 1]]), pair_x, duplicate * 4)
    assert np.array_equal(unary, np.zeros(2))
    selected = greedy_nullable_set(unary, np.array([[0, 1]]), penalties, [p['input_children'] for p in pool['parents']])
    assert selected == []
    write(ROOT / 'server_rebuild/full_dataset_v1/full_prediction_runtime_validation.json', dict(
          status='PASS_LABEL_FREE_RELATION_AND_NULL_ARC_PARENT_CONTROLS_NOT_MODEL_ACCURACY',
          run446_original_link_feature_values_exact=True, duplicate_features_real_32_columns=True,
          best_prior_sha256=priors.checkpoint_sha256, mixed_arc_shared_cut_analytic_reconstruction=True,
          positive_gap_retained=True, single_closed_circle_retained=True,
          complete_parent_tokens_keep_order_gaps_and_closure=True,
          empty_selection_has_no_rejected_candidate_false_predictions=True,
          initialized_nullable_selector_returns_empty=True, GT_label_reads=0, model_fit_calls=0,
          source_sha256=digest(__file__), sources={name: digest(ROOT / 'weld_pipeline' / name)
              for name in ['server_full_prediction_runtime.py', 'server_predicted_parent_pool.py', 'server_ordered_parent_set_model.py']}))
    print('FULL_PREDICTION_RUNTIME_CONTROLS_PASS', flush=True)


if __name__ == '__main__':
    main()
