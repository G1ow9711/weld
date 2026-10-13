"""Geometry, metadata and learned-runtime controls for the new parent module.

Includes every curve from the original five frozen TRAIN outputs and explicit
line/arc common-event controls. No complete-data fitting or quality claim.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import math
import traceback

import numpy as np
import torch

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.contract_curve_geometry import CircularArc, sample_arc
from weld_pipeline.server_full_prediction_runtime import compact_segment, sampled_segment
from weld_pipeline.server_full_weld_csv_export import export_decoded, verify_roundtrip
from weld_pipeline.server_predicted_parent_pool import FrozenRun446Priors, assemble_parent
from weld_pipeline.server_supported_directed_parent_pool import (
    DirectedLinkModel, directed_children, assemble_fixed_parent,
    build_supported_parent_pool, geometry_fingerprint)
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, digest, read, write
from tools.export_full_dataset_split_view import local_name
from tools.record_training_experiment import git_head


SOURCES = ['tools/check_supported_directed_parent_pool.py',
    'weld_pipeline/server_supported_directed_parent_pool.py',
    'weld_pipeline/server_predicted_parent_pool.py', 'weld_pipeline/server_full_prediction_runtime.py',
    'weld_pipeline/contract_curve_geometry.py', 'weld_pipeline/server_full_weld_csv_export.py',
    'weld_pipeline/sjg_child_parent_supervision_1255.py']


def metadata(child):
    return dict(child, face_pairs=[], physical_sides=['UNKNOWN'], side_known=False,
                reference_score=.5, weld_type=child.get('weld_type', 'horizontal'))


def expect_rejection(function):
    try:
        function()
    except ValueError as exc:
        return str(exc)
    raise ValueError('A required invalid-geometry rejection did not happen')


def main(args):
    torch.set_num_threads(2); torch.manual_seed(1031)
    out = ROOT / 'server_rebuild/full_dataset_v2' / local_name(args.experiment_name)
    out.mkdir(exist_ok=False)
    sources = {name: digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(source_commit=git_head(ROOT), sources=sources,
        scope='Original five complete selected-curve reversal and synthetic line/arc common-event/runtime-link controls',
        raw_XYZ_candidate_generation_executed=False, complete_TRAIN_fitting=False,
        held_or_test_reads=0, full_contract_pass=False))
    try:
        records = []
        original_model = OUT / 'run446'
        original = read(original_model / 'training_receipt.json')
        for key in original['TRAIN']:
            path = original_model / ('test_' + key) / 'final_decoded.json'
            decoded = read(path)
            children = [compact_segment(child) for p in decoded['parents'] for child in p['segments']]
            states = directed_children(children)
            for i, child in enumerate(children):
                if geometry_fingerprint(states[2 * i]) != geometry_fingerprint(child):
                    raise ValueError('The retained original orientation changed')
                ends = np.asarray(child['endpoints_mm'])
                if not np.array_equal(states[2 * i + 1]['endpoints_mm'], ends[::-1]):
                    raise ValueError('A reversed finite curve changed ordered endpoints')
                if child['kind'] == 'arc':
                    before = CircularArc.from_parameters(child['arc_parameters'])
                    after = CircularArc.from_parameters(states[2 * i + 1]['arc_parameters'])
                    q = np.linspace(0, before.length_mm, 65)
                    if np.max(np.linalg.norm(before.point_at(q) - after.point_at(before.length_mm - q), axis=1)) > 1e-6:
                        raise ValueError('Reversing an original arc changed its whole curve')
            records.append(dict(case=key, prediction_sha256=digest(path), curves=len(children),
                directed_states=len(states), arc_curves=sum(c['kind'] == 'arc' for c in children),
                every_original_curve_checked=True, old_fields_not_fabricated=True))

        first = metadata(dict(kind='line', endpoints_mm=[[0., 0., 0.], [10., 0., 0.]],
                              length_mm=10., is_closed=False, boundary_event_ids=['start', 'cut']))
        arc = CircularArc([10., 5., 0.], 5., [0., 0., 1.], [0., -1., 0.], math.pi / 2, False)
        second = metadata(dict(compact_segment(sample_arc(arc)), boundary_event_ids=['cut', 'end']))
        second['raw_export_frame_axes'] = [np.eye(3).tolist(), np.eye(3).tolist()]
        second['frame_origins_mm'] = deepcopy(second['endpoints_mm'])
        first['raw_export_frame_axes'] = [np.eye(3).tolist(), (-np.eye(3)).tolist()]
        first['frame_origins_mm'] = [[1., 2., 3.], [11., 2., 3.]]
        first['endpoint_attributes'] = dict(start_height=dict(state='EXACT_MM', value_mm=4.),
            end_height=dict(state='EXACT_MM', value_mm=6.), back_wall=dict(state='UNKNOWN'))
        events = {'cut': dict(xyz_mm=[10., 0., 0.], construction='INPUT_CURVE_INTERSECTION',
                             input_receipt_sha256='a' * 64)}
        states = directed_children([first, second])
        parent = assemble_fixed_parent(states, [0, 2], [1], events)
        reverse_parent = assemble_fixed_parent(states, [3, 1], [1], events)
        if states[1]['endpoint_attributes']['start_height'] != first['endpoint_attributes']['end_height']:
            raise ValueError('Endpoint attributes did not follow their own physical endpoint')
        if (states[1]['raw_export_frame_axes'] != first['raw_export_frame_axes'][::-1]
                or states[1]['frame_origins_mm'] != first['frame_origins_mm'][::-1]):
            raise ValueError('Reversal moved or flipped an endpoint business frame')
        curve_rejections = dict(
            duplicate_physical_child=expect_rejection(lambda: assemble_fixed_parent(states, [0, 1], [2], events)),
            missing_common_event=expect_rejection(lambda: assemble_fixed_parent(states, [0, 2], [1], {})),
            invented_singleton_closure=expect_rejection(lambda: assemble_fixed_parent(states, [0], [], events, True)),
            collapsed_positive_gap=expect_rejection(lambda: assemble_fixed_parent(states, [0, 2], [2], events)))
        moved = deepcopy(first); moved['endpoints_mm'][1][0] = 12.; moved['length_mm'] = 12.
        moved_states = directed_children([moved, second])
        curve_rejections['nearby_but_distinct_event'] = expect_rejection(
            lambda: assemble_fixed_parent(moved_states, [0, 2], [1], events))
        fixed_gap = assemble_fixed_parent(moved_states, [0, 2], [2], events)
        old = assemble_parent([moved, second], [0, 1], [1])
        old_displacement = old['joint_boundary_max_displacement_mm']
        if old_displacement <= 0 or fixed_gap['joint_boundary_max_displacement_mm'] != 0:
            raise ValueError('The control did not distinguish old averaging from fixed finite geometry')
        for label, value in [('forward', parent), ('reverse', reverse_parent), ('gap', fixed_gap)]:
            decoded = dict(parents=[dict(value, segments=[sampled_segment(s) for s in value['segments']])],
                           invalid_predictions=[], total_prediction_children=value['child_count'], GT_or_STEP_reads=0)
            export_decoded(out / ('csv_' + label), 'geometry_control', decoded)
            verify_roundtrip(out / ('csv_' + label), decoded)

        priors = FrozenRun446Priors(ROOT)
        hidden = np.zeros((2, 128), np.float32)
        model = DirectedLinkModel().eval()
        with torch.no_grad():
            model.residual[-1].bias.copy_(torch.tensor([80., -80., -80.]))
        no_links = build_supported_parent_pool([first, second], hidden, priors, model, events)
        with torch.no_grad():
            model.residual[-1].bias.copy_(torch.tensor([-80., 80., -80.]))
        shared_links = build_supported_parent_pool([first, second], hidden, priors, model, events)
        if (any(p['child_count'] != 1 for p in no_links['parents'])
                or not any(p['directed_input_states'] == [0, 2] for p in shared_links['parents'])
                or not any(p['directed_input_states'] == [3, 1] for p in shared_links['parents'])):
            raise ValueError('Actual runtime parent generation ignored the trainable directed relation')
        if any(len(p['input_children']) != len(set(p['input_children'])) for p in shared_links['parents']):
            raise ValueError('Runtime search reused two orientations of one physical child')

        # Fit the same head used by runtime, on synthetic known common-event
        # edges only. These updates do not count as complete optimization rounds.
        train = DirectedLinkModel()
        edges = shared_links['link_pairs']; eligible = np.flatnonzero(shared_links['link_legal'][:, 1])
        if not len(eligible):
            raise ValueError('Missing real shared-event control edges')
        tensor = lambda v: torch.as_tensor(v)
        x = tensor(shared_links['link_features'][eligible])
        legal = tensor(shared_links['link_legal'][eligible])
        prior = tensor(shared_links['link_prior_probabilities'][eligible].astype(np.float32))
        emb = torch.zeros((len(eligible), 128))
        target = torch.ones(len(eligible), dtype=torch.long)
        optimizer = torch.optim.Adam(train.parameters(), lr=.01)
        losses = []
        for step in range(16):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(train(x, emb, emb, legal, prior), target)
            if not torch.isfinite(loss):
                raise ValueError('The actual directed head did not have finite loss')
            loss.backward()
            gradients = [p.grad for p in train.parameters() if p.grad is not None]
            if any(not torch.isfinite(g).all() for g in gradients):
                raise ValueError('Nonfinite trainable runtime relation gradient')
            optimizer.step(); losses.append(float(loss.detach()))
        if losses[-1] >= losses[0]:
            raise ValueError('The actual runtime relation head did not fit the synthetic event control')
        fitted_pool = build_supported_parent_pool([first, second], hidden, priors, train.eval(), events)
        if (not any(p['directed_input_states'] == [0, 2] for p in fitted_pool['parents'])
                or not any(p['directed_input_states'] == [3, 1] for p in fitted_pool['parents'])):
            raise ValueError('The genuinely fitted head did not generate both known directed parents')
        empty = build_supported_parent_pool([], np.empty((0, 128), np.float32), priors, train, {})
        if empty['parents'] or len(empty['link_pairs']):
            raise ValueError('Empty input invented a nonempty parent')
        curve_rejections['explicit_capacity_failure'] = expect_rejection(lambda:
            build_supported_parent_pool([first, second], hidden, priors, train, events, max_parents=3))
        torch.save(dict(state_dict=train.state_dict(), schema='SYNTHETIC_CONTROL_ONLY_NOT_DATASET_FIT'),
                   out / 'synthetic_control_link_head.pt')
        if sources != {name: digest(ROOT / name) for name in SOURCES}:
            raise ValueError('Control source files changed')
        write(out / 'receipt.json', dict(status='PASS_SUPPORTED_DIRECTED_PARENT_GEOMETRY_AND_LEARNED_RUNTIME_CONTROLS',
            protocol_sha256=digest(out / 'protocol.json'), original_complete_outputs=records,
            original_curve_count=sum(r['curves'] for r in records),
            original_directed_state_count=sum(r['directed_states'] for r in records),
            geometry_rejections=curve_rejections, metadata_endpoint_swap_without_sign_changes=True,
            old_unsupported_shared_displacement_mm=old_displacement,
            new_endpoint_displacement_mm=0., new_arc_refits=0, new_frame_translations=0,
            learned_runtime_control=dict(no_link_parents=len(no_links['parents']),
                shared_link_parents=len(shared_links['parents']), both_orientations_preserved=True,
                head_parameters_change_actual_runtime_generation=True,
                actual_same_head_synthetic_updates=16, synthetic_losses=losses,
                genuinely_fitted_runtime_parent_count=len(fitted_pool['parents']),
                empty_input_has_zero_output=True,
                checkpoint_sha256=digest(out / 'synthetic_control_link_head.pt')),
            all_csv_roundtrips_passed=True, TRAIN_geometry_or_process_label_reads=0,
            held_or_test_reads=0, new_complete_route_trained=False, full_contract_pass=False))
        print('PASS_GEOMETRY_PRESERVATION_AND_ACTUAL_LEARNED_LINK_RUNTIME', flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
