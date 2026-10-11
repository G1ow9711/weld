"""Offline complete-parent, process-field and raw-frame diagnostics.

Callers must authorize references after freezing predictions. This module does
not read files, fit models, change predictions or invent unknown labels. Raw
ordered geometry is a separate diagnostic from development D3 reversal; neither
establishes unconfirmed business direction, side or complete CSV acceptance.
"""
from collections import Counter

import numpy as np

from .sjg_native_d3_adapter_1056 import score_native_parents

FIELDS = ('front_wall', 'back_wall', 'start_height', 'end_height',
          'start_hole_height', 'end_hole_height', 'start_base_edge', 'end_base_edge')
STATES = ('EXACT_MM', 'DISTANCE_GT_50', 'NO_HOLE', 'NO_BLOCKER_OR_DISTANCE_GT_50')


def finite_array(value, shape):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    return array if array.shape == shape and np.isfinite(array).all() else None


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0, mean=None, median=None, p95=None, maximum=None)
    if not np.isfinite(values).all():
        raise ValueError('Diagnostic values must be finite')
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def process_reference(target, records, arrays, endpoint_evidence):
    """Compile only supplied authorized labels in their original child order."""
    children = [c for p in target['parents'] for c in p['children']]
    if [c['original_child_id'] for c in children] != [r['child_id'] for r in records]:
        raise ValueError('Process labels differ from original ordered child owners')
    n = len(children)
    if n != target['original_children']:
        raise ValueError('Original child denominator changed')
    state, required, numeric = [np.asarray(arrays[k]) for k in ('state', 'required_mask', 'exact_mm')]
    if (state.shape != (n, 8) or not np.issubdtype(state.dtype, np.integer)
            or required.shape != (n, 8) or required.dtype != np.bool_
            or numeric.shape != (n, 8) or np.any((state < -1) | (state > 3))
            or not np.isfinite(numeric[state == 0]).all()
            or not np.isnan(numeric[state != 0]).all()):
        raise ValueError('Original field states, masks or numerical labels changed')
    axes = np.full((n, 2, 3, 3), np.nan)
    origins = np.full((n, 2, 3), np.nan)
    known = np.zeros((n, 2), bool)
    parents = {p['parent_id']: p for p in endpoint_evidence['parents']}
    offsets = np.r_[0, np.cumsum([len(p['children']) for p in target['parents']])]
    row = 0
    for parent in target['parents']:
        evidence = parents[parent['original_parent_id']]['children']
        if [c['original_child_id'] for c in parent['children']] != [c['child_id'] for c in evidence]:
            raise ValueError('Original frame owners or child order changed')
        for child in evidence:
            for side, key in enumerate(('start', 'end')):
                frame = child['endpoint_frames'][key]
                if frame['known']:
                    a = finite_array(frame['raw_export_axes'], (3, 3))
                    origin = finite_array(frame['translation'], (3,))
                    if a is None or origin is None:
                        raise ValueError('Known original frame is malformed')
                    axes[row, side], origins[row, side], known[row, side] = a, origin, True
            row += 1
    return dict(state=state, required=required, numeric=numeric, axes=axes,
                origins=origins, frame_known=known, parent_offsets=offsets)


def full_parent_geometry(decoded, target, score):
    matched = {}
    for match in score['matches']:
        identity = (match['predicted_parent_index'], match['original_parent_index'])
        matched.setdefault(identity, set()).add(match['child_index'])
    complete = []
    for (pi, ti), children in matched.items():
        expected = target['parents'][ti]['original_child_count']
        if children == set(range(expected)) and decoded['parents'][pi]['child_count'] == expected:
            complete.append(dict(predicted_parent_index=pi, original_parent_index=ti, children=expected))
    return dict(reference_parents=len(target['parents']), predicted_parents=len(decoded['parents']),
                fully_matched_parents=len(complete), partially_matched_parents=len(matched)-len(complete),
                complete_pairs=complete, criterion='Every original child matches within the same D3-assigned parent',
                process_or_business_frame_acceptance=False)


def assess_complete_output(decoded, target, reference, development_score=None):
    development = development_score or score_native_parents(decoded, target, [False]*len(target['parents']))
    ordered = score_native_parents(decoded, target, [True]*len(target['parents']))
    state, required = reference['state'], reference['required']
    field_rows = {name: Counter(known_required=int(np.sum(required[:, j] & (state[:, j] >= 0))),
                         unknown_required=int(np.sum(required[:, j] & (state[:, j] < 0))),
                         required=int(required[:, j].sum()), compared=0, state_correct=0,
                         state_wrong=0, prediction_missing=0, numeric_prediction_missing=0,
                         negative_exact_predictions=0)
                  for j, name in enumerate(FIELDS)}
    field_errors = {name: [] for name in FIELDS}
    frames = Counter(known_reference_endpoints=int(reference['frame_known'].sum()),
                     unknown_reference_endpoints=int((~reference['frame_known']).sum()),
                     geometry_matched_known_endpoints=0, missing_predicted_axes=0,
                     missing_predicted_origins=0, rotation_like_endpoints=0,
                     business_frame_certified_predictions=0)
    axis_errors = {name: [] for name in ('X', 'Y', 'Z')}
    origin_errors, orthogonality_errors, determinant_errors = [], [], []
    diagnostics = []
    for match in ordered['matches']:
        if match['reversed_whole_parent']:
            raise ValueError('Raw ordered diagnostic unexpectedly reversed a parent')
        pi, ti, ci = [match[k] for k in ('predicted_parent_index', 'original_parent_index', 'child_index')]
        prediction = decoded['parents'][pi]['segments'][match['predicted_child_index']]
        row = int(reference['parent_offsets'][ti]) + ci
        attributes = prediction.get('endpoint_attributes', {})
        child = dict(predicted_parent_index=pi, predicted_child_index=match['predicted_child_index'],
                     original_parent_index=ti, original_child_index=ci, attributes={}, frames=[])
        for j, name in enumerate(FIELDS):
            if not required[row, j] or state[row, j] < 0:
                continue
            metric = field_rows[name]
            predicted = attributes.get(name, {})
            code = predicted.get('state')
            expected = STATES[int(state[row, j])]
            if code not in STATES:
                metric['prediction_missing'] += 1
            else:
                metric['compared'] += 1
                metric['state_correct' if code == expected else 'state_wrong'] += 1
            item = dict(expected_state=expected, predicted_state=code,
                        state_correct=code == expected, absolute_error_mm=None)
            if state[row, j] == 0:
                value = finite_array(predicted.get('value_mm'), ()) if code == 'EXACT_MM' else None
                if value is None:
                    metric['numeric_prediction_missing'] += 1
                else:
                    error = abs(float(value)-float(reference['numeric'][row, j]))
                    field_errors[name].append(error)
                    item['absolute_error_mm'] = error
                    metric['negative_exact_predictions'] += int(float(value) < 0)
            child['attributes'][name] = item
        axes = finite_array(prediction.get('raw_export_frame_axes'), (2, 3, 3))
        origins = finite_array(prediction.get('frame_origins_mm'), (2, 3))
        for side in range(2):
            if not reference['frame_known'][row, side]:
                continue
            frames['geometry_matched_known_endpoints'] += 1
            item = dict(endpoint='start' if side == 0 else 'end', origin_error_mm=None, signed_axis_errors_deg=None)
            if origins is None:
                frames['missing_predicted_origins'] += 1
            else:
                error = float(np.linalg.norm(origins[side]-reference['origins'][row, side]))
                origin_errors.append(error);item['origin_error_mm'] = error
            if axes is None:
                frames['missing_predicted_axes'] += 1
            else:
                a, truth = axes[side], reference['axes'][row, side]
                orthogonal = float(np.abs(a@a.T-np.eye(3)).max())
                determinant = abs(float(np.linalg.det(a))-1.)
                orthogonality_errors.append(orthogonal);determinant_errors.append(determinant)
                frames['rotation_like_endpoints'] += int(orthogonal <= 1e-3 and determinant <= 1e-3)
                norms = np.linalg.norm(a,axis=1)*np.linalg.norm(truth,axis=1)
                if np.all(norms > 0):
                    angles = np.rad2deg(np.arccos(np.clip(np.sum(a*truth,axis=1)/norms,-1,1)))
                    item['signed_axis_errors_deg'] = angles.tolist()
                    for name, angle in zip(('X','Y','Z'),angles):axis_errors[name].append(float(angle))
                frames['business_frame_certified_predictions'] += int(prediction.get('frame_business_semantics_certified') is True)
            child['frames'].append(item)
        diagnostics.append(child)
    for metric in field_rows.values():
        metric['not_geometry_matched_known_required'] = (
            metric['known_required'] - metric['compared'] - metric['prediction_missing'])
    output = Counter(parents=len(decoded['parents']), children=decoded['total_prediction_children'],
                     unknown_weld_types=0, unknown_sides=0, missing_required_field_states=0,
                     missing_exact_field_values=0, malformed_frame_axes=0, malformed_frame_origins=0,
                     uncertified_business_frames=0)
    for parent in decoded['parents']:
        for child in parent['segments']:
            weld = child.get('weld_type')
            output['unknown_weld_types'] += int(weld not in ('horizontal','vertical'))
            output['unknown_sides'] += int(child.get('predicted_side') in (None,'UNKNOWN'))
            fields = FIELDS if weld == 'horizontal' else ('back_wall',) if weld == 'vertical' else ()
            for name in fields:
                attribute = child.get('endpoint_attributes',{}).get(name,{})
                code = attribute.get('state')
                output['missing_required_field_states'] += int(code not in STATES)
                output['missing_exact_field_values'] += int(code == 'EXACT_MM' and finite_array(attribute.get('value_mm'),()) is None)
            output['malformed_frame_axes'] += int(finite_array(child.get('raw_export_frame_axes'),(2,3,3)) is None)
            output['malformed_frame_origins'] += int(finite_array(child.get('frame_origins_mm'),(2,3)) is None)
            output['uncertified_business_frames'] += int(child.get('frame_business_semantics_certified') is not True)
    return dict(development_d3_geometry=development['categories'], raw_ordered_geometry=ordered['categories'],
        complete_parent_development_geometry=full_parent_geometry(decoded,target,development),
        complete_parent_raw_ordered_geometry=full_parent_geometry(decoded,target,ordered),
        fields={name: dict(field_rows[name], absolute_error_mm=statistics(field_errors[name])) for name in FIELDS},
        raw_frames=dict(frames, origin_error_mm=statistics(origin_errors),
            signed_axis_errors_deg={name:statistics(axis_errors[name]) for name in axis_errors},
            orthogonality_error=statistics(orthogonality_errors), determinant_error=statistics(determinant_errors)),
        all_output_completeness=dict(output), matched_child_diagnostics=diagnostics,
        field_and_frame_metrics_conditioned_on_raw_ordered_geometry=True,
        unknown_labels_not_counted_as_correct_or_incorrect=True, censored_codes_not_measured_as_distances=True,
        raw_frame_signs_and_axes_and_coordinates_unchanged=True, unknown_fields_not_filled=True,
        attribute_mm_tolerance_not_frozen=True, business_direction_and_side_not_certified=True,
        split_reasons_and_csv_and_raw_input_timing_not_evaluated=True, full_contract_pass=False)
