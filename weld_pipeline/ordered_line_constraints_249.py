"""Conditional offline ordered-line parameters for already selected identities.

No runtime slot is created from a reference. This checks one fixed line/parent
assignment, not full decoder expressivity, material identity, or predictions.
"""
from fractions import Fraction
from decimal import Decimal, localcontext
import math

import numpy as np

from .finite_line_expression_246 import ball_parameter_interval
from .finite_surface_sections_240 import union_intervals, intersect_interval_unions

VERSION = 'CONDITIONAL_SHARED_BOUNDARY_ORDERED_LINE_PARAMETERS_249_v1'


def _rows(value):
    return union_intervals(np.asarray(value, dtype=np.float64).reshape(-1, 2))


def _signed(rows, orientation):
    return _rows(rows if orientation > 0 else -rows[:, ::-1])


def _at_or_before(rows, bound):
    for lo, hi in rows[::-1]:
        if lo <= bound:
            return min(float(hi), bound)
    return None


def _interior_at_or_after(rows, lower, upper):
    for lo, hi in rows:
        lo, hi = max(float(lo), lower), min(float(hi), upper)
        if lo <= hi:
            with localcontext() as context:
                context.prec = 160
                midpoint = float((Decimal.from_float(lo) + Decimal.from_float(hi))/2)
            return min(hi, max(lo, midpoint))
    return None


def _ordered_values(sequence, ranges):
    latest = {}
    next_value = math.inf
    for identity in reversed(sequence):
        upper = math.nextafter(next_value, -math.inf) if math.isfinite(next_value) else next_value
        value = _at_or_before(ranges[identity], upper)
        if value is None:
            return None
        latest[identity] = value
        next_value = value
    output = {}; previous = -math.inf
    for identity in sequence:
        lower = math.nextafter(previous, math.inf) if math.isfinite(previous) else previous
        value = _interior_at_or_after(ranges[identity], lower, latest[identity])
        if value is None:
            return None
        output[identity] = value; previous = value
    return output


def _squared(a, b):
    return sum((Fraction.from_float(float(x)) - Fraction.from_float(float(y)))**2
               for x, y in zip(a, b))


def solve_fixed_line_parent(origin_mm, direction, children, gap_relations,
                            boundary_permissions_mm, *, parent_slot_id, construction_id,
                            orientation=1):
    """Find a simultaneous witness conditional on a fixed legal slot skeleton.

    Children supply line reference endpoints, existing start/end BoundarySlot
    IDs, and original finite carrier intervals. Slot membership/capacity must
    be verified independently against the frozen runtime manifest.
    """
    origin, vector = np.asarray(origin_mm, float), np.asarray(direction, float)
    if (origin.shape != (3,) or vector.shape != (3,) or not np.isfinite([origin, vector]).all()
            or isinstance(orientation, (bool, np.bool_))
            or not isinstance(orientation, (int, np.integer)) or orientation not in (-1, 1)):
        raise ValueError('Finite original line and explicit orientation required')
    if not children or len(gap_relations) != len(children)-1:
        raise ValueError('Complete ordered child and adjacency lists required')
    if len({child['child_slot_id'] for child in children}) != len(children):
        raise ValueError('Defining child identities cannot be duplicated')
    result = dict(version=VERSION, scope='CONDITIONAL_FIXED_LINE_PARENT_PARAMETERS_ONLY',
                  parent_slot_id=parent_slot_id, construction_id=construction_id, orientation=orientation,
                  original_child_count=len(children), actual_runtime_slot_capacity_verified=False,
                  complete_all_source_joint_expression_verified=False,
                  material_or_weld_identity_verified=False,
                  reference_used_for_runtime_candidate_generation=False,
                  positive_length_cutoff_applied=False, positive_gap_merged=False,
                  model_inference_calls=0, actual_weld_optimizer_updates=0, full_contract_pass=False)
    sequence = []
    for number, child in enumerate(children):
        if child.get('reference_kind', 'line') != 'line':
            return {**result, 'status': 'UNSUPPORTED_REFERENCE_KIND_FOR_FIXED_LINE_CONDITIONAL_MODULE'}
        reference = np.asarray([child['reference_start_mm'], child['reference_end_mm']], dtype=float)
        if reference.shape != (2, 3) or not np.isfinite(reference).all() or np.array_equal(*reference):
            raise ValueError('Two finite distinct reference line endpoints required')
        start, end = child['start_boundary_slot_id'], child['end_boundary_slot_id']
        if start not in boundary_permissions_mm or end not in boundary_permissions_mm:
            raise ValueError('Every boundary needs frozen finite local permissions')
        if start == end:
            return {**result, 'status': 'NO_POSITIVE_LINE_CHILD_WITH_SAME_BOUNDARY_IDENTITY'}
        if number:
            previous = children[number-1]['end_boundary_slot_id']
            relation = gap_relations[number-1]
            if relation == 'SHARED_BOUNDARY':
                if start != previous:
                    raise ValueError('Shared cut must reference the same boundary identity')
            elif relation == 'POSITIVE_GAP':
                if start == previous:
                    return {**result, 'status': 'POSITIVE_GAP_BOUNDARY_IDENTITY_COLLISION'}
                sequence.append(start)
            else:
                raise ValueError('Explicit shared-boundary or positive-gap relation required')
        else:
            sequence.append(start)
        sequence.append(end)
    if len(sequence) != len(set(sequence)):
        return {**result, 'status': 'STRICT_LINE_ORDER_IDENTITY_CYCLE', 'ordered_boundary_ids': sequence}

    ranges = {identity: _signed(_rows(boundary_permissions_mm[identity]), orientation) for identity in sequence}
    options = []
    for child in children:
        for field, reference in (('start_boundary_slot_id', child['reference_start_mm']),
                                 ('end_boundary_slot_id', child['reference_end_mm'])):
            ball = ball_parameter_interval(origin, vector, reference)
            if ball['status'] == 'BALL_MISSES_INFINITE_LINE':
                return {**result, 'status': 'REFERENCE_ENDPOINT_BALL_MISSES_FIXED_LINE', 'child_slot_id': child['child_slot_id']}
            if ball['interval_mm'] is None:
                return {**result, 'status': 'UNRESOLVED_ENDPOINT_BALL', 'child_slot_id': child['child_slot_id']}
            identity = child[field]
            ranges[identity] = intersect_interval_unions(ranges[identity], _signed(_rows([ball['interval_mm']]), orientation))
        components = _signed(_rows(child['finite_intervals_mm']), orientation)
        options.append(components)
    if any(len(value) == 0 for value in ranges.values()):
        return {**result, 'status': 'EMPTY_SHARED_BOUNDARY_PARAMETER_INTERSECTION',
                'empty_boundary_ids': [key for key, value in ranges.items() if not len(value)]}

    stack = [(0, ranges, [])]
    visited = 0; unresolved_export = 0
    while stack:
        index, allowed, selected = stack.pop(); visited += 1
        if index < len(children):
            child = children[index]
            for ordinal in reversed(range(len(options[index]))):
                component = options[index][ordinal:ordinal+1]
                if component[0, 0] == component[0, 1]:
                    continue
                local = dict(allowed)
                for boundary in (child['start_boundary_slot_id'], child['end_boundary_slot_id']):
                    local[boundary] = intersect_interval_unions(local[boundary], component)
                if all(len(local[boundary]) for boundary in sequence) and _ordered_values(sequence, local) is not None:
                    stack.append((index+1, local, selected+[ordinal]))
            continue
        coordinates = _ordered_values(sequence, allowed)
        if coordinates is None:
            continue
        world = {key: origin + orientation*value*vector for key, value in coordinates.items()}
        certificates = []; valid = True
        for number, child in enumerate(children):
            start, end = child['start_boundary_slot_id'], child['end_boundary_slot_id']
            a, b = world[start], world[end]
            errors = [_squared(a, child['reference_start_mm']), _squared(b, child['reference_end_mm'])]
            if np.array_equal(a, b) or max(errors) > 9:
                valid = False; unresolved_export += 1; break
            component = options[number][selected[number]]
            assert component[0] <= coordinates[start] < coordinates[end] <= component[1]
            certificates.append(dict(child_slot_id=child['child_slot_id'],
                start_boundary_slot_id=start, end_boundary_slot_id=end,
                start_mm=a.tolist(), end_mm=b.tolist(), selected_component_ordinal=selected[number],
                positive_parameter_span_mm=coordinates[end]-coordinates[start],
                endpoint_squared_errors_exact=[dict(numerator=str(e.numerator), denominator=str(e.denominator)) for e in errors],
                endpoint_and_bidirectional_continuous_error_limit_mm=3.,
                proof='CORRESPONDING_ENDPOINTS_AND_CONVEX_INTERPOLATION'))
        if not valid:
            continue
        assert all(coordinates[b] > coordinates[a] for a, b in zip(sequence, sequence[1:]))
        return {**result, 'status': 'SIMULTANEOUS_FIXED_LINE_PARENT_PARAMETER_WITNESS',
                'ordered_boundary_ids': sequence,
                'boundary_t_mm': {key: orientation*value for key, value in coordinates.items()},
                'shared_world_coordinates_mm': {key: value.tolist() for key, value in world.items()},
                'child_certificates': certificates, 'gap_relations': list(gap_relations),
                'selected_component_ordinals': selected, 'search_states_examined': visited,
                'shared_coordinates_and_strict_order_verified': True}
    return {**result, 'status': 'UNRESOLVED_FLOAT_EXPORT_WITNESS' if unresolved_export else 'NO_FIXED_FLOAT_LINE_PARAMETER_WITNESS',
            'search_states_examined': visited, 'unresolved_export_cases': unresolved_export,
            'no_claim_of_global_decoder_inexpressibility': True}
