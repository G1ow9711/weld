"""Recover all finite contact intervals and retain local property transitions.

These are unlabelled geometric hypotheses, not confirmed welds. Actual observed
face-region pairs remain distinct even when their infinite planes coincide.
Thickness changes split attribute regions instead of deleting the whole trace.
Unknown attributes remain unknown; no labels, STEP or model weights are used.
"""
import collections
import copy

import numpy as np

from . import server_plate_pair_evidence as check

POLICY = dict(candidate_branch='global_observed_plane_pair',
    observation_support_radius_mm=4., plane_band_mm=.05, distinct_face_support_mm=.25,
    maximum_projection_mm=3., output_sample_step_mm=1., attribute_station_step_mm=50.,
    transition_probe_bracket_mm=1., maximum_own_face_offset_spread_mm=.05,
    candidate_interval_overlap_roundoff_mm=0.,
    collect_original_endpoint_lines_before_local_plane_fitting=True,
    use_actual_face_regions_not_infinite_plane_aliases=True,
    minimum_output_length_mm=None, fill_unsupported_gaps=False,
    geometric_hypotheses_are_weld_labels=False, signed_business_frames_inferred=False,
    sub_station_property_changes_certified=False, full_contract_pass=False)


def supported_distance(xyz, tree, points, normals, offsets):
    radius = POLICY['observation_support_radius_mm']
    distances, ids = tree.query(points, k=min(64, len(xyz)), workers=2)
    nearest = xyz[ids]
    residuals = np.array([abs(nearest@n-off) for n,off in zip(normals,offsets)])
    # Each observed face needs independent off-intersection support.
    support = np.array([np.min(np.where((residuals[side]<=POLICY['plane_band_mm']) &
                        (residuals[1-side]>=POLICY['distinct_face_support_mm']), distances, np.inf), axis=1)
                        for side in range(2)])
    for i in np.flatnonzero((distances[:, -1]<=radius) & (support.max(axis=0)>radius)):
        local = xyz[tree.query_ball_point(points[i], radius, workers=2)]
        d = np.linalg.norm(local-points[i], axis=1)
        for side, (n, off) in enumerate(zip(normals, offsets)):
            selected = d[(abs(local@n-off)<=POLICY['plane_band_mm']) &
                         (abs(local@normals[1-side]-offsets[1-side])>=POLICY['distinct_face_support_mm'])]
            if len(selected): support[side, i] = selected.min()
    return support.max(axis=0)


def collect_regions(profiles, faces):
    """Retain original candidate lines before fitting their local raw surfaces.

    Cached face centers must not reject locally supported original endpoints.
    Only identical geometric lines of the same actual face pair share intervals.
    The canonical direction orders geometry; it is not a business weld frame.
    """
    lines = {}
    counts = collections.Counter()
    for index, profile in enumerate(profiles):
        counts['all_source_profiles'] += 1
        if profile.get('proposal_branch') != POLICY['candidate_branch']:
            counts['other_candidate_branches_preserved_outside_this_line_helper'] += 1
            continue
        representation = profile.get('representation', {})
        indices = profile.get('face_indices', [])
        if representation.get('kind') != 'line' or len(indices) != 2 or indices[0] == indices[1]:
            counts['not_a_two_face_line_region'] += 1
            continue
        if any(type(i) is not int or not 0 <= i < len(faces) for i in indices):
            raise ValueError('Candidate references an invalid observed finite face')
        pair = tuple(sorted(indices))
        ends = np.asarray(representation['endpoints_mm'], dtype=np.float64)
        if ends.shape != (2, 3) or not np.isfinite(ends).all():
            raise ValueError('Finite candidate endpoints are required')
        direction = ends[1] - ends[0]
        length = float(np.linalg.norm(direction))
        if length <= 0:
            counts['zero_length_candidate'] += 1
            continue
        direction /= length
        if direction[np.argmax(abs(direction))] < 0:
            direction = -direction
        anchor = ends[np.lexsort(ends.T[::-1])[0]]
        origin = anchor - np.dot(anchor, direction) * direction
        key = (pair, tuple(direction), tuple(origin))
        if key not in lines:
            normals = np.asarray([faces[i]['normal'] for i in pair], dtype=np.float64)
            lengths = np.linalg.norm(normals, axis=1)
            if normals.shape != (2, 3) or not np.isfinite(normals).all() or np.any(lengths == 0):
                raise ValueError('Observed plane geometry must be finite and nondegenerate')
            normals = normals / lengths[:, None]
            lines[key] = dict(face_indices=list(pair), normals=normals,
                origin=origin, direction=direction, intervals=[])
        item = lines[key]
        positions = (ends - item['origin']) @ item['direction']
        lo, hi = float(positions.min()), float(positions.max())
        if hi <= lo:
            counts['zero_length_candidate'] += 1
            continue
        item['intervals'].append((lo, hi, index))
        counts['all_finite_candidate_intervals_collected'] += 1
    result = []
    region_numbers = collections.Counter()
    for line_number, (key, item) in enumerate(sorted(lines.items())):
        pair = key[0]
        intervals = []
        for lo, hi, source in sorted(item['intervals']):
            if intervals and lo <= intervals[-1][1] + POLICY['candidate_interval_overlap_roundoff_mm']:
                intervals[-1][1] = max(intervals[-1][1], hi)
                intervals[-1][2].add(source)
            else:
                intervals.append([lo, hi, {source}])
        for lo, hi, sources in intervals:
            result.append(dict(face_indices=list(pair), region_index=region_numbers[pair],
                original_geometric_line_index=line_number,
                endpoints_mm=(item['origin'] + np.asarray([lo, hi])[:, None] * item['direction']).tolist(),
                observed_normals=item['normals'].tolist(),
                source_profile_indices=sorted(sources), candidate_projection_max_mm=0.,
                intervals_merged_only_when_overlapping=True, material_identity='UNKNOWN'))
            region_numbers[pair] += 1
    counts['actual_observed_face_pairs'] = len({tuple(r['face_indices']) for r in result})
    counts['all_continuous_candidate_regions'] = len(result)
    return result, dict(counts)


def station(xyz, tree, origin, direction, normals):
    ids = tree.query_ball_point(origin, 175., workers=2)
    points = np.asarray(xyz[ids])
    points = points[abs((points - origin) @ direction) <= check.POLICY['section_half_width_mm']]
    faces = [check.one_plate(points, origin, direction, normal) for normal in normals]
    return dict(position_mm=origin.tolist(), faces=faces,
        both_supported=all(face['status'] == 'SUPPORTED_PARALLEL_PLATE' for face in faces))


def compatible(samples, normals):
    if not samples or not all(sample['both_supported'] for sample in samples):
        return False
    for side in range(2):
        faces = [sample['faces'][side] for sample in samples]
        if np.ptp([face['thickness_mm'] for face in faces]) > check.POLICY['maximum_uniform_thickness_spread_mm']:
            return False
        offsets = [np.dot(face['face_center_mm'], normals[side]) for face in faces]
        if np.ptp(offsets) > POLICY['maximum_own_face_offset_spread_mm']:
            return False
        directed = np.asarray([face['directed_normal'] for face in faces])
        if np.min(directed @ directed[0]) < np.cos(np.deg2rad(1.)):
            return False
    return True


def same_own_faces(samples, normals):
    """An unknown opposite-face transition cannot bridge displaced own faces."""
    if not all(sample['both_supported'] for sample in samples):
        return False
    for side in range(2):
        faces = [sample['faces'][side] for sample in samples]
        offsets = [np.dot(face['face_center_mm'], normals[side]) for face in faces]
        if np.ptp(offsets) > POLICY['maximum_own_face_offset_spread_mm']:
            return False
        directed = np.asarray([face['directed_normal'] for face in faces])
        if np.min(directed @ directed[0]) < np.cos(np.deg2rad(1.)):
            return False
    return True


def recover_region(xyz, tree, region):
    """Keep supported geometry on both sides of changes; unknowns are separate."""
    ends = np.asarray(region['endpoints_mm'], dtype=np.float64)
    normals = np.asarray(region['observed_normals'], dtype=np.float64)
    direction = ends[1] - ends[0]
    length = float(np.linalg.norm(direction))
    if length <= 0:
        raise ValueError('A positive finite region length is required')
    direction /= length
    cache = {}

    def probe(position):
        position = float(position)
        if position not in cache:
            cache[position] = station(xyz, tree, ends[0] + position * direction, direction, normals)
        return cache[position]

    positions = np.linspace(0., length, max(2, int(np.ceil(length / POLICY['attribute_station_step_mm'])) + 1)).tolist()
    for position in positions:
        probe(position)

    def refine(a, b):
        left, right = probe(a), probe(b)
        if compatible([left, right], normals) or b - a <= POLICY['transition_probe_bracket_mm']:
            return
        if not left['both_supported'] and not right['both_supported']:
            return
        mid = (a + b) / 2.
        probe(mid)
        refine(a, mid)
        refine(mid, b)

    for a, b in zip(positions[:-1], positions[1:]):
        refine(a, b)
    positions = sorted(cache)
    if sum(sample['both_supported'] for sample in cache.values()) < 2:
        return dict(observed_face_regions=region['face_indices'], region_index=region['region_index'],
            source_profile_indices=region['source_profile_indices'], children=[],
            unresolved_intervals=[dict(endpoints_mm=ends.tolist(), reason='INSUFFICIENT_TWO_PLATE_EVIDENCE')],
            local_attribute_stations=len(cache), local_property_regions=0, geometry_fallback=True,
            geometric_trace_is_confirmed_parent_weld=False, inference_reference_labels_used=False,
            STEP_used=False, passed_learned_final_selector=False, full_contract_pass=False)
    boundaries = [0., *[(a+b)/2. for a, b in zip(positions[:-1], positions[1:])], length]
    groups = []
    for index, position in enumerate(positions):
        sample = probe(position)
        state = 'SUPPORTED_UNIFORM_LOCAL_ATTRIBUTES' if sample['both_supported'] else 'OPPOSITE_PLATE_ATTRIBUTES_UNRESOLVED'
        join = bool(groups and groups[-1]['state'] == state and (
            compatible([*groups[-1]['samples'], sample], normals) if sample['both_supported'] else True))
        if join:
            groups[-1]['hi'] = boundaries[index+1]
            groups[-1]['samples'].append(sample)
        else:
            groups.append(dict(lo=boundaries[index], hi=boundaries[index+1], state=state, samples=[sample]))
    children = []
    unresolved = []
    for index, group in enumerate(groups):
        original = ends[0] + np.asarray([group['lo'], group['hi']])[:, None] * direction
        local_faces = group['samples'][0]['faces']
        if group['state'] == 'OPPOSITE_PLATE_ATTRIBUTES_UNRESOLVED':
            neighbors = [groups[index-1]['samples'][-1], groups[index+1]['samples'][0]] if 0 < index < len(groups)-1 else []
            if not neighbors or not same_own_faces(neighbors, normals):
                unresolved.append(dict(endpoints_mm=original.tolist(),
                    reason='UNRESOLVED_PLATE_REGION_WITHOUT_SAME_OBSERVED_FACES_ON_BOTH_SIDES'))
                continue
            local_faces = neighbors[0]['faces']
        evidence = check.segment_attributes(xyz, tree, dict(endpoints_mm=original), normals)
        known = bool(group['state'] == 'SUPPORTED_UNIFORM_LOCAL_ATTRIBUTES' and evidence['plate_pair_supported'] and evidence.get('uniform_thickness'))
        planes = evidence['plates'] if known else None
        own_planes = planes if known else local_faces
        current_normals = np.asarray([p['directed_normal'] for p in own_planes])
        current_offsets = np.asarray([p['directed_plane_offset_mm'] for p in own_planes])
        gram = current_normals @ current_normals.T
        if np.linalg.det(gram) < .01:
            unresolved.append(dict(endpoints_mm=original.tolist(), reason='LOCAL_FACES_NEAR_PARALLEL'))
            continue
        projected = original + (current_offsets - original @ current_normals.T) @ np.linalg.inv(gram) @ current_normals
        movement = float(np.linalg.norm(projected-original, axis=1).max())
        if movement > POLICY['maximum_projection_mm']:
            unresolved.append(dict(endpoints_mm=original.tolist(), reason='LOCAL_PROJECTION_EXCEEDS_ORIGINAL_GATE'))
            continue
        span = float(np.linalg.norm(projected[1]-projected[0]))
        points = np.linspace(*projected, max(2, int(np.ceil(span / POLICY['output_sample_step_mm']))+1))
        distances = supported_distance(xyz, tree, points, current_normals, current_offsets)
        good = distances <= POLICY['observation_support_radius_mm']
        changes = np.diff(np.r_[False, good, False].astype(np.int8))
        for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if b-a < 2:
                unresolved.append(dict(position_mm=points[a].tolist(), reason='ISOLATED_SUPPORT_SAMPLE'))
                continue
            actual_length = float(np.linalg.norm(points[b-1]-points[a]))
            if actual_length <= 0:
                continue
            children.append(dict(kind='line', points=points[a:b].tolist(),
                endpoints_mm=points[[a, b-1]].tolist(), length_mm=actual_length,
                plate_attributes=copy.deepcopy(planes), attributes_status=(
                    'STABLE_LOCAL_GEOMETRIC_ESTIMATES_NOT_CUSTOMER_VERIFIED' if known else 'UNKNOWN'),
                weld_type='UNKNOWN', predicted_side='UNKNOWN', process_attributes=None,
                material_identity='UNKNOWN', business_coordinate_frame=None,
                split_reason='LOCAL_PLATE_PROPERTY_REGION' if len(groups)>1 else 'NONE_OBSERVED',
                evidence=dict(observed_face_regions=region['face_indices'], property_region=index,
                    local_status=evidence['status'], local_attribute_probe_count=len(group['samples']),
                    both_distinct_faces_observed=True, plate_attributes_known=known,
                    endpoint_projection_movement_mm=movement,
                    support_distance_max_mm=float(distances[a:b].max()),
                    observation_support_radius_mm=POLICY['observation_support_radius_mm'],
                    reference_labels_used=False, passed_learned_final_selector=False,
                    unsupported_gaps_filled=False, not_ground_truth_error=True)))
        missing = np.diff(np.r_[False, ~good, False].astype(np.int8))
        for a, b in zip(np.flatnonzero(missing == 1), np.flatnonzero(missing == -1)):
            unresolved.append(dict(endpoints_mm=points[[a, b-1]].tolist(),
                reason='UNSUPPORTED_OBSERVATION_GAP', observation_is_not_physical_gap_certification=True))
    return dict(observed_face_regions=region['face_indices'], region_index=region['region_index'],
        source_profile_indices=region['source_profile_indices'], children=children,
        unresolved_intervals=unresolved, local_attribute_stations=len(cache),
        local_property_regions=len(groups), geometry_fallback=True,
        geometric_trace_is_confirmed_parent_weld=False, true_weld_type_and_parent_order_unknown=True,
        inference_reference_labels_used=False, STEP_used=False, passed_learned_final_selector=False,
        complete_process_attributes=False, full_contract_pass=False)
