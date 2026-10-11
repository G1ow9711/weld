"""Finite XYZ face fits and bilateral line witnesses; no certified edge claims.

All row identifiers refer to the unmodified original cloud. Plane fits use every
row, including duplicates. Sampling coverage and fit residual remain separate.
"""
import hashlib
import numpy as np
from weld_pipeline.endpoint_local_finite_support_645 import frame

VERSION = 'FINITE_FLOAT64_FACE_BILATERAL_LINE_EVIDENCE_V1'


def fit_face(xyz, surface, face_id):
    rows = np.asarray(surface['original_row_indices'], np.int64)
    points = np.asarray(xyz[rows], np.float64)
    center = points.mean(0)
    centered = points - center
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    normal, _ = frame(vectors[:, 0])
    residual = np.abs(centered @ normal)
    eligible = bool(surface['two_dimensional_material_support'] and values[1] > .15**2)
    return dict(face_id=face_id, original_row_count=len(rows),
        original_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        component_index=int(surface['component_index']),
        eligible_finite_2D=eligible, unresolved_reason=surface.get('unresolved_reason'),
        centroid_mm=center.tolist(), normal=normal.tolist(), plane_offset_mm=float(center @ normal),
        covariance_eigenvalues_mm2=values.tolist(),
        fit_residual_max_mm=float(residual.max()), fit_residual_rms_mm=float(np.sqrt(np.mean(residual**2))),
        original_aabb_min_mm=points.min(0).tolist(), original_aabb_max_mm=points.max(0).tolist(),
        coverage=dict(original_resolution_mm=float(surface['resolution_mm']),
            material_cells=int(surface['unique_material_cells']),
            outer_boundary_cells=len(surface['outer_boundary_cells']),
            inner_boundary_cells=len(surface['inner_boundary_cells']),
            raster_inner_regions=len(surface['inner_free_regions']),
            original_boundary_kind=surface['boundary_kind']),
        original_field_uncertainty_mm=float(surface['field_uncertainty_mm']),
        hole_status='UNRESOLVED_HOLE_SUPPORT',
        certified_boundary_error_bound_mm=None, physical_plate_identity_confirmed=False,
        plane_residual_is_not_boundary_error=True)


def finite_line_runs(xyz, rows, face, origin, direction):
    """Split actual near-line witnesses at unsupported sampling gaps.

    Intervals are witnessed spans, not proof of continuous material. In
    particular they must not be used directly as a 3 mm field certificate.
    """
    points = np.asarray(xyz[rows], np.float64)
    delta = points-origin
    t = delta @ direction
    distances = np.linalg.norm(delta - t[:, None]*direction, axis=1)
    band = 1.5*face['coverage']['original_resolution_mm'] + .15
    keep = distances <= band
    selected = np.flatnonzero(keep)
    if not len(selected):
        return [], dict(support_band_mm=band, witness_rows=0, nearest_row_distance_mm=float(distances.min()))
    order = selected[np.argsort(t[selected], kind='stable')]
    cuts = np.flatnonzero(np.diff(t[order]) > 2.1*face['coverage']['original_resolution_mm'])+1
    groups = np.split(order, cuts)
    runs = []
    for group in groups:
        runs.append(dict(t_min_mm=float(t[group[0]]), t_max_mm=float(t[group[-1]]),
            original_row_indices=np.asarray(rows[group], np.int64),
            minimum_line_distance_mm=float(distances[group].min()),
            maximum_line_distance_mm=float(distances[group].max()),
            maximum_along_line_gap_mm=float(np.diff(t[group]).max()) if len(group)>1 else 0.))
    return runs, dict(support_band_mm=band,witness_rows=len(selected),
        nearest_row_distance_mm=float(distances.min()), run_count=len(runs))


def bilateral_line_evidence(xyz, first, first_rows, second, second_rows):
    if not first['eligible_finite_2D'] or not second['eligible_finite_2D']:
        return None
    normals = np.asarray([first['normal'],second['normal']],np.float64)
    direction = np.cross(*normals)
    sine = float(np.linalg.norm(direction))
    if sine <= .15:
        return None
    # Broad-phase finite adjacency uses coverage only, never GT or a field gate.
    lo=np.maximum(first['original_aabb_min_mm'],second['original_aabb_min_mm'])
    hi=np.minimum(first['original_aabb_max_mm'],second['original_aabb_max_mm'])
    reach=1.5*(first['coverage']['original_resolution_mm']+second['coverage']['original_resolution_mm'])+.3
    if np.linalg.norm(np.maximum(lo-hi,0.)) > reach:
        return None
    direction /= sine
    if direction[np.argmax(np.abs(direction))]<0:direction=-direction
    center=(np.asarray(first['centroid_mm'])+second['centroid_mm'])/2.
    offsets=np.array([first['plane_offset_mm'],second['plane_offset_mm']])-normals@center
    origin=center+normals.T@np.linalg.solve(normals@normals.T,offsets)
    ar,am=finite_line_runs(xyz,first_rows,first,origin,direction)
    br,bm=finite_line_runs(xyz,second_rows,second,origin,direction)
    overlaps=[]
    for ai,a in enumerate(ar):
        for bi,b in enumerate(br):
            low=max(a['t_min_mm'],b['t_min_mm']);high=min(a['t_max_mm'],b['t_max_mm'])
            if high<=low:continue
            witnesses=[]
            for face,run in ((first,a),(second,b)):
                row_ids=run['original_row_indices']
                along=(np.asarray(xyz[row_ids],np.float64)-origin)@direction
                numerical_slack=64*np.finfo(np.float64).eps*max(1.,float(np.abs(xyz[row_ids]).max()),float(np.abs(origin).max()))
                local=row_ids[(along>=low-numerical_slack)&(along<=high+numerical_slack)]
                witnesses.append(dict(face_id=face['face_id'],original_row_indices=local,
                    count=len(local),float64_comparison_slack_mm=numerical_slack,
                    maximum_run_line_distance_mm=run['maximum_line_distance_mm'],
                    maximum_run_along_line_gap_mm=run['maximum_along_line_gap_mm']))
            if any(w['count']<2 for w in witnesses):continue
            overlaps.append(dict(first_run=ai,second_run=bi,t_interval_mm=[low,high],
                endpoints_mm=np.stack((origin+low*direction,origin+high*direction)).tolist(),witnesses=witnesses))
    return dict(first_face=first['face_id'],second_face=second['face_id'],
        line_origin_mm=origin.tolist(),line_direction=direction.tolist(),normal_cross_norm=sine,
        line_fit_condition_number=float(np.linalg.cond(normals@normals.T)),
        first_coverage=am,second_coverage=bm,finite_bilateral_intervals=overlaps,
        status='BILATERAL_SAMPLED_LINE_SUPPORT' if overlaps else 'UNRESOLVED_FINITE_LINE_SUPPORT',
        estimated_fit_only_displacement_mm=(first['fit_residual_max_mm']+second['fit_residual_max_mm'])/sine,
        estimate_excludes_sampling_and_association_error=True,
        certified_boundary_error_bound_mm=None, field_valid=False,
        continuous_boundary_and_material_side_not_proved=True)
