"""Native finite-geometry evidence for offline process-label review only.

Gmsh getClosestPoint projects onto the supporting entity and may leave its
trim: https://gmsh.info/doc/texinfo/gmsh.html#gmsh_002fmodel_002fgetClosestPoint
Keep actual trim and boundary witnesses; never label an infinite-plane hit
as a weld or infer the PDF's side-specific flag solely from an obstruction.
"""
from __future__ import annotations

import math
import gmsh
import numpy as np

VERSION = "NATIVE_FINITE_LOCAL_SIDE_EVIDENCE_306"


def _xyz(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Finite XYZ required")
    return value


def finite_face_point(face_tag, point):
    """Return a witnessed finite point and distance upper bound.

    An interior planar projection is globally closest for a planar face.
    Boundary alternatives are valid finite witnesses but do not certify a
    global minimum on arbitrary B-splines. No surface meshing is required.
    """
    point = _xyz(point)
    projected, uv = gmsh.model.getClosestPoint(2, face_tag, point.tolist())
    projected = _xyz(projected)
    uv = np.asarray(uv,float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or np.linalg.norm(_xyz(gmsh.model.getValue(2,face_tag,uv.tolist()))-projected)>1e-6:
        uv = np.asarray(gmsh.model.getParametrization(2,face_tag,projected.tolist()),float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or np.linalg.norm(_xyz(gmsh.model.getValue(2,face_tag,uv.tolist()))-projected)>1e-6:
        return dict(point_mm=None,distance_upper_bound_mm=None,native_finite_witness=False,
                    method="SURFACE_PROJECTION_PARAMETER_INCONSISTENT",supporting_surface_projection_inside_trim=False,
                    globally_closest_if_planar=False)
    inside = bool(gmsh.model.isInside(2, face_tag, list(uv), parametric=True))
    if inside:
        return dict(point_mm=projected.tolist(), distance_upper_bound_mm=float(np.linalg.norm(projected-point)),
                    native_finite_witness=True, method="NATIVE_PROJECTION_INSIDE_TRIM",
                    supporting_surface_projection_inside_trim=True,
                    globally_closest_if_planar=True)
    alternatives = []
    errors = []
    boundary = gmsh.model.getBoundary([(2, face_tag)], combined=False, oriented=False)
    for dimension, curve in boundary:
        if dimension != 1:
            continue
        try:
            lo, hi = gmsh.model.getParametrizationBounds(1, curve)
            lower, upper = float(lo[0]), float(hi[0])
            if not math.isfinite(lower+upper) or upper < lower:
                raise ValueError("Nonfinite curve bounds")
            parameters = [lower, upper]
            curve_projection, returned_u = gmsh.model.getClosestPoint(1, curve, point.tolist())
            curve_projection = _xyz(curve_projection)
            # Native OCC can return correct XYZ with an uninitialized curve
            # parameter. Reparameterize XYZ and verify the forward map;
            # do not silently reconstruct a different boundary point.
            checked_u = gmsh.model.getParametrization(1,curve,curve_projection.tolist())
            if len(checked_u) == 1 and math.isfinite(float(checked_u[0])):
                reconstructed = _xyz(gmsh.model.getValue(1,curve,list(checked_u)))
                if np.linalg.norm(reconstructed-curve_projection)>1e-6:
                    raise ValueError("Curve XYZ reparameterization failed round-trip check")
                parameter = float(checked_u[0])
                parameters.append(float(np.clip(parameter, lower, upper)))
                if gmsh.model.getType(1, curve) in ("Circle", "Ellipse"):
                    for turn in range(-2, 3):
                        adjusted = parameter+2*math.pi*turn
                        if lower <= adjusted <= upper:
                            parameters.append(adjusted)
            for parameter in parameters:
                actual = _xyz(gmsh.model.getValue(1, curve, [parameter]))
                on_curve = bool(gmsh.model.isInside(1, curve, [parameter], parametric=True))
                if on_curve:
                    alternatives.append((float(np.linalg.norm(actual-point)), actual, curve, parameter))
            # Explicit topological vertices remain valid even when a native
            # endpoint parameter is rejected by an entity's trim tolerance.
            for dim, vertex in gmsh.model.getBoundary([(1, curve)], combined=False, oriented=False):
                if dim == 0:
                    actual = _xyz(gmsh.model.getValue(0, vertex, []))
                    alternatives.append((float(np.linalg.norm(actual-point)), actual, curve, None))
        except Exception as error:
            errors.append(dict(curve=curve, exception=type(error).__name__, message=str(error)))
    if not alternatives:
        return dict(point_mm=None, distance_upper_bound_mm=None, native_finite_witness=False,
                    method="NO_SUPPORTED_FINITE_POINT", supporting_surface_projection_inside_trim=False,
                    boundary_errors=errors, globally_closest_if_planar=False)
    distance, actual, curve, parameter = min(alternatives, key=lambda item: item[0])
    return dict(point_mm=actual.tolist(), distance_upper_bound_mm=distance,
                native_finite_witness=True, method="NATIVE_TRIM_BOUNDARY_POINT",
                supporting_surface_projection_inside_trim=False,
                boundary_curve=curve, boundary_parameter=parameter,
                boundary_errors=errors, globally_closest_if_planar=False)


def signed_face_normal(volume_tag, face):
    normal = _xyz(face["parametric_normal"])
    normal /= np.linalg.norm(normal)
    lower, upper = gmsh.model.getParametrizationBounds(2, face["face"])
    for fraction in (.5, .3, .7, .15, .85):
        uv = np.asarray(lower)+fraction*(np.asarray(upper)-lower)
        if not gmsh.model.isInside(2, face["face"], uv.tolist(), parametric=True):
            continue
        point = _xyz(gmsh.model.getValue(2, face["face"], uv.tolist()))
        plus = bool(gmsh.model.isInside(3, volume_tag, (point+.01*normal).tolist()))
        minus = bool(gmsh.model.isInside(3, volume_tag, (point-.01*normal).tolist()))
        if plus != minus:
            return dict(normal=(normal if minus else -normal).tolist(), verified=True,
                        interior_face_point_mm=point.tolist(), plus_in_material=plus, minus_in_material=minus)
    return dict(normal=None, verified=False)


def line_neighbor_events(points, associated_volumes, free_side_normals, faces,
                         near_threshold_mm=10.0, numerical_tolerance_mm=1e-6):
    """Locate finite crossing-face witnesses relative to a known weld path.

    Associated materials and their signed normals are hypotheses supplied
    by a separate material audit. Events are offline candidates, not new
    truth, not a complete cutter/torch model, and not side-plate true flags.
    """
    points = np.asarray(points, float)
    if points.shape != (2,3) or not np.isfinite(points).all():
        raise ValueError("Two finite line endpoints required")
    threshold = float(near_threshold_mm)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Positive configurable threshold required")
    normals = np.asarray(free_side_normals, float)
    if normals.shape != (2,3) or not np.isfinite(normals).all() or np.max(np.abs(np.linalg.norm(normals,axis=1)-1)) > 1e-3:
        raise ValueError("Two signed unit normals required")
    direction = points[1]-points[0]
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        raise ValueError("Positive line length required")
    tangent = direction/length
    excluded = set(associated_volumes)
    events = []
    for face in faces:
        if face["volume"] in excluded:
            continue
        normal = _xyz(face["outward_normal"])
        divisor = float(normal@tangent)
        if abs(divisor) <= 1e-8:
            continue
        station = float(normal@(_xyz(face["center_mm"])-points[0]))/divisor
        if station < -numerical_tolerance_mm or station > length+numerical_tolerance_mm:
            continue
        station = float(np.clip(station,0,length))
        point = points[0]+station*tangent
        bounds = np.asarray(face["bbox_mm"],float)
        lower_bound = float(np.linalg.norm(np.maximum(np.maximum(bounds[:3]-point,point-bounds[3:]),0)))
        if lower_bound > threshold+numerical_tolerance_mm:
            continue
        closest = finite_face_point(face["face"],point)
        if not closest["native_finite_witness"]:
            events.append(dict(volume=face["volume"], face=face["face"], station_mm=station,
                               state="FINITE_BOUNDARY_QUERY_UNRESOLVED", closest=closest))
            continue
        displacement = _xyz(closest["point_mm"])-point
        signed_offsets = normals@displacement
        distance = closest["distance_upper_bound_mm"]
        margin = threshold-distance
        wedge = bool(np.min(signed_offsets)>=-numerical_tolerance_mm)
        below = margin > numerical_tolerance_mm
        material_normal = normal/np.linalg.norm(normal)
        surface_point = _xyz(closest["point_mm"])
        plus = bool(gmsh.model.isInside(3,face["volume"],(surface_point+.01*material_normal).tolist()))
        minus = bool(gmsh.model.isInside(3,face["volume"],(surface_point-.01*material_normal).tolist()))
        events.append(dict(volume=face["volume"], face=face["face"], station_mm=station,
            xyz_CAD_mm=point.tolist(), closest=closest, finite_bbox_distance_lower_bound_mm=lower_bound,
            candidate_boundary_in_free_wedge=wedge, signed_free_side_offsets_mm=signed_offsets.tolist(),
            near_threshold_mm=threshold, distance_strictly_below_threshold_supported=below,
            threshold_margin_mm=margin, near_equality_requires_review=abs(margin)<=numerical_tolerance_mm,
            neighbor_boundary_material_witness=dict(plus_in_material=plus,minus_in_material=minus,
                material_interior_supported=minus and not plus),
            possible_local_process_cut=wedge and below and minus and not plus,
            at_original_segment_endpoint=station<=numerical_tolerance_mm or length-station<=numerical_tolerance_mm,
            side_plate_split_truth=None, original_weld_geometry_changed=False))
    stations = sorted({0.0,length,*[e["station_mm"] for e in events]})
    occupied = []
    for a,b in zip(stations,stations[1:]):
        if b-a <= numerical_tolerance_mm:
            continue
        middle = points[0]+((a+b)/2)*tangent
        materials = sorted({face["volume"] for face in faces if face["volume"] not in excluded
                            and gmsh.model.isInside(3,face["volume"],middle.tolist())})
        if materials:
            occupied.append(dict(start_station_mm=a,end_station_mm=b,midpoint_inside_volumes=materials,
                                 scope="NATIVE_MIDPOINT_OCCUPANCY_NOT_FULL_INTERVAL_CERTIFICATE"))
    return dict(version=VERSION,events=events,occupied_interval_midpoint_witnesses=occupied,
                complete_finite_geometry_coverage_certified=False,
                side_plate_split_truth_inferred=False, customer_reviewed=False,
                STEP_is_runtime_input=False, original_weld_geometry_changed=False)
