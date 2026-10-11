"""Offline whole-line material hypotheses from finite, original CAD solids.

These are auxiliary geometry witnesses, not customer-approved attribute truth.
No source names, labels, neural weights or pointwise auxiliary values select
faces or thicknesses. Positive missing intervals are never closed by tolerance.
"""
import math

import gmsh
import numpy as np

from .continuous_step_support_273 import (
    CadQueryError, FiniteStepSupport, native, union_and_missing,
)
from .customer_process_rules_302 import order_contact_plates


VERSION = "OFFLINE_FINITE_ALL_PLANAR_PAIRS_ALL_BACK_PLANES_WHOLE_LINE_341_v1"


class WholeLineMaterial(FiniteStepSupport):
    def _thickness_options(self, front):
        normal = np.asarray(front["fresh_material_normal"]["normal_CAD"])
        center = np.asarray(front["center_mm"])
        cosine = math.cos(math.radians(self.tolerances["coplanar_normal_group_degrees"]))
        options = []
        for other in self.by_volume[front["volume"]]:
            sign = other["fresh_material_normal"]
            if not sign["verified"]:
                continue
            opposite = np.asarray(sign["normal_CAD"])
            depth = -float((np.asarray(other["center_mm"]) - center) @ normal)
            if depth > 0 and opposite @ normal <= -cosine:
                # Keep every actual opposite-face hypothesis and its identity.
                # There is no whole-body inertia assumption or upper-thickness cap.
                options.append(dict(volume=front["volume"], face=front["face"],
                    board_name=front["board_name"], opposed_seed_face=other["face"],
                    thickness_candidate_mm=depth,
                    outward_normal_world=(self.rotation @ normal).tolist()))
        return options

    def enumerate_hypotheses(self, original_world_endpoints):
        points = np.asarray(original_world_endpoints, dtype=np.float64)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError("Two finite original line endpoints are required")
        directed = points[1] - points[0]
        length = float(np.linalg.norm(directed))
        if length <= 0:
            raise ValueError("A positive original line is required")
        local = (points - self.translation) @ self.rotation
        local_tangent = (directed / length) @ self.rotation
        front_options, excluded = [], []
        band = self.tolerances["reference_contact_projection_band_mm"]
        for face in self.faces.values():
            sign = face["fresh_material_normal"]
            if not sign["verified"]:
                excluded.append(dict(volume=face["volume"], face=face["face"],
                    reason="NATIVE_MATERIAL_DIRECTION_UNVERIFIED"))
                continue
            normal = np.asarray(sign["normal_CAD"])
            center = np.asarray(face["center_mm"])
            # This bound only excludes a plane that cannot be within the frozen
            # contact band of the entire original line. It never bridges a gap.
            endpoint_distances = np.abs((local - center) @ normal)
            if np.max(endpoint_distances) > band:
                continue
            bounds = np.asarray(face["bbox_mm"])
            if np.any(np.maximum(local[0], local[1]) < bounds[:3] - band) or np.any(
                    np.minimum(local[0], local[1]) > bounds[3:] + band):
                continue
            if abs(float(local_tangent @ normal)) > math.sin(math.radians(3.0)):
                continue
            for board in self._thickness_options(face):
                front_options.append(board)
        hypotheses = []
        for ai, a in enumerate(front_options):
            for b in front_options[ai + 1:]:
                if a["volume"] == b["volume"]:
                    continue
                normals = np.asarray([a["outward_normal_world"], b["outward_normal_world"]])
                if np.linalg.det(normals @ normals.T) < .1:
                    continue
                pair = order_contact_plates([a, b], directed)
                hypotheses.append(dict(ordered_boards=pair,
                    order_basis="SIGNED_NORMAL_CROSS_PRODUCT_DOT_ORIGINAL_DIRECTED_TANGENT",
                    original_world_endpoints_mm=points.tolist(),
                    actual_opposite_face_aliases_retained=True))
        return dict(original_world_endpoints_mm=points.tolist(),
            original_length_mm=length, candidate_front_back_options=front_options,
            unverified_native_faces=excluded, hypotheses=hypotheses,
            pointwise_auxiliary_labels_read=False)

    def _solid_line_overlap(self, a, b, volume):
        delta = b - a
        length = float(np.linalg.norm(delta))
        bounds = np.asarray(native(gmsh.model.getBoundingBox, 3, volume))
        pad = self.tolerances["bbox_exclusion_padding_mm"]
        if np.any(np.maximum(a, b) < bounds[:3] - pad) or np.any(
                np.minimum(a, b) > bounds[3:] + pad):
            return dict(volume=volume, bbox_excluded=True, occupied_intervals=[],
                zero_dimensional_intersections=0, strict_clear=True)
        pa = native(gmsh.model.occ.addPoint, *map(float, a))
        pb = native(gmsh.model.occ.addPoint, *map(float, b))
        query = native(gmsh.model.occ.addLine, pa, pb)
        copied = native(gmsh.model.occ.copy, [(3, volume)])
        common, _ = native(gmsh.model.occ.intersect, [(1, query)], copied)
        native(gmsh.model.occ.synchronize)
        intervals, witnesses, isolated = [], [], 0
        for dim, tag in common:
            if dim == 0:
                isolated += 1
                continue
            if dim != 1:
                raise CadQueryError("Unexpected dimension in line/solid intersection")
            lower, upper = native(gmsh.model.getParametrizationBounds, 1, tag)
            points = np.asarray(native(gmsh.model.getValue, 1, tag,
                [float(lower[0]), float(upper[0])])).reshape(2, 3)
            off_axis = np.linalg.norm(np.cross(points - a, delta / length), axis=1)
            if np.max(off_axis) > self.tolerances["native_line_axis_consistency_mm"]:
                raise CadQueryError("Finite occupancy intersection leaves its query line")
            mass = float(native(gmsh.model.occ.getMass, 1, tag))
            if abs(mass - np.linalg.norm(points[1] - points[0])) > self.tolerances[
                    "native_line_length_consistency_mm"]:
                raise CadQueryError("Finite occupancy curve length disagrees with endpoints")
            interval = sorted(map(float, (points - a) @ delta / length**2))
            intervals.append(interval)
            witnesses.append(dict(interval=interval, length_mm=mass,
                endpoints_CAD_mm=points.tolist()))
        occupied, _ = union_and_missing(intervals)
        return dict(volume=volume, bbox_excluded=False, occupied_intervals=occupied,
            zero_dimensional_intersections=isolated, native_intersections=witnesses,
            strict_clear=not occupied and isolated == 0)

    def check_hypothesis(self, original_world_endpoints, hypothesis):
        pair = hypothesis["ordered_boards"]
        target = dict(kind="line", original_reference_points=original_world_endpoints)
        result = self.child(target, pair)
        result["ordered_boards"] = pair
        result["order_basis"] = hypothesis["order_basis"]
        if result.get("strict_complete_geometric_candidate_support"):
            joint = np.asarray(result["joint_reference_endpoints_CAD_mm"])
            normals = np.asarray([self.rotation.T @ np.asarray(b["outward_normal_world"])
                                  for b in pair])
            direction = normals.sum(axis=0)
            direction /= np.linalg.norm(direction)
            offset = .1 * direction
            occupancy = []
            for volume in self.volume_mass:
                rec = self._solid_line_overlap(joint[0] + offset, joint[1] + offset, volume)
                occupancy.append(rec)
                self.cleanup_queries()
            result["whole_external_probe"] = dict(offset_CAD_mm=offset.tolist(),
                query_endpoints_CAD_mm=(joint + offset).tolist(),
                original_solids=occupancy, strict_clear=all(x["strict_clear"] for x in occupancy),
                scope="ENTIRE_FINITE_0P1MM_EXTERNAL_BISECTOR_LINE_NOT_GLOBAL_ACCESSIBILITY")
        result["strict_complete_contact_and_external_probe"] = bool(
            result.get("strict_complete_geometric_candidate_support") and
            result.get("whole_external_probe", {}).get("strict_clear"))
        result.update(original_geometry_preserved=True, customer_reviewed=False,
            approved_whole_E_truth=False, runtime_STEP_allowed=False,
            current_neural_training_labels_changed=False)
        return result


def compare_preserved_ordered_samples(hypothesis, auxiliary_samples):
    """Compare both plates in their original order; never abs(dot) or swap."""
    pair = hypothesis["ordered_boards"]
    errors, angles, unit_errors = [], [], []
    for row in auxiliary_samples:
        sample = row["original_sample"]
        thickness = np.asarray(sample["thickness_mm"], dtype=float)
        normals = np.asarray(sample["normals"], dtype=float)
        if thickness.shape != (2,) or normals.shape != (2, 3) or not np.isfinite(
                thickness).all() or not np.isfinite(normals).all():
            raise ValueError("Preserved two-plate sample payload is invalid")
        for i, plate in enumerate(pair):
            norm = float(np.linalg.norm(normals[i]))
            if norm <= 0:
                raise ValueError("Preserved signed normal is zero")
            expected = np.asarray(plate["outward_normal_world"])
            expected_norm = float(np.linalg.norm(expected))
            errors.append(abs(float(thickness[i]) - plate["thickness_candidate_mm"]))
            angles.append(math.degrees(math.acos(float(np.clip(
                normals[i] @ expected / (norm * expected_norm), -1., 1.)))))
            unit_errors.extend([abs(norm - 1.), abs(expected_norm - 1.)])
    return dict(original_samples_compared=len(auxiliary_samples),
        original_sample_order_preserved=True, plate_order_exchanged=False,
        abs_dot_used=False, maximum_thickness_error_mm=max(errors, default=None),
        maximum_signed_normal_angle_degrees=max(angles, default=None),
        maximum_normal_unit_length_error=max(unit_errors, default=None),
        all_preserved_point_values_consistent=bool(auxiliary_samples and
            max(errors) <= 1. and max(angles) <= 3. and max(unit_errors) <= 1e-3),
        formal_truth_approved=False)
