"""Offline native verification of two-plate material candidates, not accuracy.

Inspect signed normals at actual finite surfaces and same-body front/back
witnesses. Preserve board correspondence and do not use abs(dot) for E4.
This module neither changes training labels nor certifies a complete curve.
"""
from __future__ import annotations
import math
import gmsh
import numpy as np
from weld_pipeline.finite_side_plate_evidence_306 import finite_face_point

VERSION = "OFFLINE_NATIVE_ORDERED_MATERIAL_POINT_REVIEW_308_v1"
DISTANCE_SUPPORT_MM = .2
THICKNESS_ERROR_MM = 1.0
SIGNED_NORMAL_DEGREES = 3.0
UNIT_LENGTH_ERROR = 1e-3


def xyz(value):
    result = np.asarray(value, float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Three finite coordinates required")
    return result


class NativeMaterialGeometry:
    """Fresh native queries; cached only within one unchanged STEP import."""
    def __init__(self, catalogue):
        self.volumes = {v["volume"]: v for v in catalogue["volumes"]}
        self.boundaries = {}
        self.faces = {}

    def face(self, volume, tag):
        key = (volume, tag)
        if key in self.faces:
            return self.faces[key]
        if volume not in self.boundaries:
            self.boundaries[volume] = {abs(t) for d, t in gmsh.model.getBoundary(
                [(3, volume)], combined=False, oriented=False) if d == 2}
        if tag not in self.boundaries[volume]:
            raise ValueError("Candidate face does not belong to the claimed material")
        center = xyz(gmsh.model.occ.getCenterOfMass(2, tag))
        lower, upper = gmsh.model.getParametrizationBounds(2, tag)
        result = dict(face=tag, volume=volume, center_mm=center.tolist(),
                      normal=None, verified=False, native_type=gmsh.model.getType(2, tag))
        for fraction in (.5, .3, .7, .15, .85):
            uv = np.asarray(lower) + fraction * (np.asarray(upper) - lower)
            if not gmsh.model.isInside(2, tag, uv.tolist(), parametric=True):
                continue
            point = xyz(gmsh.model.getValue(2, tag, uv.tolist()))
            raw = xyz(gmsh.model.getNormal(tag, uv.tolist()))
            length = float(np.linalg.norm(raw))
            if length <= 1e-12:
                continue
            unit = raw / length
            plus = bool(gmsh.model.isInside(3, volume, (point + .01 * unit).tolist()))
            minus = bool(gmsh.model.isInside(3, volume, (point - .01 * unit).tolist()))
            if plus != minus:
                result.update(normal=(unit if minus else -unit).tolist(), verified=True,
                    native_interior_point_mm=point.tolist(), native_normal_parameter=uv.tolist(),
                    native_parametric_orientation_sign=1 if minus else -1,
                    plus_in_material=plus, minus_in_material=minus)
                break
        self.faces[key] = result
        return result

    def review_plate(self, plate, local_point, rotation):
        volume, tag = int(plate["volume"]), int(plate["face"])
        native = self.face(volume, tag)
        thickness = float(plate["thickness_candidate_mm"])
        label_normal = xyz(plate["outward_normal_world"])
        label_length = float(np.linalg.norm(label_normal))
        if not math.isfinite(thickness) or thickness <= 0 or label_length <= 1e-12:
            raise ValueError("Positive thickness and nonzero signed normal required")
        finite = finite_face_point(tag, local_point)
        unit_pass = abs(label_length - 1) <= UNIT_LENGTH_ERROR
        support = bool(finite["native_finite_witness"] and
                       finite["distance_upper_bound_mm"] <= DISTANCE_SUPPORT_MM)
        output = dict(volume=volume, face=tag, candidate_thickness_mm=thickness,
            candidate_normal_world=label_normal.tolist(), candidate_unit_length_error=abs(label_length - 1),
            unit_length_pass=unit_pass, finite_contact_witness=finite, finite_contact_supported=support,
            native_normal_witness=native, native_material_outward_direction_verified=native["verified"],
            signed_normal_angle_degrees=None, signed_normal_pass=False,
            opposed_finite_face_witnesses=[], same_solid_probe_replayed=False,
            candidate_point_eligible=False)
        if not native["verified"]:
            output["state"] = "NATIVE_SIGNED_FACE_NORMAL_UNRESOLVED"
            return output
        normal = xyz(native["normal"])
        contact = xyz(finite["point_mm"]) if finite["native_finite_witness"] else local_point
        uv = np.asarray(gmsh.model.getParametrization(2, tag, contact.tolist()), float)
        if uv.shape != (2,) or not np.isfinite(uv).all() or np.linalg.norm(xyz(gmsh.model.getValue(2,tag,uv.tolist()))-contact)>1e-6:
            output["state"] = "NATIVE_FINITE_CONTACT_NORMAL_PARAMETER_UNRESOLVED"
            return output
        contact_raw = xyz(gmsh.model.getNormal(tag, uv.tolist()))
        contact_length = float(np.linalg.norm(contact_raw))
        if contact_length <= 1e-12:
            output["state"] = "NATIVE_FINITE_CONTACT_ZERO_NORMAL"
            return output
        contact_normal = contact_raw / contact_length * native["native_parametric_orientation_sign"]
        native_world = rotation @ contact_normal
        cosine = float(np.clip(label_normal @ native_world / label_length, -1, 1))
        angle = float(np.rad2deg(np.arccos(cosine)))
        output.update(native_outward_normal_world=native_world.tolist(), signed_normal_angle_degrees=angle,
                      signed_normal_pass=angle <= SIGNED_NORMAL_DEGREES,
                      actual_native_contact_normal_parameter=uv.tolist())

        stored = plate["same_solid_thickness_witness"]
        front = xyz(stored["probe_front_point_CAD_mm"])
        front_finite = finite_face_point(tag, front)
        front_on_finite_face = bool(front_finite["native_finite_witness"] and
                                   front_finite["distance_upper_bound_mm"] <= 1e-6)
        locations = [front + .01 * normal, front - .01 * normal,
                     front - (thickness - .01) * normal, front - (thickness + .01) * normal]
        membership = [bool(gmsh.model.isInside(3, volume, p.tolist())) for p in locations]
        probe_pass = bool(front_on_finite_face and membership == [False, True, True, False]
                          and stored["membership"] == membership
                          and abs(stored["plane_separation_mm"] - thickness) <= 1e-7)
        output.update(same_solid_probe_replayed=probe_pass, actual_probe_membership=membership,
                      stored_front_probe_finite_witness=front_finite,
                      stored_probe_scope="NEAR_CONTACT_ONLY_NOT_FULL_CURVE_CERTIFICATE")
        record = self.volumes[volume]
        for pair in record["parallel_major_face_pairs"]:
            if tag not in pair["faces"] or pair["inertia_estimate_error_mm"] > .05:
                continue
            opposite_tag = next(t for t in pair["faces"] if t != tag)
            opposite = self.face(volume, opposite_tag)
            if not opposite["verified"]:
                continue
            opposing_cosine = float(normal @ xyz(opposite["normal"]))
            separation = -float((xyz(opposite["center_mm"]) - xyz(native["center_mm"])) @ normal)
            if separation <= 0 or opposing_cosine > -math.cos(math.radians(SIGNED_NORMAL_DEGREES)):
                continue
            back = front - separation * normal
            back_finite = finite_face_point(opposite_tag, back)
            opposed_support = bool(back_finite["native_finite_witness"] and
                                   back_finite["distance_upper_bound_mm"] <= 1e-6)
            actual_membership = [bool(gmsh.model.isInside(3, volume, p.tolist())) for p in
                (front + .01 * normal, front - .01 * normal,
                 back + .01 * normal, back - .01 * normal)]
            error = abs(separation - thickness)
            valid = bool(opposed_support and actual_membership == [False, True, True, False]
                         and error <= THICKNESS_ERROR_MM)
            output["opposed_finite_face_witnesses"].append(dict(opposed_face=opposite_tag,
                native_plane_separation_mm=separation, candidate_thickness_error_mm=error,
                opposite_material_outward_dot=opposing_cosine, finite_opposed_face_witness=back_finite,
                front_back_membership=actual_membership, thickness_1mm_and_same_body_pass=valid))
        thickness_pass = any(w["thickness_1mm_and_same_body_pass"] for w in output["opposed_finite_face_witnesses"])
        output.update(native_thickness_pass=thickness_pass,
            candidate_point_eligible=bool(record["constant_thickness_supported"] and support and unit_pass
                and output["signed_normal_pass"] and probe_pass and thickness_pass),
            state="NATIVE_POINT_CANDIDATE_WITNESS_REVIEW_NOT_CUSTOMER_TRUTH")
        return output

    def review_pair(self, sample, directed_tangent, transform):
        transform = np.asarray(transform, float)
        if transform.shape != (4,4) or not np.isfinite(transform).all():
            raise ValueError("Finite 4x4 CAD-to-world transform required")
        rotation, translation = transform[:3, :3], transform[:3, 3]
        if transform.shape != (4, 4) or np.max(np.abs(rotation.T @ rotation - np.eye(3))) > 1e-6 or abs(np.linalg.det(rotation)-1) > 1e-6:
            raise ValueError("Proper rigid CAD-to-world transform required")
        if not sample["unique_pair"] or len(sample["associated_plate_options"]) != 1:
            raise ValueError("Review only an explicitly unique two-material candidate")
        pair = sample["associated_plate_options"][0]
        if len(pair) != 2 or pair[0]["volume"] == pair[1]["volume"]:
            raise ValueError("Two distinct associated materials required")
        world = xyz(sample["xyz_world_mm"])
        local = (world - translation) @ rotation
        tangent = xyz(directed_tangent)
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1e-12:
            raise ValueError("Nonzero directed tangent required")
        tangent /= tangent_length
        reviews = [self.review_plate(p, local, rotation) for p in pair]
        normal = [xyz(p["outward_normal_world"]) for p in pair]
        orientation = float(np.cross(normal[0], normal[1]) @ tangent)
        fixed_order = orientation > 1e-8
        return dict(version=VERSION, fraction=sample["fraction"], xyz_world_mm=world.tolist(),
            xyz_CAD_mm=local.tolist(), original_ordered_volume_ids=[p["volume"] for p in pair],
            signed_plate_order_orientation=orientation, original_plate_correspondence_pass=fixed_order,
            plates=reviews, candidate_point_eligible=bool(fixed_order and all(p["candidate_point_eligible"] for p in reviews)),
            original_plate_sides_permuted=False, normal_abs_dot_used=False,
            source_label_modified=False, customer_reviewed=False, whole_curve_certificate=False,
            model_accuracy_measured=False, STEP_is_runtime_input=False)
