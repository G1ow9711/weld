"""A common per-parent analytic correction inside a fixed 3 mm envelope.

LINE endpoints receive transverse corrections; every intermediate point is
their affine interpolation. CIRCLE receives one center/radius/rigid rotation.
Coordinate frames here belong to curves, never to signed board normals.
"""
import math
import numpy as np
from .contract_curve_geometry import CircularArc, sample_arc
from .process_geometry_constraints_327 import validate_component_geometry

VERSION = "COMMON_ANALYTIC_DIRECT_3MM_RESIDUAL_327_v1"
LIMIT_MM = 3.
NUMERICAL_INTERIOR = 1. - 1e-6


def line_transverse_basis(direction):
    axis = np.asarray(direction, np.float64)
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(axis, reference); first /= np.linalg.norm(first)
    return np.stack((first, np.cross(axis, first)), axis=0)


def _bounded_vector(value, radius):
    value = np.asarray(value, np.float64)
    norm = float(np.linalg.norm(value))
    return value * float(radius) * (math.tanh(norm) / norm if norm > 1e-8 else 1. - norm * norm / 3.)


def _rotation(omega):
    omega = np.asarray(omega, float)
    angle = float(np.linalg.norm(omega))
    x, y, z = omega
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    sinc = math.sin(angle) / angle if angle > 1e-5 else 1. - angle * angle / 6.
    cosc = (1. - math.cos(angle)) / (angle * angle) if angle > 1e-5 else .5 - angle * angle / 24.
    return np.eye(3) + sinc * skew + cosc * (skew @ skew)


class BoundedAnalyticGeometry:
    def __init__(self, carrier, raw_parameters):
        self.carrier = carrier
        raw = np.asarray(raw_parameters, np.float64)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise ValueError("One finite seven-component parent residual required")
        if carrier.kind == "line":
            basis = line_transverse_basis(carrier.direction)
            self.end_corrections = np.stack((_bounded_vector(raw[:2], LIMIT_MM * NUMERICAL_INTERIOR) @ basis,
                _bounded_vector(raw[2:4], LIMIT_MM * NUMERICAL_INTERIOR) @ basis))
            self.continuous_displacement_bound_mm = float(np.linalg.norm(self.end_corrections, axis=1).max())
            self.arc = None
        else:
            old = carrier.arc
            self.center_delta = _bounded_vector(raw[:3], NUMERICAL_INTERIOR)
            radius_budget = min(1., old.radius_mm / 2.) * NUMERICAL_INTERIOR
            self.radius_delta = float(np.tanh(raw[3])) * radius_budget
            angle_cap = 2. * math.asin(min(1., NUMERICAL_INTERIOR / (2. * old.radius_mm)))
            self.omega = _bounded_vector(raw[4:], angle_cap)
            rotation = _rotation(self.omega)
            self.arc = CircularArc(old.center_mm + self.center_delta, old.radius_mm + self.radius_delta,
                rotation @ old.plane_normal, rotation @ old.start_direction, old.sweep_rad, old.is_closed)
            self.continuous_displacement_bound_mm = float(np.linalg.norm(self.center_delta)) + abs(self.radius_delta) + 2. * old.radius_mm * math.sin(float(np.linalg.norm(self.omega)) / 2.)
        if self.continuous_displacement_bound_mm > LIMIT_MM:
            raise ValueError("Full finite curve residual exceeds the frozen 3 mm envelope")

    def point_at(self, q_mm):
        q = np.asarray(q_mm, np.float64)
        if not np.isfinite(q).all() or np.any(q < 0) or np.any(q > self.carrier.length_mm):
            raise ValueError("Correction cannot extend the original carrier domain")
        if self.carrier.kind == "line":
            fraction = q / self.carrier.length_mm
            return self.carrier.point_at(q) + (1. - fraction[..., None]) * self.end_corrections[0] + fraction[..., None] * self.end_corrections[1]
        # q stays in ORIGINAL physical arc-length units. Radius correction
        # changes delivered length, never changes the registered slot domains.
        corrected_positions = np.clip(q * self.arc.radius_mm / self.carrier.arc.radius_mm, 0., self.arc.length_mm)
        return self.arc.point_at(corrected_positions)

    def segment(self, q0, q1, first_point, last_point, is_closed=False):
        if q0 == q1:
            raise ValueError("Zero physical interval")
        if self.carrier.kind == "line":
            derivative = self.carrier.direction + (self.end_corrections[1] - self.end_corrections[0]) / self.carrier.length_mm
            length = abs(q1 - q0) * float(np.linalg.norm(derivative))
            return dict(kind="line", points=[first_point, last_point], length_mm=length)
        q0_new, q1_new = q0 * self.arc.radius_mm / self.carrier.arc.radius_mm, q1 * self.arc.radius_mm / self.carrier.arc.radius_mm
        direction = (np.asarray(first_point) - self.arc.center_mm) / self.arc.radius_mm
        arc = CircularArc(self.arc.center_mm, self.arc.radius_mm, self.arc.plane_normal, direction,
            (q1_new - q0_new) / self.arc.radius_mm * np.sign(self.arc.sweep_rad), bool(is_closed))
        segment = sample_arc(arc)
        segment["points"][0], segment["points"][-1] = first_point, last_point
        return segment


def assemble_corrected_parent(carrier, boundary_nodes, children, gaps, raw_residual, is_closed=False):
    geometry = BoundedAnalyticGeometry(carrier, raw_residual)
    nodes, slots = {}, set()
    for node in boundary_nodes:
        identity, slot, q = node["node_id"], int(node["slot_index"]), float(node["q_mm"])
        spec = carrier.slot_spec(slot)
        if identity in nodes or slot in slots or not spec["lower_mm"] <= q <= spec["upper_mm"]:
            raise ValueError("Repeated node/slot or parameter outside its declared slot")
        nodes[identity] = dict(q_mm=q, point_mm=geometry.point_at(q).tolist(), slot_index=slot)
        slots.add(slot)
    if not children or len(gaps) != len(children) - 1:
        raise ValueError("Explicit nonempty child references and adjacency required")
    segments, directions = [], []
    for child in children:
        start, end = nodes[child["start_node"]], nodes[child["end_node"]]
        q0, q1 = start["q_mm"], end["q_mm"]
        directions.append(float(np.sign(q1 - q0)))
        if q0 == q1 or (carrier.weld_type == "vertical" and (q1 <= q0 or end["point_mm"][2] >= start["point_mm"][2])):
            raise ValueError("Nonpositive child or vertical process is not downward")
        closed = bool(is_closed and len(children) == 1 and abs(q1 - q0) == carrier.length_mm)
        segment = geometry.segment(q0, q1, start["point_mm"], end["point_mm"], closed)
        if segment["length_mm"] < (5. if carrier.kind == "line" else 2.):
            raise ValueError("Predicted geometry below the process minimum")
        segment.update(weld_type=carrier.weld_type, start_boundary_reference=child["start_node"], end_boundary_reference=child["end_node"],
            carrier_ordinal=carrier.ordinal)
        segments.append(segment)
    if len(set(directions)) != 1:
        raise ValueError("One parent requires consistent native traversal")
    for index, gap in enumerate(gaps, 1):
        before, after = children[index - 1]["end_node"], children[index]["start_node"]
        if gap == "SHARED" and before != after:
            raise ValueError("SHARED requires one authority reference")
        if gap == "POSITIVE_GAP" and (before == after or directions[0] * (nodes[after]["q_mm"] - nodes[before]["q_mm"]) <= 0):
            raise ValueError("POSITIVE_GAP requires distinct forward authority references")
        if gap not in ("SHARED", "POSITIVE_GAP"):
            raise ValueError("Unknown predicted adjacency")
    mode = "LINE_CHAIN" if carrier.kind == "line" else "CIRCLE_CHAIN"
    authority = validate_component_geometry(mode, segments, gaps)
    return dict(mode=mode, child_count=len(segments), segments=segments, gaps=list(gaps), is_closed=bool(is_closed), boundary_nodes=nodes,
        common_analytic_authority=authority, raw_residual_parameters=list(map(float, raw_residual)),
        analytic_residual_continuous_displacement_bound_mm=geometry.continuous_displacement_bound_mm)


def differentiable_point_at(carrier, q_mm, raw_parameters):
    """Identical analytic map for positive-child geometry gradients, in mm."""
    import torch
    q, raw = q_mm.to(torch.float64), raw_parameters.to(torch.float64)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=q.device)

    def bounded(value, radius):
        norm = torch.linalg.vector_norm(value)
        ratio = torch.where(norm < 1e-6, 1. - norm.square() / 3., torch.tanh(norm) / norm.clamp(min=1e-12))
        return value * radius * ratio

    if carrier.kind == "line":
        basis = tensor(line_transverse_basis(carrier.direction))
        e0 = bounded(raw[:2], LIMIT_MM * NUMERICAL_INTERIOR) @ basis
        e1 = bounded(raw[2:4], LIMIT_MM * NUMERICAL_INTERIOR) @ basis
        fraction = q / carrier.length_mm
        return tensor(carrier.origin_mm) + q[..., None] * tensor(carrier.direction) + (1. - fraction[..., None]) * e0 + fraction[..., None] * e1
    old = carrier.arc
    dc = bounded(raw[:3], NUMERICAL_INTERIOR)
    dr = torch.tanh(raw[3]) * min(1., old.radius_mm / 2.) * NUMERICAL_INTERIOR
    omega = bounded(raw[4:], 2. * math.asin(min(1., NUMERICAL_INTERIOR / (2. * old.radius_mm))))
    theta = torch.linalg.vector_norm(omega)
    x, y, z = omega.unbind()
    zero = omega.new_zeros(())
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    sinc = torch.where(theta < 1e-5, 1. - theta.square() / 6., theta.sin() / theta.clamp(min=1e-12))
    cosc = torch.where(theta < 1e-5, .5 - theta.square() / 24., (1. - theta.cos()) / theta.square().clamp(min=1e-24))
    rotation = torch.eye(3, dtype=torch.float64, device=q.device) + sinc * skew + cosc * (skew @ skew)
    u, v = rotation @ tensor(old.start_direction), rotation @ tensor(np.cross(old.plane_normal, old.start_direction))
    angle = q / old.radius_mm * np.sign(old.sweep_rad)
    return tensor(old.center_mm) + dc + (old.radius_mm + dr) * (angle.cos()[..., None] * u + angle.sin()[..., None] * v)
