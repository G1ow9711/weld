"""Finite analytic carriers, two lazy boundary slots per <=1 mm cell.

This runtime representation receives pure XYZ proposals and selected boundary
references. Labels, CAD, source identifiers, GT counts and searches are absent.
Parameter slots are fixed before supervision. Serialization aliases add no
capacity. Independent parents never share a boundary by coordinate proximity.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
from .contract_curve_geometry import CircularArc, sample_arc
from .process_geometry_constraints_327 import validate_component_geometry

VERSION = "FINITE_ANALYTIC_LAZY_DOUBLE_BOUNDARY_SLOTS_327_v1"
CELL_LIMIT_MM = 1.
INDEPENDENT_SLOTS_PER_CELL = 2
SPATIAL_RESIDUAL_ENVELOPE_MM = 3.


def exact_geometry_key(record):
    """Exact normalized serialization identity; never a proximity NMS."""
    primitive = record["segments"][0]
    if primitive["kind"] == "line":
        endpoints = np.asarray(primitive["points"], np.float64)[[0, -1]]
        if tuple(endpoints[1]) < tuple(endpoints[0]):
            endpoints = endpoints[::-1]
        geometry = dict(kind="line", endpoints_mm=endpoints.tolist())
    else:
        arc = CircularArc.from_parameters(primitive["arc_parameters"])
        geometry = dict(kind="arc", parameters=arc.to_parameters())
    body = dict(weld_type=record["type"], geometry=geometry)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Carrier:
    ordinal: int
    weld_type: str
    kind: str
    length_mm: float
    origin_mm: object
    direction: object
    arc: object
    geometry_identity: str
    aliases: tuple
    structural_events_mm: tuple = ()

    @classmethod
    def from_record(cls, ordinal, record, aliases=()):
        if record["type"] not in ("horizontal", "vertical") or len(record["segments"]) != 1:
            raise ValueError("One finite carrier with a known process category required")
        segment = record["segments"][0]
        if segment["kind"] == "line":
            points = np.asarray(segment["points"], dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError("Finite LINE points required")
            endpoints = points[[0, -1]].copy()
            delta = endpoints[1] - endpoints[0]
            if (record["type"] == "vertical" and delta[2] > 0) or (record["type"] == "horizontal" and tuple(endpoints[1]) < tuple(endpoints[0])):
                endpoints = endpoints[::-1].copy()
                delta = endpoints[1] - endpoints[0]
            length = float(np.linalg.norm(delta))
            if not math.isfinite(length) or length <= 0 or (record["type"] == "vertical" and delta[2] >= 0):
                raise ValueError("Nondegenerate LINE; vertical process must point down")
            arc, origin, direction = None, endpoints[0], delta / length
        elif segment["kind"] == "arc":
            arc = CircularArc.from_parameters(segment["arc_parameters"])
            length, origin, direction = arc.length_mm, None, None
        else:
            raise ValueError("Unsupported carrier kind stays explicitly unsupported")
        events = tuple(sorted(set(float(q) for q in record.get("structural_events_mm", ()))))
        if any(not math.isfinite(q) or not 0 <= q <= length for q in events):
            raise ValueError("Structure event must lie on the finite carrier")
        return cls(int(ordinal), record["type"], segment["kind"], length, origin, direction, arc,
            exact_geometry_key(record), tuple(aliases or (record["id"],)), events)

    @property
    def cell_count(self):
        return int(math.ceil(self.length_mm / CELL_LIMIT_MM))

    @property
    def cell_width_mm(self):
        return self.length_mm / self.cell_count

    @property
    def slot_count(self):
        return INDEPENDENT_SLOTS_PER_CELL * self.cell_count + 2 + len(self.structural_events_mm)

    def slot_spec(self, slot_index):
        if isinstance(slot_index, bool) or not isinstance(slot_index, (int, np.integer)) or not 0 <= slot_index < self.slot_count:
            raise ValueError("Declared finite boundary slot required")
        regular = INDEPENDENT_SLOTS_PER_CELL * self.cell_count
        if slot_index < regular:
            cell, subslot = divmod(int(slot_index), INDEPENDENT_SLOTS_PER_CELL)
            lo = cell * self.cell_width_mm
            hi = self.length_mm if cell == self.cell_count - 1 else (cell + 1) * self.cell_width_mm
            return dict(kind="CELL", cell_index=cell, independent_subslot=subslot, lower_mm=lo, upper_mm=hi)
        q = (0., self.length_mm)[slot_index - regular] if slot_index < regular + 2 else self.structural_events_mm[slot_index - regular - 2]
        return dict(kind="OBSERVED_FINITE_ENDPOINT" if slot_index < regular + 2 else "STRUCTURAL_EVENT",
            lower_mm=q, upper_mm=q)

    def addresses_at(self, q_mm):
        q = float(q_mm)
        if not math.isfinite(q) or not 0 <= q <= self.length_mm:
            raise ValueError("Boundary parameter outside original finite domain")
        cell = min(self.cell_count - 1, int(math.floor(q / self.cell_width_mm)))
        result = [2 * cell, 2 * cell + 1]
        end = 2 * self.cell_count
        if q == 0.:
            result.insert(0, end)
        if q == self.length_mm:
            result.insert(0, end + 1)
        for index, event in enumerate(self.structural_events_mm):
            if q == event:
                result.insert(0, end + 2 + index)
        return result

    def point_at(self, q_mm):
        q = np.asarray(q_mm, dtype=np.float64)
        if not np.isfinite(q).all() or np.any(q < 0) or np.any(q > self.length_mm):
            raise ValueError("Point parameter outside original finite domain")
        return self.origin_mm + q[..., None] * self.direction if self.kind == "line" else self.arc.point_at(q)

    def project(self, points_mm):
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Finite XYZ projection points required")
        if self.kind == "line":
            return np.clip((points - self.origin_mm) @ self.direction, 0., self.length_mm)
        radial = points - self.arc.center_mm
        signed_angle = np.sign(self.arc.sweep_rad) * np.arctan2(radial @ np.cross(self.arc.plane_normal, self.arc.start_direction),
            radial @ self.arc.start_direction)
        branches = (signed_angle[:, None] + 2 * math.pi * np.arange(-2, 3)[None]) * self.arc.radius_mm
        legal = np.clip(branches, 0., self.length_mm)
        distance = np.linalg.norm(self.point_at(legal) - points[:, None], axis=2)
        return legal[np.arange(len(points)), distance.argmin(1)]

    def summary(self):
        return dict(ordinal=self.ordinal, weld_type=self.weld_type, kind=self.kind, length_mm=self.length_mm,
            geometry_identity=self.geometry_identity, aliases=list(self.aliases), cell_count=self.cell_count,
            cell_width_mm=self.cell_width_mm, independent_slots_per_cell=INDEPENDENT_SLOTS_PER_CELL,
            finite_endpoint_slots=2, structural_event_slots=len(self.structural_events_mm), total_slots=self.slot_count,
            residual_envelope_mm=SPATIAL_RESIDUAL_ENVELOPE_MM, geometry_is_not_reviewed_material=True)


def build_carriers(records):
    """Only exact geometry serialization aliases share a capacity domain."""
    groups, order = {}, []
    for record in records:
        key = exact_geometry_key(record)
        if key not in groups:
            groups[key] = [record, []]
            order.append(key)
        groups[key][1].append(record["id"])
    return [Carrier.from_record(index, groups[key][0], groups[key][1]) for index, key in enumerate(order)]


def analytic_whole_curve_certificate(prediction, truth):
    """Upper bound for the entire continuous curves, in both directions.

    Both use the same normalized traversal t in [0,1], a surjective pairing.
    LINE difference is affine, so its norm is bounded by the two endpoints.
    CIRCLE uses center/radius, orthonormal basis operator norm and sweep bounds.
    No finite sample table alone is called a continuous certificate.
    """
    if prediction["kind"] != truth["kind"]:
        raise ValueError("Whole-curve certificate requires the same analytic kind")
    if prediction["kind"] == "line":
        a = np.asarray(prediction["points"], np.float64)[[0, -1]]
        b = np.asarray(truth["endpoints_mm"], np.float64)
        error = float(np.linalg.norm(a - b, axis=1).max())
        return dict(method="AFFINE_CONTINUOUS_ENDPOINT_CONVEX_BOUND", ordered_endpoint_maximum_mm=error,
            continuous_bidirectional_upper_bound_mm=error)
    pa, ta = CircularArc.from_parameters(prediction["arc_parameters"]), CircularArc.from_parameters(truth["arc_parameters"])
    bp = np.stack((pa.start_direction, np.sign(pa.sweep_rad) * np.cross(pa.plane_normal, pa.start_direction)), axis=1)
    bt = np.stack((ta.start_direction, np.sign(ta.sweep_rad) * np.cross(ta.plane_normal, ta.start_direction)), axis=1)
    center = float(np.linalg.norm(pa.center_mm - ta.center_mm))
    radius = abs(pa.radius_mm - ta.radius_mm)
    basis = float(np.linalg.norm(bp - bt, ord=2))
    sweep = min(2., abs(abs(pa.sweep_rad) - abs(ta.sweep_rad)))
    bound = center + radius + max(pa.radius_mm, ta.radius_mm) * (basis + sweep)
    endpoint = float(np.linalg.norm(pa.point_at(np.array([0., pa.length_mm])) - ta.point_at(np.array([0., ta.length_mm])), axis=1).max())
    return dict(method="CONTINUOUS_CIRCLE_NORMALIZED_TRAVERSAL_OPERATOR_BOUND", ordered_endpoint_maximum_mm=endpoint,
        continuous_bidirectional_upper_bound_mm=bound, center_difference_mm=center, radius_difference_mm=radius,
        basis_operator_difference=basis, sweep_difference_bound=sweep)


def assemble_parent(carrier, boundary_nodes, children, gaps, is_closed=False):
    """Construct geometry from authoritative selected references, without snaps."""
    nodes = {}
    slots = set()
    for node in boundary_nodes:
        identity, slot, q = node["node_id"], int(node["slot_index"]), float(node["q_mm"])
        spec = carrier.slot_spec(slot)
        if identity in nodes or slot in slots or not spec["lower_mm"] <= q <= spec["upper_mm"]:
            raise ValueError("Repeated node/slot or parameter outside its declared slot")
        nodes[identity] = dict(q_mm=q, point_mm=carrier.point_at(q).tolist(), slot_index=slot)
        slots.add(slot)
    if not children or len(gaps) != len(children) - 1:
        raise ValueError("Explicit nonempty child references and adjacency required")
    segments, traversals = [], []
    for child in children:
        start, end = nodes[child["start_node"]], nodes[child["end_node"]]
        q0, q1 = start["q_mm"], end["q_mm"]
        if q0 == q1 or (carrier.weld_type == "vertical" and q1 <= q0):
            raise ValueError("Positive physical child travel; vertical path points down")
        traversals.append(float(np.sign(q1 - q0)))
        if carrier.kind == "line":
            segment = dict(kind="line", points=[start["point_mm"], end["point_mm"]], length_mm=abs(q1 - q0))
        else:
            direction = (np.asarray(start["point_mm"]) - carrier.arc.center_mm) / carrier.arc.radius_mm
            arc = CircularArc(carrier.arc.center_mm, carrier.arc.radius_mm, carrier.arc.plane_normal, direction,
                (q1 - q0) / carrier.arc.radius_mm * np.sign(carrier.arc.sweep_rad),
                bool(is_closed and len(children) == 1 and abs(q1 - q0) == carrier.length_mm))
            segment = sample_arc(arc)
            # These values come from the very same analytic boundary authority.
            segment["points"][0], segment["points"][-1] = start["point_mm"], end["point_mm"]
        minimum = 5. if carrier.kind == "line" else 2.
        if segment["length_mm"] < minimum:
            raise ValueError("Predicted child below registered process minimum")
        segment.update(weld_type=carrier.weld_type, start_boundary_reference=child["start_node"],
            end_boundary_reference=child["end_node"], carrier_ordinal=carrier.ordinal)
        segments.append(segment)
    if len(set(traversals)) != 1:
        raise ValueError("One parent requires consistent native traversal")
    for index, gap in enumerate(gaps, 1):
        before, after = children[index - 1]["end_node"], children[index]["start_node"]
        if gap == "SHARED" and before != after:
            raise ValueError("SHARED requires the same boundary reference")
        if gap == "POSITIVE_GAP" and (before == after or traversals[0] * (nodes[after]["q_mm"] - nodes[before]["q_mm"]) <= 0):
            raise ValueError("POSITIVE_GAP requires distinct forward boundary references")
        if gap not in ("SHARED", "POSITIVE_GAP"):
            raise ValueError("Unknown adjacency state")
    mode = "LINE_CHAIN" if carrier.kind == "line" else "CIRCLE_CHAIN"
    authority = validate_component_geometry(mode, segments, gaps)
    return dict(mode=mode, child_count=len(segments), segments=segments, gaps=list(gaps), is_closed=bool(is_closed),
        boundary_nodes=nodes, common_analytic_authority=authority, residual_parameters="ZERO_VALID_MEMBER_OF_FROZEN_3MM_ENVELOPE")
