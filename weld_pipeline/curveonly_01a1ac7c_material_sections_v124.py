"""Small finite-solid sections; observations only, never welding permission."""
import time
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_exact_mesh_index_v110 as owner_cache
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box

NORMAL_OFFSETS_MM = (1., 8., 35.)
BASE_OFFSETS_MM = (-1., 3., 35.)
CLASSIFY_TOLERANCE_MM = 1e-6
PROBES = tuple((a, b) for a in NORMAL_OFFSETS_MM for b in BASE_OFFSETS_MM)
FIELDS = ('inside_count_log', 'on_count_log', 'owner_occupied', 'state_available')
NAMES = ('frame_available', 'owner_known') + tuple(
    f'side{side}_normal{a:g}_base{b:g}_{field}'
    for side in (0, 1) for a, b in PROBES for field in FIELDS)


class SolidProbe:
    def __init__(self, solids):
        self.solids = list(solids)
        self.classifiers = [BRepClass3d_SolidClassifier(solid) for solid in solids]
        boxes = []
        for solid in solids:
            box = Bnd_Box()
            BRepBndLib.AddOptimal_s(solid, box, False, True)
            boxes.append(box.Get())
        self.boxes = np.asarray(boxes, float).reshape(-1, 6)
        self.calls = 0

    def classify(self, points, owners):
        """Retain IN, ON, OUT and UNKNOWN separately before feature encoding."""
        points = np.asarray(points, float)
        owners = np.asarray(owners, int)
        assert points.shape == (len(owners), 3)
        inside = np.zeros(len(points), np.int16)
        on = np.zeros(len(points), np.int16)
        own = np.zeros(len(points), bool)
        available = np.ones(len(points), bool)
        for mi, (box, classifier) in enumerate(zip(self.boxes, self.classifiers)):
            candidates = np.flatnonzero(np.all((points >= box[:3] - CLASSIFY_TOLERANCE_MM)
                                              & (points <= box[3:] + CLASSIFY_TOLERANCE_MM), axis=1))
            for index in candidates:
                classifier.Perform(cad.gp_Pnt(*map(float, points[index])), CLASSIFY_TOLERANCE_MM)
                state = classifier.State(); self.calls += 1
                if state == TopAbs_IN:
                    inside[index] += 1
                elif state == TopAbs_ON:
                    on[index] += 1
                elif state != TopAbs_OUT:
                    available[index] = False
                if mi == owners[index] and state in (TopAbs_IN, TopAbs_ON):
                    own[index] = True
        return dict(inside=inside, on=on, owner_occupied=own, available=available)


def tangents(points, closed):
    p = np.asarray(points, float)
    if closed:
        unique = p[:-1]
        value = np.roll(unique, -1, axis=0) - np.roll(unique, 1, axis=0)
        value = np.vstack([value, value[0]])
    else:
        value = np.empty_like(p)
        value[0] = p[1] - p[0]; value[-1] = p[-1] - p[-2]
        if len(p) > 2:
            value[1:-1] = p[2:] - p[:-2]
    norm = np.linalg.norm(value, axis=1)
    return value / np.maximum(norm[:, None], 1e-20)


def section_values(points, tangent, base_axis, owners, transform, probe):
    points = np.asarray(points, float); tangent = np.asarray(tangent, float)
    owners = np.asarray(owners, int); transform = np.asarray(transform, float)
    axis = np.asarray(base_axis, float).copy(); axis /= np.linalg.norm(axis)
    normal = np.cross(tangent, axis)
    scale = np.linalg.norm(normal, axis=1)
    valid = scale > 1e-8
    normal /= np.maximum(scale[:, None], 1e-20)
    result = np.zeros((len(points), len(NAMES)), np.float32)
    result[:, 0] = valid
    result[:, 1] = (owners >= 0) & (owners < len(probe.solids))
    selected = np.flatnonzero(valid)
    if not len(selected):
        return result
    offsets = np.asarray(PROBES)
    side = np.asarray([-1., 1.])
    world = (points[selected, None, None, :]
             + side[None, :, None, None] * offsets[None, None, :, 0, None] * normal[selected, None, None, :]
             + offsets[None, None, :, 1, None] * axis[None, None, None, :])
    native = (world.reshape(-1, 3) - transform[:3, 3]) @ transform[:3, :3]
    own = np.broadcast_to(owners[selected, None, None], world.shape[:-1]).reshape(-1)
    state = probe.classify(native, own)
    values = np.column_stack([np.log1p(state['inside']), np.log1p(state['on']),
                              state['owner_occupied'].astype(float), state['available'].astype(float)])
    values = values.reshape(len(selected), 2, len(PROBES) * len(FIELDS))
    # Curve reversal exchanges the two normals. Sort whole side signatures,
    # preserving the arrangements within each side and never using material IDs.
    for local_index, index in enumerate(selected):
        a, b = values[local_index]
        if tuple(b) < tuple(a):
            a, b = b, a
        result[index, 2:] = np.r_[a, b]
    assert result.shape[1] == 74 and np.isfinite(result).all()
    return result


def build(packet, solids):
    start = time.perf_counter(); n = packet['offsets'][-1]
    values = np.zeros((n, len(NAMES)), np.float32)
    pose = packet.get('pose', {})
    transform = np.asarray(pose.get('transform_STEP_to_XYZ', []), float)
    if not (solids and pose.get('passed') and transform.shape == (4, 4)):
        return values, dict(available=False, reason='NO_QUALIFIED_FINITE_STEP', geometry_retained=True,
                            no_weld_decision=True, seconds=time.perf_counter() - start)
    rotation = transform[:3, :3]
    if not (np.isfinite(transform).all() and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            and abs(np.linalg.det(rotation) - 1.) <= 1e-7):
        return values, dict(available=False, reason='NONRIGID_OR_INVALID_POSE', geometry_retained=True,
                            no_weld_decision=True, seconds=time.perf_counter() - start)
    probe = SolidProbe(solids); owner_at = owner_cache.owner_function(packet)
    details = []
    for ci, candidate in enumerate(packet['candidates']):
        lo, hi = packet['offsets'][ci:ci + 2]
        points = np.asarray(candidate['stations'], float)
        closed = np.linalg.norm(np.asarray(candidate['points'])[0] - candidate['points'][-1]) <= 1e-6
        tangent = tangents(points, closed)
        if candidate.get('STEP') and candidate.get('pose'):
            owners = np.asarray([owner_at(ci, float(s), packet) for s in candidate['station_s_mm']], int)
        else:
            owners = np.full(len(points), -1, int)
        values[lo:hi] = section_values(points, tangent, packet['base_axis'], owners, transform, probe)
        details.append(dict(candidate=ci, rows=hi - lo,
                            frame_available_rows=int(values[lo:hi, 0].sum()), owner_known_rows=int(values[lo:hi, 1].sum())))
    state_channels = values[:, 2:].reshape(n, 2, len(PROBES), len(FIELDS))[:, :, :, 3]
    receipt = dict(available=True, columns=len(NAMES), names=list(NAMES), material_instances=len(solids),
        exact_solid_classifier_calls=probe.calls, total_probe_points=n * 2 * len(PROBES),
        unknown_probe_count=int((state_channels == 0).sum()), candidates=details,
        normal_offsets_mm=list(NORMAL_OFFSETS_MM), base_offsets_mm=list(BASE_OFFSETS_MM),
        classifier_tolerance_mm=CLASSIFY_TOLERANCE_MM, frame='Input curve tangent and verified base axis; unordered transverse sides.',
        all_finite_materials_considered=True, AABB_pruning_only_not_filled_material=True,
        inside_surface_and_unknown_distinct=True, unknown_owner_not_nonweld=True,
        no_new_curve_or_endpoint_change=True, no_weld_side_joint_or_torch_permission_certified=True,
        no_labels_or_models_read=True, seconds=time.perf_counter() - start)
    return values, receipt
