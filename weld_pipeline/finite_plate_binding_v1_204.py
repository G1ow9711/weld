"""Pure XYZ finite surface and material/contact *candidates*, version 204.

No labels, CAD, source names, learned model, coordinate identity, or best-object
selection enters this API. Triangles certify sampled interpolation only; unseen
material, holes, closed-solid identity, and customer side semantics stay unknown.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, Delaunay, QhullError

VERSION = "finite_plate_binding_candidates_v1_204"


@dataclass(frozen=True)
class CandidateConfig:
    plane_tolerance_mm: float = .15  # Existing finite joint/plate tolerance.
    joint_tolerance_mm: float = 3.  # Existing finite-joint candidate tolerance.
    parallel_cosine: float = .99999  # Existing raw direction dedup tolerance.
    triangle_scaled_edge_limit: float = 1.8  # Existing supported-base triangles.
    neighbor_count: int = 24


def _unit(value):
    value = np.asarray(value, np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Finite XYZ vector required")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("Nondegenerate vector required")
    return value / norm


def _basis(normal):
    # Input-independent basis; sign is numerical canonicalization, not outward.
    n = _unit(normal)
    if n[np.argmax(np.abs(n))] < 0:
        n = -n
    reference = np.eye(3)[np.argmin(np.abs(n))]
    u = _unit(np.cross(n, reference))
    return n, np.stack((u, np.cross(n, u)))


def _spacing(uv, count):
    tree = cKDTree(uv)
    samples = [[], []]
    k = min(count, len(uv))
    for begin in range(0, len(uv), 4096):
        q = uv[begin:begin + 4096]
        distances, indices = tree.query(q, k=k, workers=1)
        if k == 1:
            return None
        displacement = uv[indices] - q[:, None]
        for axis in range(2):
            cone = (np.abs(displacement[:, :, axis]) >= .75 * distances) & (distances > 1e-8)
            nearest = np.where(cone, distances, np.inf).min(1)
            samples[axis].append(nearest[np.isfinite(nearest)])
    medians = []
    for rows in samples:
        values = np.concatenate(rows)
        if len(values) < 3:
            return None
        medians.append(float(np.median(values)))
    return np.asarray(medians)


class FinitePlateCandidates:
    """Input-local finite surface candidates. IDs are enumeration ordinals.

    ``plane_hypotheses`` consists exclusively of XYZ-derived {normal,offset_mm}
    records. Distinct explicit hypotheses retain distinct IDs even if identical.
    No object identity is inferred by equality/proximity of coordinates.
    """
    def __init__(self, xyz_mm, plane_hypotheses, config=CandidateConfig()):
        self.xyz = np.asarray(xyz_mm, np.float64)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3 or not np.isfinite(self.xyz).all():
            raise ValueError("Finite N by 3 millimeter XYZ required")
        scalar_config = np.asarray([config.plane_tolerance_mm, config.joint_tolerance_mm,
                                   config.parallel_cosine, config.triangle_scaled_edge_limit], float)
        if (not np.isfinite(scalar_config).all() or (scalar_config <= 0).any()
                or config.parallel_cosine > 1.
                or isinstance(config.neighbor_count, (bool, np.bool_))
                or not isinstance(config.neighbor_count, (int, np.integer)) or config.neighbor_count < 3):
            raise ValueError("Finite positive tolerances, cosine <=1 and integer neighbor_count >=3 required")
        self.config = config
        self.faces, self.plates, self.plane_diagnostics = [], [], []
        self._geometry = []
        for plane_id, hypothesis in enumerate(plane_hypotheses):
            self._plane(plane_id, hypothesis)
        self._material_pairs()

    def _plane(self, plane_id, hypothesis):
        original = _unit(hypothesis['normal'])
        normal, basis = _basis(original)
        offset = float(hypothesis['offset_mm']) * float(original @ normal)
        if not np.isfinite(offset):
            raise ValueError("Finite plane offset required")
        raw_ids = np.flatnonzero(np.abs(self.xyz @ normal - offset) <= self.config.plane_tolerance_mm)
        origin = normal * offset
        uv, first, inverse = np.unique((self.xyz[raw_ids] - origin) @ basis.T,
                                       axis=0, return_index=True, return_inverse=True)
        diagnostic = dict(plane_id=plane_id, normal=normal.tolist(), offset_mm=offset,
            raw_support_count=len(raw_ids), unique_support_count=len(uv),
            admitted_face_ids=[], status="INSUFFICIENT_SAMPLED_2D_SUPPORT")
        self.plane_diagnostics.append(diagnostic)
        if len(uv) < 3 or np.linalg.matrix_rank(uv - uv.mean(0), tol=1e-8) < 2:
            return
        spacing = _spacing(uv, self.config.neighbor_count)
        if spacing is None or (spacing <= 0).any():
            diagnostic['status'] = 'TWO_DIMENSIONAL_SAMPLING_GAPS_UNRESOLVED'
            return
        try:
            tri = Delaunay(uv)
        except QhullError:
            diagnostic['status'] = 'TRIANGULATION_UNRESOLVED'
            return
        triangles = tri.simplices
        edges = uv[triangles[:, [1, 2, 0]]] - uv[triangles]
        supported = (np.linalg.norm(edges / spacing, axis=2).max(1)
                     <= self.config.triangle_scaled_edge_limit)
        accepted = triangles[supported]
        diagnostic.update(sampling_spacing_uv_mm=spacing.tolist(),
            proposed_triangles=len(triangles), accepted_triangles=len(accepted),
            rejected_gap_triangles=int((~supported).sum()))
        if not len(accepted):
            diagnostic['status'] = 'ONLY_UNSUPPORTED_GAP_TRIANGLES'
            return
        links = np.concatenate((accepted[:, [0, 1]], accepted[:, [1, 2]], accepted[:, [2, 0]]))
        used = np.unique(links)
        adjacency = coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(len(uv), len(uv)))
        _, labels = connected_components(adjacency, directed=False)
        for component_id, label in enumerate(np.unique(labels[used])):
            ids = used[labels[used] == label]
            component_triangles = accepted[labels[accepted[:, 0]] == label]
            lookup = np.full(len(uv), -1, np.int64); lookup[ids] = np.arange(len(ids))
            local_triangles = lookup[component_triangles]
            component_uv = uv[ids]
            component_raw = raw_ids[np.isin(inverse, ids)]
            raw_representatives = raw_ids[first[ids]]
            boundary = np.sort(np.concatenate((local_triangles[:, [0, 1]],
                local_triangles[:, [1, 2]], local_triangles[:, [2, 0]])), axis=1)
            boundary, counts = np.unique(boundary, axis=0, return_counts=True)
            boundary = boundary[counts == 1]
            face_id = len(self.faces)
            points = component_uv @ basis + origin
            tree = cKDTree(component_uv)
            geometry = dict(uv=component_uv, xyz=points, triangles=local_triangles,
                boundary=boundary, raw_ids=component_raw, raw_representatives=raw_representatives,
                tree=tree, normal=normal, basis=basis, origin=origin, spacing=spacing)
            # Barycentric membership uses only accepted triangles, never bbox/hull fill.
            geometry['tri_centres'] = component_uv[local_triangles].mean(1)
            geometry['tri_tree'] = cKDTree(geometry['tri_centres'])
            geometry['max_triangle_radius'] = float(np.linalg.norm(
                component_uv[local_triangles] - geometry['tri_centres'][:, None], axis=2).max())
            self._geometry.append(geometry)
            area = .5 * np.abs(np.cross(component_uv[local_triangles[:, 1]] - component_uv[local_triangles[:, 0]],
                                          component_uv[local_triangles[:, 2]] - component_uv[local_triangles[:, 0]])).sum()
            self.faces.append(dict(face_id=face_id, plane_id=plane_id, component_id=component_id,
                normal=normal.tolist(), offset_mm=offset, basis_uv=basis.tolist(), origin_mm=origin.tolist(),
                raw_support_count=len(component_raw), unique_support_count=len(ids),
                sampled_triangle_area_mm2=float(area), triangle_count=len(local_triangles),
                boundary_edge_count=len(boundary), sampling_spacing_uv_mm=spacing.tolist(),
                uv_bounds_mm=[component_uv.min(0).tolist(), component_uv.max(0).tolist()],
                z_bounds_mm=[float(points[:, 2].min()), float(points[:, 2].max())],
                support_array_key=f'face_{face_id}_raw_ids',
                unique_point_array_key=f'face_{face_id}_raw_representatives',
                triangle_array_key=f'face_{face_id}_triangles',
                boundary_array_key=f'face_{face_id}_boundary',
                status='FINITE_SAMPLED_PATCH_CANDIDATE',
                material_identity_verified=False, closed_solid_verified=False,
                hidden_holes_ruled_out=False, reviewed_customer_truth=False))
            diagnostic['admitted_face_ids'].append(face_id)
        diagnostic['status'] = 'FINITE_SAMPLED_PATCH_CANDIDATES'

    def observe_face(self, face_id, point_mm):
        if (isinstance(face_id, (bool, np.bool_)) or not isinstance(face_id, (int, np.integer))
                or not 0 <= face_id < len(self.faces)):
            raise ValueError('Valid nonnegative face ordinal required')
        q = np.asarray(point_mm, np.float64)
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError('Finite point required')
        face, g = self.faces[face_id], self._geometry[face_id]
        distance = float(q @ g['normal'] - face['offset_mm'])
        projected = q - distance * g['normal']
        uvq = (projected - g['origin']) @ g['basis'].T
        nearest, index = g['tree'].query(uvq)
        inside, triangle_id = False, None
        candidates = g['tri_tree'].query_ball_point(uvq, g['max_triangle_radius'] + 1e-8)
        for candidate in candidates:
            v = g['uv'][g['triangles'][candidate]]
            matrix = np.stack((v[1] - v[0], v[2] - v[0]), axis=1)
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) < 1e-12:
                continue
            b = np.linalg.solve(matrix, uvq - v[0])
            if min(b) >= -1e-8 and b.sum() <= 1. + 1e-8:
                inside, triangle_id = True, int(candidate)
                break
        # An edge point may lie just outside a sampled patch; report candidate
        # tolerance separately instead of claiming interior support.
        near = float(nearest) <= self.config.joint_tolerance_mm
        return dict(face_id=face_id, projected_point_mm=projected.tolist(),
            signed_distance_to_canonical_plane_mm=distance, nearest_sample_mm=float(nearest),
            nearest_original_raw_index=int(g['raw_representatives'][int(index)]),
            inside_supported_triangle=inside, supported_triangle_id=triangle_id,
            near_sample_within_joint_tolerance=near,
            admissible_sample_candidate=abs(distance) <= self.config.joint_tolerance_mm and (inside or near),
            certified_material_interior=False)

    def _material_pairs(self):
        for left, right in combinations(range(len(self.faces)), 2):
            a, b = self.faces[left], self.faces[right]
            n, m = np.asarray(a['normal']), np.asarray(b['normal'])
            if abs(float(n @ m)) < self.config.parallel_cosine:
                continue
            # Evaluate separation at a real left support point, not bbox centres.
            point = self._geometry[left]['xyz'][0]
            delta = b['offset_mm'] - float(point @ m)
            if abs(delta) <= 1e-8:
                continue  # Coincident surfaces remain two face hypotheses.
            witnesses = []
            for source, target in ((left, right), (right, left)):
                g = self._geometry[source]
                # Fixed deterministic evidence sample; not an object score/search.
                indices = np.unique(np.linspace(0, len(g['triangles']) - 1, min(17, len(g['triangles']))).astype(int))
                for tri_id in indices:
                    q = g['xyz'][g['triangles'][tri_id]].mean(0)
                    observed = self.observe_face(target, q)
                    if observed['inside_supported_triangle']:
                        witnesses.append(dict(source_face_id=source, source_triangle_id=int(tri_id),
                            source_point_mm=q.tolist(), target_face_id=target,
                            projected_point_mm=observed['projected_point_mm'],
                            target_triangle_id=observed['supported_triangle_id']))
            outward_left = -np.sign(delta) * m
            plate_id = len(self.plates)
            self.plates.append(dict(plate_id=plate_id, face_ids=[left, right],
                thickness_candidate_mm=float(abs(delta)), parallel_cosine=float(abs(n @ m)),
                overlap_sample_witnesses=witnesses,
                overlap_status='SAMPLED_OVERLAP_WITNESSED' if witnesses else 'FINITE_OVERLAP_UNRESOLVED',
                overlap_search_exhaustive=False,
                conditional_material_outward_normals=[outward_left.tolist(), (-outward_left).tolist()],
                outward_condition='ONLY_IF_THESE_FACES_BOUND_ONE_FILLED_MATERIAL_INTERVAL',
                status='UNREVIEWED_MATERIAL_INTERVAL_CANDIDATE',
                actual_solid_identity_verified=False, closed_slab_verified=False,
                hidden_holes_ruled_out=False, signed_customer_side_semantics='PENDING',
                reviewed_customer_truth=False))

    def bind_child(self, points_mm):
        points = np.asarray(points_mm, np.float64)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError('Two finite unmodified child endpoints required')
        direction = points[1] - points[0]
        if np.linalg.norm(direction) < 1e-12:
            return dict(original_points_mm=points.tolist(), status='DEGENERATE_CHILD_UNKNOWN',
                        contact_alternatives=[], endpoint_local_base_alternatives=[[], []], output_deleted=False)
        tangent = _unit(direction)
        samples = points[0] + np.asarray([0., .25, .5, .75, 1.])[:, None] * direction
        observations = [[self.observe_face(i, q) for q in samples] for i in range(len(self.faces))]
        touching = [i for i, obs in enumerate(observations)
                    if all(o['admissible_sample_candidate'] for o in obs)
                    and abs(float(np.asarray(self.faces[i]['normal']) @ tangent)) < .15]
        incident = {i: [p['plate_id'] for p in self.plates if i in p['face_ids']] for i in touching}
        contacts = []
        for left, right in combinations(touching, 2):
            cosine = abs(float(np.asarray(self.faces[left]['normal']) @ np.asarray(self.faces[right]['normal'])))
            if cosine >= self.config.parallel_cosine:
                continue
            contacts.append(dict(contact_id=len(contacts), face_ids=[left, right],
                plate_alternatives_by_face=[incident[left], incident[right]],
                original_five_samples_mm=samples.tolist(),
                face_sample_evidence=[observations[left], observations[right]],
                status='SAMPLED_CONTACT_ALTERNATIVE', actual_contact_solid_ids=None,
                complete_along_child_continuity_verified=False, reviewed_customer_truth=False))
        bases = [[], []]
        for end, point in enumerate(points):
            for face_id, face in enumerate(self.faces):
                if abs(float(np.asarray(face['normal']) @ np.array([0., 0., 1.]))) < .99999:
                    continue
                obs = self.observe_face(face_id, point)
                if not obs['admissible_sample_candidate']:
                    continue
                vertical_contact_faces = [i for i in touching if abs(self.faces[i]['normal'][2]) < .15]
                base_plate_ids = [p['plate_id'] for p in self.plates if face_id in p['face_ids']]
                for contact_face_id in vertical_contact_faces:
                    zrange = self.faces[contact_face_id]['z_bounds_mm']
                    bases[end].append(dict(base_face_id=face_id, base_plate_alternatives=base_plate_ids,
                        contact_face_id=contact_face_id, contact_plate_alternatives=incident[contact_face_id],
                        local_base_point_mm=obs['projected_point_mm'], raw_support=obs,
                        candidate_face_z_span_mm=float(zrange[1] - zrange[0]),
                        sampled_top_above_local_base_mm=float(zrange[1] - obs['projected_point_mm'][2]),
                        sampled_bottom_below_local_base_mm=float(obs['projected_point_mm'][2] - zrange[0]),
                        measurement_scope='SAMPLED_CONTACT_FACE_EXTENT_ONLY',
                        actual_contact_plate_height_mm=None, material_continuity_verified=False,
                        reviewed_customer_truth=False))
        reasons = ['ACTUAL_MATERIAL_IDENTITY_UNREVIEWED', 'CLOSED_SLAB_AND_HIDDEN_HOLES_UNKNOWN',
                   'SIDE1_SIDE2_AND_SIGNED_NORMAL_SEMANTICS_PENDING']
        if not contacts:
            reasons.append('NO_TWO_FINITE_CONTACT_FACES_SUPPORTED')
        if any(not ids for ids in incident.values()):
            reasons.append('MISSING_OPPOSITE_FACE_FOR_SOME_CONTACT_CANDIDATE')
        if len(contacts) > 1 or any(len(ids) > 1 for ids in incident.values()):
            reasons.append('MULTIPLE_MATERIAL_OR_CONTACT_ALTERNATIVES_RETAINED')
        return dict(original_points_mm=points.tolist(), status='CANDIDATES_WITH_EXPLICIT_UNKNOWN',
            touching_face_ids=touching, contact_alternatives=contacts,
            endpoint_local_base_alternatives=bases, unknown_reasons=reasons,
            output_deleted=False, original_geometry_changed=False, full_contract_attributes_pass=False)

    def to_dict(self):
        return dict(version=VERSION, config=vars(self.config), original_raw_point_count=len(self.xyz),
            planes=self.plane_diagnostics, faces=self.faces, plate_candidates=self.plates,
            runtime_inputs='XYZ_AND_PURE_XYZ_PLANE_HYPOTHESES', ids='INPUT_LOCAL_ENUMERATION_ORDINALS',
            plane_hypothesis_duplicate_identity_retained=True, full_contract_pass=False)

    def support_arrays(self):
        arrays = {}
        for face_id, g in enumerate(self._geometry):
            arrays[f'face_{face_id}_raw_ids'] = g['raw_ids']
            arrays[f'face_{face_id}_raw_representatives'] = g['raw_representatives']
            arrays[f'face_{face_id}_triangles'] = g['triangles']
            arrays[f'face_{face_id}_boundary'] = g['boundary']
        return arrays
