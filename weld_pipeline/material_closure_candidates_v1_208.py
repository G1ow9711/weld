"""Candidate-local finite sidewall, seam and oriented sampled-shell evidence.

No disk, CAD, labels, names, learned scores, or material winner selection.
Clipped intersections are derived interpolation inside supplied supported
triangles. Shell volume is algebraic surface geometry, never filled material
truth or proof against hidden cavities/self-intersections.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np

VERSION = 'finite_material_closure_candidates_v1_208'


@dataclass(frozen=True)
class ClosureConfig:
    numerical_epsilon_mm: float = 1e-8
    near_seam_tolerance_mm: float = .15
    parallel_cosine: float = .99999
    triangle_chunk: int = 4096


@dataclass
class SurfacePatch:
    id: int
    raw_xyz: np.ndarray
    triangles: np.ndarray
    normal: np.ndarray
    offset_mm: float
    boundary: np.ndarray = None
    raw_ids: np.ndarray = None
    source_topology_verified: bool = False

    def __post_init__(self):
        _id(self.id)
        if not isinstance(self.source_topology_verified, (bool, np.bool_)):
            raise ValueError('Explicit boolean source topology provenance required')
        self.id = int(self.id)
        self.source_topology_verified = bool(self.source_topology_verified)
        self.raw_xyz = np.asarray(self.raw_xyz, np.float64)
        self.normal = np.asarray(self.normal, np.float64)
        self.triangles = _indices(self.triangles, 3, len(self.raw_xyz))
        if (self.raw_xyz.ndim != 2 or self.raw_xyz.shape[1] != 3 or not np.isfinite(self.raw_xyz).all()
                or self.normal.shape != (3,) or not np.isfinite(self.normal).all()
                or np.linalg.norm(self.normal) < 1e-12 or not np.isfinite(self.offset_mm)):
            raise ValueError('Finite raw vertex XYZ and plane required')
        self.normal = self.normal / np.linalg.norm(self.normal)
        sorted_triangles = np.sort(self.triangles, axis=1)
        self._duplicate_source_triangles = len(self.triangles) - len(np.unique(sorted_triangles, axis=0))
        self._conflicting_source_two_sided_edges = 0 if self.source_topology_verified else None
        if self.boundary is None:
            triangles = self.triangles.copy()
            vertices = _project(self.raw_xyz[triangles], self)
            cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
            flipped = cross @ self.normal < 0
            triangles[flipped] = triangles[flipped][:, [0, 2, 1]]
            directed = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
            edge, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
            signed = np.bincount(inverse, weights=np.where(directed[:, 0] < directed[:, 1], 1, -1))
            self._conflicting_source_two_sided_edges = int(((counts == 2) & (np.abs(signed) == 2)).sum())
            self.boundary = edge[counts == 1]
            self._nonmanifold_edges = int((counts > 2).sum())
        else:
            self.boundary = _indices(self.boundary, 2, len(self.raw_xyz))
            self._nonmanifold_edges = 0 if self.source_topology_verified else None
        if self.raw_ids is not None:
            self.raw_ids = np.asarray(self.raw_ids)
            if (self.raw_ids.shape != (len(self.raw_xyz),) or self.raw_ids.dtype.kind not in 'iu'
                    or (self.raw_ids < 0).any()):
                raise ValueError('Original nonnegative raw-row IDs required')


@dataclass(frozen=True)
class IntervalCandidate:
    id: int
    left_face_id: int
    right_face_id: int

    def __post_init__(self):
        for value in (self.id, self.left_face_id, self.right_face_id):
            _id(value)
        for name in ('id', 'left_face_id', 'right_face_id'):
            object.__setattr__(self, name, int(getattr(self, name)))


def _id(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('Nonnegative explicit integer candidate identity required')


def _indices(value, width, count):
    a = np.asarray(value)
    if a.size == 0:
        return np.empty((0, width), np.int64)
    if a.ndim != 2 or a.shape[1] != width or a.dtype.kind not in 'iu' or (a < 0).any() or (a >= count).any():
        raise ValueError('Valid integer local vertex references required')
    return a.astype(np.int64, copy=False)


def _project(xyz, patch):
    return xyz - (xyz @ patch.normal - patch.offset_mm)[..., None] * patch.normal


def _clip_polygon(poly, halfspaces, epsilon):
    poly = [np.asarray(p) for p in poly]
    for normal, offset in halfspaces:
        result = []
        if not poly:
            break
        for p, q in zip(poly, poly[1:] + poly[:1]):
            a, b = float(p @ normal - offset), float(q @ normal - offset)
            if a <= epsilon:
                result.append(p)
            if (a > epsilon) != (b > epsilon):
                result.append(p + (q - p) * (a / (a - b)))
        poly = result
    return poly


def _clip_segment(a, b, halfspaces, epsilon):
    a, b = a.copy(), b.copy()
    for normal, offset in halfspaces:
        x, y = float(a @ normal - offset), float(b @ normal - offset)
        if x > epsilon and y > epsilon:
            return None
        if (x > epsilon) != (y > epsilon):
            cut = a + (b - a) * x / (x - y)
            if x > epsilon:
                a = cut
            else:
                b = cut
    if np.linalg.norm(b - a) <= epsilon:
        return None
    return np.stack((a, b))


def _clipped_patch(patch, halfspaces, reference, config):
    """Stream source triangles; retain only boundary geometry and integrals."""
    eps = config.numerical_epsilon_mm
    nv = len(patch.raw_xyz)
    keys = np.minimum(patch.boundary[:, 0], patch.boundary[:, 1]) * max(nv, 1) + np.maximum(patch.boundary[:, 0], patch.boundary[:, 1])
    order = np.argsort(keys); sorted_keys = keys[order]
    directions = np.zeros(len(keys), np.int8)
    area, volume, kept, degenerate = 0., 0., 0, 0
    cap_segments, cap_source_ids = [], []
    for begin in range(0, len(patch.triangles), config.triangle_chunk):
        ids = patch.triangles[begin:begin + config.triangle_chunk]
        tri = _project(patch.raw_xyz[ids], patch)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        flipped = cross @ patch.normal < 0
        tri[flipped] = tri[flipped][:, [0, 2, 1]]
        local = ids.copy(); local[flipped] = local[flipped][:, [0, 2, 1]]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge_keys = np.minimum(local[:, a], local[:, b]) * max(nv, 1) + np.maximum(local[:, a], local[:, b])
            positions = np.searchsorted(sorted_keys, edge_keys)
            valid = positions < len(sorted_keys)
            rows = np.flatnonzero(valid)
            rows = rows[sorted_keys[positions[rows]] == edge_keys[rows]]
            directions[order[positions[rows]]] = np.where(local[rows, a] < local[rows, b], 1, -1)
        distances = np.stack([tri @ n - o for n, o in halfspaces], axis=-1)
        inside = (distances <= eps).all(axis=(1, 2))
        outside = (distances > eps).all(axis=1).any(axis=1)
        full = tri[inside]
        if len(full):
            cr = np.cross(full[:, 1] - full[:, 0], full[:, 2] - full[:, 0])
            sizes = np.linalg.norm(cr, axis=1)
            area += float(sizes.sum() * .5)
            q = full - reference
            volume += float(np.einsum('ij,ij->i', q[:, 0], np.cross(q[:, 1], q[:, 2])).sum() / 6.)
            kept += len(full); degenerate += int((sizes <= 1e-16).sum())
            full_rows = np.flatnonzero(inside)
            for normal, offset in halfspaces:
                if abs(normal @ patch.normal) >= 1. - 1e-12:
                    continue
                on_plane = np.abs(full @ normal - offset) <= eps
                for x, y in ((0, 1), (1, 2), (2, 0)):
                    for row in np.flatnonzero(on_plane[:, x] & on_plane[:, y] & (sizes > 1e-16)):
                        if np.linalg.norm(full[row, y] - full[row, x]) > eps:
                            cap_segments.append(full[row, [x, y]].copy())
                            cap_source_ids.append(begin + int(full_rows[row]))
        for row in np.flatnonzero(~inside & ~outside):
            poly = _clip_polygon(tri[row], halfspaces, eps)
            if len(poly) >= 3:
                polygon_area = 0.
                for j in range(1, len(poly) - 1):
                    v = np.asarray([poly[0], poly[j], poly[j + 1]])
                    cr = np.cross(v[1] - v[0], v[2] - v[0]); size = float(np.linalg.norm(cr))
                    if size > 1e-16:
                        polygon_area += size * .5
                        area += size * .5; kept += 1
                        q = v - reference
                        volume += float(q[0] @ np.cross(q[1], q[2]) / 6.)
                for p, q in zip(poly, poly[1:] + poly[:1]) if polygon_area > 1e-16 else ():
                    if np.linalg.norm(q - p) <= eps:
                        continue
                    for normal, offset in halfspaces:
                        if (abs(normal @ patch.normal) < 1. - 1e-12
                                and max(abs(p @ normal - offset), abs(q @ normal - offset)) <= eps):
                            cap_segments.append(np.stack((p, q))); cap_source_ids.append(begin + int(row))
                            break
    segments, lineage = [], []
    for edge_ordinal, edge in enumerate(patch.boundary):
        vertices = _project(patch.raw_xyz[edge], patch)
        if directions[edge_ordinal] < 0:
            vertices = vertices[::-1]
        clipped = _clip_segment(*vertices, halfspaces, eps)
        if clipped is not None:
            segments.append(clipped); lineage.append(('ORIGINAL_BOUNDARY_CLIPPED', edge_ordinal))
    segments.extend(cap_segments)
    lineage.extend(('DERIVED_TRIANGLE_PLANE_CUT_NO_RAW_ID', i) for i in cap_source_ids)
    # A clipped triangle lying on a source boundary can contribute the same
    # segment twice. Cancel opposite equal directed segments within this patch.
    boundary = np.asarray(segments, np.float64).reshape(-1, 2, 3)
    retained = np.ones(len(boundary), bool)
    if len(boundary):
        seen = {}
        for i, segment in enumerate(boundary):
            # Numeric key only indexes potential same-patch segment matches;
            # actual endpoint distances are checked before cancellation.
            key = tuple(np.round(segment.mean(0) / eps).astype(np.int64))
            found = False
            for j in seen.get(key, []):
                if not retained[j]:
                    continue
                if np.max(np.linalg.norm(boundary[j] - segment[::-1], axis=1)) <= eps:
                    retained[i] = retained[j] = False; found = True; break
                if np.max(np.linalg.norm(boundary[j] - segment, axis=1)) <= eps:
                    retained[i] = False; found = True; break
            if not found:
                seen.setdefault(key, []).append(i)
    boundary = boundary[retained]
    return dict(id=patch.id, boundary=boundary, area=area, volume=volume,
        triangle_count=kept, degenerate_triangles=degenerate,
        source_nonmanifold_edges=patch._nonmanifold_edges,
        duplicate_source_triangles=patch._duplicate_source_triangles,
        conflicting_source_two_sided_edges=patch._conflicting_source_two_sided_edges,
        raw_vertex_plane_residual_max_mm=float(np.abs(patch.raw_xyz @ patch.normal - patch.offset_mm).max()) if nv else 0.,
        source_triangle_count=len(patch.triangles), normal=patch.normal,
        lineage=[lineage[i] for i in np.flatnonzero(retained)],
        unknown_boundary_directions=int((directions == 0).sum()),
        reference=reference, material_identity_verified=False)


def _coverage(segment, others, epsilon):
    a, b = segment; length = float(np.linalg.norm(b - a))
    if length <= epsilon:
        return [], [], [], length
    direction = (b - a) / length
    matches = []
    for other in others:
        if not len(other['boundary']):
            continue
        r = other['boundary'] - a
        along = r @ direction
        perpendicular = r - along[..., None] * direction
        valid = np.linalg.norm(perpendicular, axis=2).max(1) <= epsilon
        for index in np.flatnonzero(valid):
            lo, hi = sorted(along[index]); lo, hi = max(0., float(lo)), min(length, float(hi))
            if hi - lo > epsilon:
                relation = -1 if along[index, 1] > along[index, 0] else 1
                matches.append((lo, hi, other['id'], relation))
    positions = sorted(set([0., length] + [x for m in matches for x in m[:2]]))
    gaps, ambiguities, relations = [], [], []
    for lo, hi in zip(positions[:-1], positions[1:]):
        if hi - lo <= epsilon:
            continue
        owners = {(m[2], m[3]) for m in matches if m[0] <= (lo + hi) / 2 <= m[1]}
        if not owners:
            gaps.append(np.stack((a + direction * lo, a + direction * hi)))
        elif len({x[0] for x in owners}) > 1:
            ambiguities.append(dict(start_mm=(a + direction * lo).tolist(),
                end_mm=(a + direction * hi).tolist(), alternative_face_ids=sorted({x[0] for x in owners})))
        for owner, relation in owners:
            relations.append((owner, relation))
    return gaps, ambiguities, relations, length


def _validate_raw_identity(patches, provider):
    if provider is not None:
        provider = np.asarray(provider)
        if provider.ndim != 2 or provider.shape[1] != 3 or not np.isfinite(provider).all():
            raise ValueError('Finite original raw-coordinate provider required')
        for patch in patches:
            if patch.raw_ids is not None:
                if (patch.raw_ids >= len(provider)).any() or not np.array_equal(provider[patch.raw_ids], patch.raw_xyz):
                    raise ValueError('Raw-row ID does not match original provider coordinates')
        return
    seen = {}
    for patch in patches:
        if patch.raw_ids is None:
            continue
        for index, xyz in zip(patch.raw_ids, patch.raw_xyz):
            if int(index) in seen and not np.array_equal(seen[int(index)], xyz):
                raise ValueError('Shared raw-row ID has conflicting raw XYZ')
            seen[int(index)] = xyz


def iter_material_closure_candidates(patches, intervals, original_raw_xyz=None, config=ClosureConfig()):
    """Yield (JSON evidence, complete boundary/gap arrays), one interval at a time."""
    values = np.asarray([config.numerical_epsilon_mm, config.near_seam_tolerance_mm, config.parallel_cosine])
    if (not np.isfinite(values).all() or (values <= 0).any() or config.parallel_cosine > 1
            or isinstance(config.triangle_chunk, bool) or not isinstance(config.triangle_chunk, int) or config.triangle_chunk < 1):
        raise ValueError('Valid finite closure config required')
    patches, intervals = tuple(patches), tuple(intervals)
    lookup = {p.id: p for p in patches}
    if len(lookup) != len(patches) or len({i.id for i in intervals}) != len(intervals):
        raise ValueError('Explicit unique patch and interval IDs required')
    _validate_raw_identity(patches, original_raw_xyz)
    eps = config.numerical_epsilon_mm
    for interval in intervals:
        if interval.left_face_id not in lookup or interval.right_face_id not in lookup or interval.left_face_id == interval.right_face_id:
            raise ValueError('Two existing different main-face IDs required')
        a, b = lookup[interval.left_face_id], lookup[interval.right_face_id]
        record = dict(version=VERSION, material_candidate_id=interval.id,
            main_face_ids=[a.id, b.id], all_original_interval_candidates_retained=True,
            hidden_cavities_ruled_out=False, material_filled_verified=False,
            self_intersections_ruled_out=False, actual_solid_identity_verified=False,
            reviewed_customer_truth=False, actual_material_volume_mm3=None)
        if not len(a.raw_xyz) or not len(b.raw_xyz) or abs(a.normal @ b.normal) < config.parallel_cosine:
            record.update(status='UNKNOWN_MAIN_FACE_GEOMETRY', observed_shell_volume_mm3=None, sidewall_alternatives=[])
            yield record, {}; continue
        reference = _project(a.raw_xyz[:1], a)[0]
        delta = float(b.offset_mm - reference @ b.normal)
        if abs(delta) <= eps:
            record.update(status='UNKNOWN_COINCIDENT_MAIN_FACES', observed_shell_volume_mm3=None, sidewall_alternatives=[])
            yield record, {}; continue
        sa = -np.sign(delta) * np.sign(a.normal @ b.normal); sb = np.sign(delta)
        halfspaces = [(sa * a.normal, sa * a.offset_mm), (sb * b.normal, sb * b.offset_mm)]
        main = [_clipped_patch(a, halfspaces, reference, config), _clipped_patch(b, halfspaces, reference, config)]
        prospective_walls, wall_records = [], []
        for face in patches:
            if face.id in (a.id, b.id) or abs(face.normal @ a.normal) >= config.parallel_cosine:
                continue
            wall = _clipped_patch(face, halfspaces, reference, config)
            coverage = []
            for main_patch in main:
                total, missing = 0., 0.
                for edge in main_patch['boundary']:
                    gaps, _, _, length = _coverage(edge, [wall], eps)
                    total += length; missing += sum(np.linalg.norm(g[1] - g[0]) for g in gaps)
                coverage.append(max(0., total - missing))
            direct = wall['area'] > 0 and all(x > eps for x in coverage)
            wall_records.append(dict(face_id=face.id, clipped_supported_area_mm2=wall['area'],
                clipped_triangle_count=wall['triangle_count'], clipped_boundary_segments=len(wall['boundary']),
                main_boundary_match_lengths_mm=coverage, directly_touches_both_main_boundaries=direct,
                eligible_in_incident_sidewall_chain=False,
                status='PARTIAL_OR_NO_TWO_MAIN_BOUNDARY_SUPPORT',
                derived_cut_vertices_are_original_raw_ids=False, actual_material_identity_verified=False))
            if wall['area'] > 0:
                prospective_walls.append(wall)
        # Entire walls may be split into several sampled patches. Retain every
        # finite wall patch connected by boundary-segment support to either main,
        # then follow all wall-to-wall alternatives. Never select a best cover.
        walls, frontier = [], list(main)
        remaining = list(prospective_walls)
        while frontier:
            following = []
            for wall in remaining:
                linked = any(_coverage(edge, frontier, eps)[2] for edge in wall['boundary'])
                if linked:
                    walls.append(wall); following.append(wall)
            reached = {p['id'] for p in following}
            remaining = [p for p in remaining if p['id'] not in reached]
            frontier = following
        reached = {p['id'] for p in walls}
        for row in wall_records:
            if row['face_id'] in reached:
                row['eligible_in_incident_sidewall_chain'] = True
                row['status'] = 'SUPPORTED_SIDEWALL_CHAIN_ALTERNATIVE'
        shell = main + walls
        arrays, gaps, ambiguity, relations = {}, [], [], {}
        for patch in prospective_walls:
            arrays[f'face_{patch["id"]}_candidate_boundary_segments_mm'] = patch['boundary']
        for patch in shell:
            arrays[f'face_{patch["id"]}_candidate_boundary_segments_mm'] = patch['boundary']
            others = [p for p in shell if p['id'] != patch['id']]
            relations[patch['id']] = []
            for edge in patch['boundary']:
                uncovered, competing, linked, _ = _coverage(edge, others, eps)
                gaps.extend(uncovered); ambiguity.extend(competing)
                relations[patch['id']].extend(linked)
        arrays['all_uncovered_boundary_segments_mm'] = np.asarray(gaps, np.float64).reshape(-1, 2, 3)
        signs, conflict = {a.id: int(sa), b.id: int(sb)}, False
        todo = [a.id, b.id]
        while todo:
            current = todo.pop()
            for target, relation in relations[current]:
                wanted = signs[current] * relation
                if target in signs:
                    conflict |= signs[target] != wanted
                else:
                    signs[target] = wanted; todo.append(target)
        connected = set(); todo = [a.id]
        while todo:
            current = todo.pop()
            if current in connected:
                continue
            connected.add(current); todo.extend(t for t, _ in relations[current] if t not in connected)
        closed = (len(shell) >= 4 and not gaps and not ambiguity and not conflict
                  and len(connected) == len(shell) and all(p['triangle_count'] > 0 for p in shell)
                  and all(p['degenerate_triangles'] == 0 and p['source_nonmanifold_edges'] == 0
                          and p['unknown_boundary_directions'] == 0
                          and p['duplicate_source_triangles'] == 0 and p['conflicting_source_two_sided_edges'] == 0
                          and p['raw_vertex_plane_residual_max_mm'] <= config.near_seam_tolerance_mm for p in shell))
        algebraic = sum(signs.get(p['id'], 0) * p['volume'] for p in shell)
        record.update(status='OBSERVED_BOUNDARY_CLOSED_ORIENTABLE_SHELL_CANDIDATE' if closed else 'UNKNOWN_OPEN_OR_AMBIGUOUS_SHELL',
            main_clipped_support=[dict(face_id=p['id'], area_mm2=p['area'], triangles=p['triangle_count'],
                                      boundary_segments=len(p['boundary'])) for p in main],
            sidewall_alternatives=wall_records, shell_sidewall_face_ids=[p['id'] for p in walls],
            uncovered_boundary_segment_count=len(gaps),
            uncovered_boundary_length_mm=float(sum(np.linalg.norm(g[1] - g[0]) for g in gaps)),
            boundary_competing_cover_count=len(ambiguity), boundary_competing_covers=ambiguity,
            boundary_orientation_conflict=bool(conflict), patch_incidence_graph_connected=len(connected) == len(shell),
            observed_boundary_closed=closed, observed_shell_volume_mm3=abs(float(algebraic)) if closed else None,
            algebraic_oriented_patch_integral_mm3=float(algebraic), oriented_patch_signs={str(k): v for k, v in signs.items()},
            shell_connected_after_clipping_verified=False,
            volume_scope='ALGEBRAIC_OBSERVED_SAMPLE_SHELL_ONLY_NOT_FILLED_MATERIAL_VOLUME',
            numerical_coordinate_stitch_is_candidate_local=True, different_candidates_may_share_face_or_raw_ids=True,
            boundary_gaps_array_key='all_uncovered_boundary_segments_mm',
            near_seam_tolerance_only_diagnostic_not_used_to_close_mm=config.near_seam_tolerance_mm,
            raw_coordinates_changed=False, projected_triangle_coordinates_separate=True,
            raw_vertex_plane_residuals_mm={str(p['id']): p['raw_vertex_plane_residual_max_mm'] for p in shell},
            duplicate_source_triangle_counts={str(p['id']): p['duplicate_source_triangles'] for p in shell},
            source_directed_edge_conflicts={str(p['id']): p['conflicting_source_two_sided_edges'] for p in shell},
            source_plane_support_tolerance_mm=config.near_seam_tolerance_mm,
            unknown_reasons=['ACTUAL_SOLID_IDENTITY_UNREVIEWED', 'MATERIAL_FILL_AND_HIDDEN_CAVITIES_UNKNOWN',
                'SELF_INTERSECTIONS_NOT_CERTIFIED', 'CONNECTED_TRIANGLE_SHELL_AFTER_CLIPPING_NOT_CERTIFIED'])
        yield record, arrays


def analyze_material_closure(patches, intervals, original_raw_xyz=None, config=ClosureConfig()):
    """Convenience for tiny inputs; large pilots should consume the iterator."""
    return list(iter_material_closure_candidates(patches, intervals, original_raw_xyz, config))
