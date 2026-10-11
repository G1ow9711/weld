"""Finite current-XYZ material supports; holes and disconnected faces survive.

Normals supplied by the unchanged pure-XYZ candidate factory propose planes.
Raster cells are uncertain material supports, not approved CAD boundaries.
Original row indices and duplicate rows are retained. No top-K plate pruning.
Only the measured two-dimensional core gate changes; other rules match522.
"""
from itertools import product
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage

VERSION = "CORE_SEEDED_BOUNDARY_ENDPOINT_LOCAL_SUPPORT_XYZ_V3"
HYPOTHESIS_FEATURE_COUNT = 16


def frame(normal):
    n = np.asarray(normal, np.float64).copy()
    length = float(np.linalg.norm(n))
    if length < 1e-8 or not np.isfinite(n).all():
        raise ValueError("Unresolved plane normal")
    n /= length
    if n[np.argmax(np.abs(n))] < 0:
        n = -n
    axis = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, axis); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, np.stack((u, v), -1)


def connected_cells(cells):
    """Sparse adjacency; never allocates a point-by-point adjacency matrix."""
    lookup = {tuple(cell): i for i, cell in enumerate(cells)}
    parent = np.arange(len(cells))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = int(parent[i])
        return i
    for i, (a, b) in enumerate(cells):
        for da, db in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
            j = lookup.get((int(a + da), int(b + db)))
            if j is not None:
                parent[find(i)] = find(j)
    roots = np.asarray([find(i) for i in range(len(cells))])
    _, labels = np.unique(roots, return_inverse=True)
    return labels, lookup


def rotation_aware_material_core(cells, unique_uv, origin, resolution, transverse_tolerance_mm):
    """Actual noncollinear UV support, including rotated/diagonal neighbours.

    Empty cells are never filled. One-dimensional intersecting edge rows stay
    unresolved and cannot provide a bridge between finite material faces.
    Cell connectivity and boundary/uncertainty rules remain unchanged.
    """
    unique_cells = np.floor((unique_uv-origin)/resolution).astype(np.int64)
    occupied, inverse = np.unique(unique_cells, axis=0, return_inverse=True)
    assert np.array_equal(occupied,cells)
    counts = np.bincount(inverse)
    representatives = np.column_stack([np.bincount(inverse,weights=unique_uv[:,j])/counts for j in range(2)])
    lookup = {tuple(cell):i for i,cell in enumerate(cells)}
    core = np.zeros(len(cells),bool)
    for i,(a,b) in enumerate(cells):
        neighbours = [lookup[(int(a+da),int(b+db))] for da,db in product(range(-2,3),repeat=2)
                      if (da or db) and (int(a+da),int(b+db)) in lookup]
        if len(neighbours)<2:
            continue
        vectors = representatives[neighbours]-representatives[i]
        lengths = np.linalg.norm(vectors,axis=1)
        vectors = vectors[(lengths>1e-7)&(lengths<=2.01*resolution)]
        if len(vectors)<2:
            continue
        longest = vectors[np.linalg.norm(vectors,axis=1).argmax()]
        transverse = np.abs(vectors[:,0]*longest[1]-vectors[:,1]*longest[0])/np.linalg.norm(longest)
        core[i] = float(transverse.max())>transverse_tolerance_mm
    return core


def physical_hypothesis_metadata(surface,value,station,kind,hole_region=None):
    if 'original_support_rows_sha256' not in surface:
        surface['original_support_rows_sha256'] = hashlib.sha256(np.asarray(surface['original_row_indices'],np.int64).tobytes()).hexdigest()
    source = [surface['original_support_rows_sha256'],np.asarray(station,np.float64).tolist(),kind]
    if hole_region is not None:
        source.append(hashlib.sha256(np.asarray(hole_region,np.int64).tobytes()).hexdigest())
    key = hashlib.sha256(repr(source).encode('utf-8')).hexdigest()
    uncertainty = float(surface['field_uncertainty_mm'])
    return dict(physical_source_key=key,support_interval_mm=[max(0.,value-uncertainty),value+uncertainty],
                known_role_constraints_consistent=True,physical_plate_identity_confirmed=False,
                estimated_sampling_uncertainty_not_certified_error_bound=True)


def attach_unique_boundary_cells(cells, core, labels):
    """One-ring occupied boundary support; never propagates or merges cores."""
    original=labels.copy();lookup={tuple(cell):i for i,cell in enumerate(cells)}
    attached=0;ambiguous=0
    for i in np.flatnonzero(~core):
        a,b=cells[i]
        neighbours={int(original[lookup[(int(a+da),int(b+db))]])
            for da,db in product((-1,0,1),repeat=2)
            if (da or db) and (int(a+da),int(b+db)) in lookup
            and original[lookup[(int(a+da),int(b+db))]]>=0}
        if len(neighbours)==1:labels[i]=next(iter(neighbours));attached+=1
        elif len(neighbours)>1:ambiguous+=1
    assert np.array_equal(labels[core],original[core])
    return dict(attached_occupied_boundary_cells=attached,ambiguous_cells_remain_unassigned=ambiguous,
        propagation_rounds=1,core_components_merged=0,empty_cells_added=0)


class FiniteSurfaceIndex:
    def __init__(self, current_original_xyz_mm, *, plane_tolerance_mm=.15,
                 maximum_dense_boundary_cells=2000000):
        self.xyz = np.asarray(current_original_xyz_mm)
        assert self.xyz.ndim == 2 and self.xyz.shape[1] == 3 and np.isfinite(self.xyz).all()
        self.tolerance = float(plane_tolerance_mm)
        self.maximum_dense_boundary_cells = int(maximum_dense_boundary_cells)
        self.planes = {}
        self.original_rows = len(self.xyz)

    def propose(self, normal, point):
        n, basis = frame(normal)
        offset = float(np.asarray(point) @ n)
        key = (*n.round(7), round(offset, 3))
        if key in self.planes:
            return self.planes[key]
        rows = []
        for a in range(0, len(self.xyz), 262144):
            p = self.xyz[a:a + 262144]
            rows.append(np.flatnonzero(np.abs(p @ n - offset) <= self.tolerance) + a)
        rows = np.concatenate(rows)
        if len(rows) < 4:
            self.planes[key] = []
            return []
        points = self.xyz[rows]
        uv = np.asarray(points @ basis, np.float64)
        unique_uv = np.unique(uv, axis=0)
        if len(unique_uv) < 4:
            self.planes[key] = []
            return []
        tree = cKDTree(unique_uv)
        sample = unique_uv[np.linspace(0, len(unique_uv) - 1, min(4096, len(unique_uv)), dtype=int)]
        spacing = tree.query(sample, k=2)[0][:, 1]
        spacing = spacing[spacing > 1e-7]
        if not len(spacing):
            self.planes[key] = []
            return []
        resolution = float(np.median(spacing))
        origin = uv.min(0) - resolution * 1e-7
        row_cells = np.floor((uv - origin) / resolution).astype(np.int64)
        cells, inverse = np.unique(row_cells, axis=0, return_inverse=True)
        lookup = set(map(tuple, cells.tolist()))
        # Coplanar edge rows from an intersecting base must not bridge two
        # separate upright faces. A face cell needs support in two axes.
        core = rotation_aware_material_core(cells, unique_uv, origin, resolution, self.tolerance)
        labels = np.full(len(cells), -1, int)
        if core.any():
            labels[core] = connected_cells(cells[core])[0]
        boundary_attachment = attach_unique_boundary_cells(cells,core,labels)
        surfaces = []
        for label in range(int(labels.max()) + 1):
            take = np.flatnonzero(labels == label)
            occupied = cells[take]
            mask = labels[inverse] == label
            support_rows = rows[mask]
            lo, hi = occupied.min(0), occupied.max(0)
            shape = hi - lo + 3
            boundary_kind, inner_regions = "UNRESOLVED", []
            outer_cells, inner_cells = [], []
            if int(np.prod(shape)) <= self.maximum_dense_boundary_cells:
                material = np.zeros(tuple(shape), bool)
                pos = occupied - lo + 1
                material[pos[:, 0], pos[:, 1]] = True
                free, count = ndimage.label(~material)
                exterior = set(np.r_[free[0], free[-1], free[:, 0], free[:, -1]].tolist())
                holes = [j for j in range(1, count + 1) if j not in exterior]
                for hole in holes:
                    inner_regions.append(np.argwhere(free == hole) + lo - 1)
                for cell, position in zip(occupied, pos):
                    neighbours = [free[position[0] + da, position[1] + db]
                                  for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1))]
                    if any(j in exterior for j in neighbours if j):
                        outer_cells.append(cell)
                    if any(j not in exterior for j in neighbours if j):
                        inner_cells.append(cell)
                boundary_kind = "FINITE_OUTER_AND_INNER_SUPPORTS"
            surface = dict(original_row_indices=support_rows, occupied_cells=occupied,
                normal=n, basis=basis, plane_offset_mm=offset, uv_origin=origin,
                resolution_mm=resolution, component_index=label, original_rows=len(support_rows),
                unique_material_cells=len(occupied), boundary_kind=boundary_kind,
                outer_boundary_cells=np.asarray(outer_cells, np.int64).reshape(-1, 2),
                inner_boundary_cells=np.asarray(inner_cells, np.int64).reshape(-1, 2),
                inner_free_regions=inner_regions, normal_sign_status="UNRESOLVED_CANONICAL_PROPOSAL_ONLY",
                two_dimensional_material_support=True,
                maximum_plane_residual_mm=float(np.abs(self.xyz[support_rows] @ n - offset).max()),
                field_uncertainty_mm=2. * resolution + self.tolerance,
                boundary_attachment=boundary_attachment)
            surface["occupied_lookup"] = set(map(tuple, occupied.tolist()))
            surfaces.append(surface)
        unresolved = labels < 0
        if unresolved.any():
            mask = labels[inverse] == -1
            # Preserve orphan row IDs and the unresolved proposal. They cannot
            # assert a finite face or silently connect different plate roles.
            surfaces.append(dict(original_row_indices=rows[mask], occupied_cells=cells[unresolved],
                normal=n, basis=basis, plane_offset_mm=offset, uv_origin=origin,
                resolution_mm=resolution, component_index=-1, original_rows=int(mask.sum()),
                unique_material_cells=int(unresolved.sum()), boundary_kind="UNRESOLVED",
                outer_boundary_cells=np.empty((0, 2), int), inner_boundary_cells=np.empty((0, 2), int),
                inner_free_regions=[], normal_sign_status="UNRESOLVED_CANONICAL_PROPOSAL_ONLY",
                two_dimensional_material_support=False, maximum_plane_residual_mm=self.tolerance,
                field_uncertainty_mm=1e99, occupied_lookup=set(map(tuple, cells[unresolved].tolist())),
                unresolved_reason="ONE_DIMENSIONAL_OVERLAP_ROWS_NOT_A_FINITE_MATERIAL_FACE"))
        # Every component is retained, including ones not touching this weld.
        assert sum(s["original_rows"] for s in surfaces) == len(rows)
        self.planes[key] = surfaces
        return surfaces

    def touches(self, surface, point):
        if abs(float(np.asarray(point) @ surface["normal"] - surface["plane_offset_mm"])) > self.tolerance:
            return False
        uv = np.asarray(point) @ surface["basis"]
        cell = np.floor((uv - surface["uv_origin"]) / surface["resolution_mm"]).astype(int)
        # Adjacent cell allowance is explicitly included in uncertainty.
        return any((int(cell[0] + da), int(cell[1] + db)) in surface["occupied_lookup"]
                   for da, db in product((-1, 0, 1), repeat=2))

    def section(self, surface, point, direction):
        """Actual same-component points in a finite cross-section band."""
        u = np.asarray(direction, float); u /= np.linalg.norm(u)
        cross = np.cross(surface["normal"], u)
        if np.linalg.norm(cross) < 1e-8:
            return None
        cross /= np.linalg.norm(cross)
        key = tuple(cross.round(7))
        cache = surface.setdefault("section_projection_index", {})
        if key not in cache:
            rows = surface["original_row_indices"]
            scalar = self.xyz[rows] @ cross
            order = np.argsort(scalar, kind="stable")
            cache[key] = scalar[order], rows[order]
        scalar, rows = cache[key]
        center = float(np.asarray(point) @ cross)
        width = surface["resolution_mm"] * 1.5
        a, z = np.searchsorted(scalar, [center - width, center + width])
        if a == z:
            return None
        values = (self.xyz[rows[a:z]] - np.asarray(point)) @ u
        return float(values.min()), float(values.max())

    def ordered_hypotheses(self, sequence, a_mm, z_mm):
        q = np.asarray(sequence["xyz"], float)
        step = float(sequence["sample_step_mm"])
        positions = np.asarray([a_mm, (a_mm + z_mm) / 2., z_mm])
        chart = np.arange(len(q)) * step
        stations = np.column_stack([np.interp(positions, chart, q[:, j]) for j in range(3)])
        axes = np.asarray(sequence["features"])[len(q) // 2, 60:66].reshape(2, 3)
        proposed = [self.propose(n, stations[1]) if np.linalg.norm(n) > 1e-8 else [] for n in axes]
        fields = [[] for _ in range(16)]
        pairs = []
        local_pairs=[]
        for endpoint_index in (0,2):
            touching=[[s for s in surfaces if self.touches(s,stations[endpoint_index])] for surfaces in proposed]
            local_pairs.extend((endpoint_index,first,second) for first,second in product(*touching))
        for endpoint_index, first, second in local_pairs:
            if abs(float(first["normal"] @ second["normal"])) > .15:
                continue
            for base, upright in ((first, second), (second, first)):
                # Roles are geometric hypotheses. Plate1/2 order and signed
                # export normals remain unresolved without their definitions.
                role_known = (base["two_dimensional_material_support"] and upright["two_dimensional_material_support"] and
                    abs(float(base["normal"][2])) >= .95 and abs(float(upright["normal"][2])) <= .15)
                pair = dict(base=base, upright=upright, geometric_role_known=role_known,
                    plate1_plate2_order_status="UNRESOLVED", normal_sign_status="UNRESOLVED",
                    endpoint_index=endpoint_index,whole_path_plate_identity_confirmed=False)
                pairs.append(pair)
                if not role_known:
                    continue
                for station, field in ((stations[endpoint_index],6 if endpoint_index==0 else 7),):
                    section = self.section(upright, station, [0., 0., 1.])
                    if section is None:
                        continue
                    height = section[1] - section[0]
                    uncertainty = upright["field_uncertainty_mm"]
                    valid = upright["boundary_kind"] == "FINITE_OUTER_AND_INNER_SUPPORTS" and uncertainty <= 3.
                    fields[field].append(dict(physical_value_mm=height, field_valid=valid,
                        boundary_kind="ASSOCIATED_UPRIGHT_OUTER_SECTION_HEIGHT",
                        order_status="ROLE_UPRIGHT_BASE_GEOMETRIC", sign_status="UNRESOLVED",
                        uncertainty_mm=uncertainty, features=hypothesis_features(upright, height, valid),
                        **physical_hypothesis_metadata(upright,height,station,'OUTER_HEIGHT')))
                for station, field in ((stations[endpoint_index],8 if endpoint_index==0 else 9),):
                    for region in upright["inner_free_regions"]:
                        uv = upright["uv_origin"] + (region + .5) * upright["resolution_mm"]
                        points = uv @ upright["basis"].T + upright["normal"] * upright["plane_offset_mm"]
                        delta = points - station
                        band = np.linalg.norm(delta[:, :2], axis=1) <= upright["resolution_mm"] * 1.5
                        if not band.any():
                            continue
                        height = float(np.ptp(delta[band, 2]) + upright["resolution_mm"])
                        uncertainty = upright["field_uncertainty_mm"]
                        valid = uncertainty <= 3.
                        fields[field].append(dict(physical_value_mm=height, field_valid=valid,
                            boundary_kind="ASSOCIATED_FINITE_INNER_HOLE_HEIGHT",
                            order_status="ROLE_UPRIGHT_BASE_GEOMETRIC", sign_status="UNRESOLVED",
                            uncertainty_mm=uncertainty, features=hypothesis_features(upright, height, valid),
                            **physical_hypothesis_metadata(upright,height,station,'INNER_HOLE_HEIGHT',region)))
        return dict(fields=fields, pairs=pairs, stations_mm=stations,
                    all_ragged_hypotheses_retained=True, unresolved_fields_explicit=True,
                    endpoint_fields_use_endpoint_local_contact=True,whole_path_plate_identity_confirmed=False)


def hypothesis_features(surface, value, valid):
    return np.asarray([value / 1000., surface["resolution_mm"] / 10.,
        surface["maximum_plane_residual_mm"], surface["original_rows"] / 100000.,
        surface["unique_material_cells"] / 100000., len(surface["inner_free_regions"]),
        abs(float(surface["normal"][2])), float(valid),
        float(surface["boundary_kind"] != "UNRESOLVED"), 0., 0., 0., 1., 0., 0., 0.], np.float32)


def build_finite_surface_index(current_original_xyz_mm, global_config=None):
    return FiniteSurfaceIndex(current_original_xyz_mm, **(global_config or {}))


def query_ordered_plate_hypotheses(curve, stations, finite_surface_index):
    return finite_surface_index.ordered_hypotheses(curve, float(stations[0]), float(stations[-1]))
