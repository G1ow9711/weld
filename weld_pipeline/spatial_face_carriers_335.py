"""Unselected finite face-intersection hypotheses from all original XYZ.

Geometric PCA normals are unsigned surface evidence. They do not establish
plate identity, thickness, material contact, weld side or E16 truth. Convex
support hulls may bridge coplanar disconnected patches: selection/contact
validation remains a subsequent task. No source IDs or reference data enter.
"""
import numpy as np
from scipy.spatial import cKDTree,ConvexHull,QhullError

VERSION = "PURE_XYZ_MULTISCALE_3D_FACE_FINITE_INTERSECTIONS_335_v1"
NEIGHBORS = (8,16,32,64)
PLANAR_RATIO = 1e-5
SUPPORT_MM = .12
MIN_LENGTH_MM = 5.


def planar_evidence(xyz,progress):
    # Exact duplicates are redundant only for neighborhood PCA. Original
    # rows remain intact and every global support test uses all raw rows.
    unique,first = np.unique(xyz,axis=0,return_index=True)
    tree = cKDTree(unique)
    count = len(unique)
    normals,centers = np.zeros((count,3)),np.zeros((count,3))
    ratios,spreads = np.full(count,np.inf),np.zeros(count)
    for start in range(0,count,5000):
        end = min(count,start+5000)
        _,ids = tree.query(unique[start:end],k=min(max(NEIGHBORS),count),workers=2)
        for size in NEIGHBORS:
            if size > ids.shape[1]:
                continue
            points = unique[ids[:,:size]]
            center = points.mean(1)
            delta = points-center[:,None]
            eigenvalues,vectors = np.linalg.eigh(np.einsum("nki,nkj->nij",delta,delta)/size)
            ratio = np.maximum(0.,eigenvalues[:,0])/np.maximum(1e-30,eigenvalues[:,1])
            valid = (eigenvalues[:,1] >= .25) & (ratio <= PLANAR_RATIO)
            current = ratios[start:end]
            better = valid & ((ratio < current-1e-14) |
                ((np.abs(ratio-current) <= 1e-14) & (eigenvalues[:,1] > spreads[start:end])))
            chosen = vectors[better,:,0]
            if len(chosen):
                major = np.argmax(np.abs(chosen),axis=1)
                chosen *= np.where(chosen[np.arange(len(chosen)),major] < 0.,-1.,1.)[:,None]
            normals[start:end][better] = chosen
            centers[start:end][better] = center[better]
            ratios[start:end][better] = ratio[better]
            spreads[start:end][better] = eigenvalues[better,1]
        if start % 50000 == 0 or end == count:
            progress("PURE_XYZ_MULTISCALE_3D_PLANAR_EVIDENCE",unique_positions_examined=end,
                unique_positions=count,original_rows_retained=len(xyz),
                local_planar_positions=int(np.isfinite(ratios[:end]).sum()))
    accepted = np.flatnonzero(np.isfinite(ratios))
    order = accepted[np.lexsort((first[accepted],-spreads[accepted],ratios[accepted]))]
    return normals[order],centers[order],first[order],count


def fit_finite_faces(xyz,progress):
    normals,centers,seeds,unique_count = planar_evidence(xyz,progress)
    covered = np.zeros(len(seeds),bool)
    faces = []
    failed = 0
    for index,seed in enumerate(seeds):
        if covered[index]:
            continue
        normal,center = normals[index].copy(),centers[index].copy()
        rows = np.flatnonzero(np.abs((xyz-center) @ normal) <= SUPPORT_MM)
        valid = len(rows) >= 6
        if valid:
            for _ in range(3):
                points = xyz[rows]
                refined_center = points.mean(0)
                delta = points-refined_center
                values,vectors = np.linalg.eigh(delta.T @ delta/len(points))
                refined = vectors[:,0]
                if refined @ normal < 0.:
                    refined = -refined
                if values[1] < .25 or refined @ normal < .999:
                    valid = False
                    break
                center,normal = refined_center,refined
                rows = np.flatnonzero(np.abs((xyz-center) @ normal) <= SUPPORT_MM)
                if len(rows) < 6:
                    valid = False
                    break
        covered[index] = True
        if not valid:
            failed += 1
            continue
        points = xyz[rows]
        delta = points-center
        values,vectors = np.linalg.eigh(delta.T @ delta/len(points))
        if values[1] < .25 or np.quantile(np.abs(delta @ normal),.9) > .10:
            failed += 1
            continue
        basis = vectors[:,1:]
        projected = delta @ basis
        try:
            hull = ConvexHull(projected)
        except QhullError:
            failed += 1
            continue
        if hull.volume < 5.:
            failed += 1
            continue
        # Avoid repeating nearly identical plane proposals; no candidate or
        # raw data is deleted. An offset test distinguishes parallel faces.
        same = (np.abs(normals @ normal) > .999999) & (np.abs((centers-center) @ normal) <= SUPPORT_MM)
        covered |= same
        faces.append(dict(face_id=len(faces),center_mm=center.tolist(),geometric_normal_unsigned=normal.tolist(),
            tangent_basis=basis.tolist(),hull_equations=hull.equations.tolist(),
            finite_hull_vertices_mm=points[hull.vertices].tolist(),
            hull_projected_vertices_mm=projected[hull.vertices].tolist(),
            hull_original_row_indices=rows[hull.vertices].tolist(),raw_support_rows=rows.tolist(),
            support_count=len(rows),source_local_seed_row_metadata_only=int(seed),
            original_support_AABB_mm=[points.min(0).tolist(),points.max(0).tolist()],
            support_distance_mm=SUPPORT_MM,planar_area_mm2=float(hull.volume),
            material_identity="MATERIAL_UNVERIFIED",oriented_E16_normal_truth=False,
            disconnected_coplanar_support_may_be_bridged=True))
        if len(faces) % 25 == 0:
            progress("PURE_XYZ_FINITE_3D_FACE_SUPPORT",fitted_faces=len(faces),
                local_planar_seed_position=index,all_local_planar_positions=len(seeds),
                failed_seed_fits=failed,original_rows_retained=len(xyz))
    return faces,dict(version=VERSION,original_rows=len(xyz),unique_PCA_positions=unique_count,
        local_planar_positions=len(seeds),fitted_finite_faces=len(faces),failed_seed_fits=failed,
        neighbor_sizes=list(NEIGHBORS),planar_ratio_limit=PLANAR_RATIO,
        support_distance_mm=SUPPORT_MM,target_reads_for_generation=0,STEP_reads=0,
        exact_duplicate_positions_only_for_neighborhood_fitting=True,
        all_original_rows_retained_for_support=True,full_contract_pass=False)


def finite_hull_parameter_interval(face,origin,direction):
    center = np.asarray(face["center_mm"],float)
    basis = np.asarray(face["tangent_basis"],float)
    equations = np.asarray(face["hull_equations"],float)
    position = (origin-center) @ basis
    slope = direction @ basis
    constants = equations[:,:2] @ position+equations[:,2]
    coefficients = equations[:,:2] @ slope
    lower,upper = -np.inf,np.inf
    for coefficient,constant in zip(coefficients,constants):
        if abs(coefficient) <= 1e-12:
            if constant > 1e-8:
                return None
        elif coefficient > 0.:
            upper = min(upper,float(-constant/coefficient))
        else:
            lower = max(lower,float(-constant/coefficient))
    if not np.isfinite((lower,upper)).all() or upper <= lower:
        return None
    return lower,upper


def spatial_face_carriers(xyz_mm,progress=lambda *args,**kwargs:None):
    xyz = np.asarray(xyz_mm,float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Complete finite original XYZ required")
    faces,statistics = fit_finite_faces(xyz,progress)
    statistics.update(actual_candidates=0,nonparallel_face_pairs=0,finite_intersection_domains=0,
        observed_raw_line_support_domains=0,face_hull_boundary_domains=0,
        horizontal_class_hypotheses=0,descending_vertical_class_hypotheses=0,
        minimum_line_length_mm=MIN_LENGTH_MM,source_or_filename_branches=False,
        finite_intervals_never_extended_from_GT=True,material_contact_and_weld_side_unverified=True)

    def emit():
        domain_index = 0
        def records(origin,direction,lower,upper,rows,kind,ids):
            nonlocal domain_index
            if upper-lower < MIN_LENGTH_MM:
                return
            points = origin+np.array([lower,upper])[:,None]*direction
            base = dict(segments=[dict(kind="line",points=points.tolist())],
                raw_support_rows=rows.tolist(),source_kind=kind,plane_ids=ids,
                line_origin_mm=origin.tolist(),line_direction=direction.tolist(),
                observed_parameter_range_mm=[float(lower),float(upper)],
                material_identity="MATERIAL_UNVERIFIED",weld_identity="UNSELECTED_GEOMETRY_HYPOTHESIS",
                geometric_normals_not_oriented_E16_truth=True,actual_contact_and_weld_side_unverified=True)
            statistics["actual_candidates"] += 1
            statistics["horizontal_class_hypotheses"] += 1
            yield dict(base,id=f"spatial_face_h_{domain_index:07d}",type="horizontal")
            if abs(float(points[1,2]-points[0,2])) > 1e-10:
                statistics["actual_candidates"] += 1
                statistics["descending_vertical_class_hypotheses"] += 1
                yield dict(base,id=f"spatial_face_v_{domain_index:07d}",type="vertical")
            domain_index += 1

        # Observed finite hull boundaries supply geometric edge hypotheses
        # even when a small adjacent face has insufficient planar evidence.
        for face in faces:
            points = np.asarray(face["finite_hull_vertices_mm"],float)
            ids = np.asarray(face["hull_original_row_indices"],np.int64)
            for j in range(len(points)):
                origin = points[j]
                direction = points[(j+1) % len(points)]-origin
                length = float(np.linalg.norm(direction))
                if length < MIN_LENGTH_MM:
                    continue
                direction /= length
                statistics["face_hull_boundary_domains"] += 1
                yield from records(origin,direction,0.,length,ids[[j,(j+1) % len(ids)]],
                    "RAW_FINITE_3D_FACE_HULL_BOUNDARY",[face["face_id"]])

        for i,first in enumerate(faces):
            for second in faces[i+1:]:
                na = np.asarray(first["geometric_normal_unsigned"],float)
                nb = np.asarray(second["geometric_normal_unsigned"],float)
                direction = np.cross(na,nb)
                sine = float(np.linalg.norm(direction))
                if sine < .05:
                    continue
                statistics["nonparallel_face_pairs"] += 1
                direction /= sine
                major = int(np.argmax(np.abs(direction)))
                if direction[major] < 0.:
                    direction = -direction
                center = (np.asarray(first["center_mm"])+np.asarray(second["center_mm"]))/2
                matrix = np.array([na,nb,direction])
                rhs = np.array([np.asarray(first["center_mm"]) @ na,
                    np.asarray(second["center_mm"]) @ nb,center @ direction])
                origin = np.linalg.solve(matrix,rhs)
                ia = finite_hull_parameter_interval(first,origin,direction)
                ib = finite_hull_parameter_interval(second,origin,direction)
                if ia is None or ib is None:
                    continue
                lower,upper = max(ia[0],ib[0]),min(ia[1],ib[1])
                if upper-lower < MIN_LENGTH_MM:
                    continue
                # Finite intersected hulls already bound the proposal. Only
                # original points from these face supports form line support.
                support = np.union1d(first["raw_support_rows"],second["raw_support_rows"])
                delta = xyz[support]-origin
                q = delta @ direction
                residual = delta-q[:,None]*direction
                keep = (np.einsum("ni,ni->n",residual,residual) <= SUPPORT_MM**2)
                rows = support[keep]
                parameters = q[keep]
                statistics["finite_intersection_domains"] += 1
                yield from records(origin,direction,lower,upper,rows,
                    "RAW_FINITE_3D_FACE_HULL_INTERSECTION",[first["face_id"],second["face_id"]])
                if len(rows) >= 3 and np.ptp(parameters) >= MIN_LENGTH_MM:
                    raw_lower,raw_upper = float(parameters.min()),float(parameters.max())
                    statistics["observed_raw_line_support_domains"] += 1
                    yield from records(origin,direction,raw_lower,raw_upper,rows,
                        "RAW_3D_FACE_INTERSECTION_ORIGINAL_LINE_SUPPORT",[first["face_id"],second["face_id"]])
            if i % 10 == 0 or i+1 == len(faces):
                progress("PURE_XYZ_FINITE_3D_FACE_INTERSECTIONS",complete_first_faces=i+1,
                    total_finite_faces=len(faces),actual_candidates=statistics["actual_candidates"],
                    nonparallel_face_pairs=statistics["nonparallel_face_pairs"],all_original_rows_retained=len(xyz))
        assert statistics["actual_candidates"] == statistics["horizontal_class_hypotheses"]+statistics["descending_vertical_class_hypotheses"]
    return emit(),statistics,faces
