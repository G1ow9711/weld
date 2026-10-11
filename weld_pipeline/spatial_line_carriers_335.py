"""Finite 3D line hypotheses from local original-XYZ linear evidence.

Class hypotheses remain unselected. PCA directions are geometric evidence,
not reviewed plate normals, contact or welding-side truth. Old banks are not
pruned. Every interval is bounded by observed points without GT extensions.
"""
import numpy as np
from scipy.spatial import cKDTree

VERSION = "PURE_XYZ_MULTISCALE_SPATIAL_LINE_FINITE_SUPPORT_335_v1"
NEIGHBORS = (4,8,16)
LOCAL_LINEAR_RATIO = .002
LINE_DISTANCE_MM = .12
MINIMUM_LINE_MM = 5.


def local_linear_seeds(xyz,progress):
    count = len(xyz)
    tree = cKDTree(xyz)
    directions = np.zeros((count,3),float)
    centers = np.zeros((count,3),float)
    ratios = np.full(count,np.inf,float)
    spreads = np.zeros(count,float)
    for begin in range(0,count,10000):
        stop = min(count,begin+10000)
        _,indices = tree.query(xyz[begin:stop],k=min(max(NEIGHBORS),count),workers=2)
        best_ratio = ratios[begin:stop]
        best_spread = spreads[begin:stop]
        for size in NEIGHBORS:
            if size > indices.shape[1]:
                continue
            neighbors = xyz[indices[:,:size]]
            center = neighbors.mean(axis=1)
            delta = neighbors-center[:,None,:]
            covariance = np.einsum("nki,nkj->nij",delta,delta)/size
            values,vectors = np.linalg.eigh(covariance)
            ratio = np.divide(np.maximum(values[:,1],0.),np.maximum(values[:,2],1e-30))
            along = np.sort(np.einsum("nki,ni->nk",delta,vectors[:,:,-1]),axis=1)
            distinct = 1+np.sum(np.diff(along,axis=1) > 1e-6,axis=1)
            valid = (values[:,2] > .25) & (ratio <= LOCAL_LINEAR_RATIO) & (distinct >= 3)
            better = valid & ((ratio < best_ratio-1e-14) |
                ((np.abs(ratio-best_ratio) <= 1e-14) & (values[:,2] > best_spread)))
            chosen = vectors[better,:,-1]
            if len(chosen):
                major = np.argmax(np.abs(chosen),axis=1)
                chosen *= np.where(chosen[np.arange(len(chosen)),major] < 0.,-1.,1.)[:,None]
            directions[begin:stop][better] = chosen
            centers[begin:stop][better] = center[better]
            best_ratio[better] = ratio[better]
            best_spread[better] = values[better,2]
        if begin % 100000 == 0 or stop == count:
            progress("PURE_XYZ_LOCAL_3D_LINE_EVIDENCE",original_rows_examined=stop,
                original_rows_total=count,accepted_local_linear_rows=int(np.isfinite(ratios[:stop]).sum()))
    accepted = np.flatnonzero(np.isfinite(ratios))
    ordering = accepted[np.lexsort((accepted,-spreads[accepted],ratios[accepted]))]
    del tree,ratios,spreads
    return directions,centers,ordering


def line_support(xyz,origin,direction):
    delta = xyz-origin
    parameters = delta @ direction
    residual = delta-parameters[:,None]*direction
    rows = np.flatnonzero(np.einsum("ni,ni->n",residual,residual) <= LINE_DISTANCE_MM**2)
    return rows,parameters[rows]


def spatial_line_carriers(xyz_mm,progress=lambda *args,**kwargs:None):
    """Return a lazy emitter and mutable statistics after pure XYZ evidence."""
    xyz = np.asarray(xyz_mm,dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 8 or not np.isfinite(xyz).all():
        raise ValueError("All finite original XYZ required")
    directions,centers,ordering = local_linear_seeds(xyz,progress)
    statistics = dict(version=VERSION,original_XYZ=len(xyz),local_neighbors=list(NEIGHBORS),
        local_linear_ratio_limit=LOCAL_LINEAR_RATIO,line_support_distance_mm=LINE_DISTANCE_MM,
        minimum_line_length_mm=MINIMUM_LINE_MM,accepted_local_linear_rows=len(ordering),
        minimum_distinct_local_linear_positions=3,
        fitted_observed_lines=0,finite_observed_domains=0,horizontal_class_hypotheses=0,
        descending_vertical_class_hypotheses=0,actual_candidates=0,
        target_reads_for_generation=0,STEP_reads=0,source_identifiers_or_filename_branches=False,
        finite_ranges_are_observed_point_minmax_not_GT=True,
        complete_spans_and_observed_components_both_retained=True,
        material_contact_and_weld_side_unverified=True,full_contract_pass=False)

    def emit():
        covered = np.zeros(len(ordering),bool)
        seed_centers,seed_directions = centers[ordering],directions[ordering]
        domain_index = 0
        for position,seed in enumerate(ordering):
            if covered[position]:
                continue
            origin,direction = centers[seed].copy(),directions[seed].copy()
            rows,parameters = line_support(xyz,origin,direction)
            if len(rows) < 2 or np.ptp(parameters) < MINIMUM_LINE_MM:
                covered[position] = True
                continue
            for _ in range(2):
                pts = xyz[rows]
                origin = pts.mean(axis=0)
                delta = pts-origin
                values,vectors = np.linalg.eigh(delta.T @ delta / len(pts))
                refined = vectors[:,-1]
                if refined @ direction < 0.:
                    refined = -refined
                if abs(float(refined @ direction)) < .999:
                    break
                direction = refined
                rows,parameters = line_support(xyz,origin,direction)
                if len(rows) < 2:
                    break
            if len(rows) < 2 or np.ptp(parameters) < MINIMUM_LINE_MM:
                covered[position] = True
                continue
            # Coverage only avoids repeated PCA proposal seeds. It never
            # removes an original point, candidate, parent or label.
            local_delta = seed_centers-origin
            local_q = local_delta @ direction
            local_residual = local_delta-local_q[:,None]*direction
            same_direction = np.abs(seed_directions @ direction) > .9999999
            covered |= same_direction & (np.einsum("ni,ni->n",local_residual,local_residual) <= LINE_DISTANCE_MM**2)
            covered[position] = True
            statistics["fitted_observed_lines"] += 1
            unique = np.unique(parameters)
            differences = np.diff(unique)
            typical = float(np.median(differences[differences > 1.])) if np.any(differences > 1.) else 8.
            boundaries = np.r_[0,np.flatnonzero(differences > max(15.,1.8*typical))+1,len(unique)]
            complete = (float(unique[0]),float(unique[-1]))
            domains = [complete]
            for begin,end in zip(boundaries[:-1],boundaries[1:]):
                if end-begin < 2:
                    continue
                interval = (float(unique[begin]),float(unique[end-1]))
                if interval[1]-interval[0] >= MINIMUM_LINE_MM and interval != complete:
                    domains.append(interval)
            for lower,upper in domains:
                if upper-lower < MINIMUM_LINE_MM:
                    continue
                points = origin+np.array([lower,upper])[:,None]*direction
                supporting = rows[(parameters >= lower) & (parameters <= upper)]
                statistics["finite_observed_domains"] += 1
                common = dict(segments=[dict(kind="line",points=points.tolist())],
                    raw_support_rows=supporting.tolist(),source_kind="RAW_MULTISCALE_3D_LINEAR_SUPPORT",
                    complete_observed_span=bool((lower,upper) == complete),
                    observed_parameter_range_mm=[lower,upper],line_origin_mm=origin.tolist(),
                    line_direction=direction.tolist(),source_local_seed_row_metadata_only=int(seed),
                    material_identity="MATERIAL_UNVERIFIED",weld_identity="UNSELECTED_GEOMETRY_HYPOTHESIS",
                    actual_contact_continuity_and_weld_side_unverified=True,
                    field_normals_or_thickness_or_GT_not_used=True)
                statistics["horizontal_class_hypotheses"] += 1
                statistics["actual_candidates"] += 1
                yield dict(common,id=f"spatial_line_h_{domain_index:07d}",type="horizontal")
                if abs(float(points[-1,2]-points[0,2])) > 1e-10:
                    statistics["descending_vertical_class_hypotheses"] += 1
                    statistics["actual_candidates"] += 1
                    yield dict(common,id=f"spatial_line_v_{domain_index:07d}",type="vertical")
                domain_index += 1
            if statistics["fitted_observed_lines"] % 100 == 0:
                progress("PURE_XYZ_OBSERVED_FINITE_3D_LINE_DOMAINS",
                    fitted_observed_lines=statistics["fitted_observed_lines"],
                    emitted_candidates=statistics["actual_candidates"],local_seed_position=position,
                    all_original_rows_retained=len(xyz))
        assert statistics["actual_candidates"] == statistics["horizontal_class_hypotheses"]+statistics["descending_vertical_class_hypotheses"]

    return emit(),statistics
