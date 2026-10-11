"""One-sided material and direct XYZ traces on finite face intersections.

These are geometric evidence, not a metrology certificate or an association to
an exported field. Interior sampled layers and missing edge rows stay explicit.
"""
from collections import defaultdict
import numpy as np

VERSION='ONE_SIDED_DIRECT_FINITE_EDGE_AND_OPPOSED_HEIGHT_V1'


def face_trace(xyz, rows, face, origin, direction, interval, tolerance=.15):
    points=np.asarray(xyz[rows],np.float64);delta=points-origin
    along=delta@direction
    transverse=np.cross(np.asarray(face['normal']),direction)
    transverse/=np.linalg.norm(transverse)
    signed=delta@transverse
    distance=np.linalg.norm(delta-along[:,None]*direction,axis=1)
    slack=64*np.finfo(np.float64).eps*max(1.,float(np.abs(points).max()),float(np.abs(origin).max()))
    low,high=interval
    within=(along>=low-slack)&(along<=high+slack)
    band=1.5*face['coverage']['original_resolution_mm']+.15
    material=within&(distance<=band+slack)
    positive=material&(signed>tolerance+slack)
    negative=material&(signed<-tolerance-slack)
    direct=within&(distance<=tolerance+slack)
    sign=1 if positive.any() and not negative.any() else (-1 if negative.any() and not positive.any() else 0)
    reason=('MATERIAL_ON_BOTH_SIDES' if positive.any() and negative.any() else
        'NO_TRANSVERSE_MATERIAL_SUPPORT' if sign==0 else 'DIRECT_EDGE_ROWS_MISSING' if direct.sum()<2 else 'ONE_SIDED_WITH_DIRECT_ROWS')
    selected=np.flatnonzero(direct);runs=[]
    if sign and len(selected)>=2:
        order=selected[np.argsort(along[selected],kind='stable')]
        cuts=np.flatnonzero(np.diff(along[order])>2.1*face['coverage']['original_resolution_mm'])+1
        for group in np.split(order,cuts):
            if len(group)<2 or along[group[-1]]-along[group[0]]<=tolerance:continue
            runs.append(dict(t_interval_mm=[float(along[group[0]]),float(along[group[-1]])],
                original_row_indices=np.asarray(rows[group],np.int64),
                maximum_direct_line_distance_mm=float(distance[group].max()),
                maximum_along_gap_mm=float(np.diff(along[group]).max())))
    if reason=='ONE_SIDED_WITH_DIRECT_ROWS' and not runs:reason='NO_FINITE_DIRECT_TRACE_RUN'
    return dict(face_id=face['face_id'],status=reason,local_material_rows=int(material.sum()),
        positive_material_rows=int(positive.sum()),negative_material_rows=int(negative.sum()),direct_rows=int(direct.sum()),
        material_inward_unit=(sign*transverse).tolist() if sign else None,
        material_side_sign=sign,support_band_mm=band,direct_trace_tolerance_mm=tolerance,
        float64_comparison_slack_mm=slack,direct_trace_runs=runs)


def direct_interval_evidence(xyz,first,first_rows,second,second_rows,line,interval):
    origin=np.asarray(line['line_origin_mm']);direction=np.asarray(line['line_direction'])
    sides=[face_trace(xyz,rows,face,origin,direction,interval['t_interval_mm'])
        for face,rows in ((first,first_rows),(second,second_rows))]
    boundaries=[]
    for a in sides[0]['direct_trace_runs']:
        for b in sides[1]['direct_trace_runs']:
            low=max(a['t_interval_mm'][0],b['t_interval_mm'][0]);high=min(a['t_interval_mm'][1],b['t_interval_mm'][1])
            if high-low<=.15:continue
            witnesses=[]
            for side,run in zip(sides,(a,b)):
                rows=run['original_row_indices'];along=(np.asarray(xyz[rows],np.float64)-origin)@direction
                slack=side['float64_comparison_slack_mm']
                rows=rows[(along>=low-slack)&(along<=high+slack)]
                witnesses.append(dict(face_id=side['face_id'],original_row_indices=rows,count=len(rows),
                    maximum_run_line_distance_mm=run['maximum_direct_line_distance_mm'],
                    maximum_run_gap_mm=run['maximum_along_gap_mm']))
            if min(w['count'] for w in witnesses)<2:continue
            measured_trace=max(w['maximum_run_line_distance_mm'] for w in witnesses)
            boundaries.append(dict(t_interval_mm=[low,high],line_origin_mm=origin.tolist(),line_direction=direction.tolist(),
                endpoints_mm=np.stack((origin+low*direction,origin+high*direction)).tolist(),witnesses=witnesses,
                fit_only_displacement_estimate_mm=line['estimated_fit_only_displacement_mm'],
                maximum_direct_trace_distance_mm=measured_trace,
                estimated_geometric_error_mm=line['estimated_fit_only_displacement_mm']+measured_trace,
                estimate_excludes_sensor_error_and_field_association=True,
                certified_boundary_error_bound_mm=None,field_valid=False))
    return dict(first_face=first['face_id'],second_face=second['face_id'],
        status='DIRECT_BILATERAL_ONE_SIDED_BOUNDARY' if boundaries else 'UNRESOLVED_DIRECT_BOUNDARY',
        input_t_interval_mm=interval['t_interval_mm'],sides=sides,boundaries=boundaries,
        all_unresolved_input_intervals_retained=True)


def opposed_height_hypotheses(records,groups):
    """Same-upright top/bottom pairs in a shared finite horizontal section.

    Opposing material directions distinguish an outer height from a gap. All
    feasible pairs remain; no bbox height, GT selection or cross-board average.
    """
    per_face=defaultdict(list)
    for record_index,record in enumerate(records):
        ids=(record['first_alias_group'],record['second_alias_group'])
        for side_index in (0,1):
            group=ids[side_index];face=groups[group]['representative'];other=groups[ids[1-side_index]]['representative']
            if abs(face['normal'][2])>.15 or abs(other['normal'][2])<.95:continue
            inward=record['sides'][side_index]['material_inward_unit']
            if inward is None or abs(inward[2])<.95:continue
            role='bottom' if inward[2]>0 else 'top'
            u=np.cross([0.,0.,1.],face['normal']);u/=np.linalg.norm(u)
            if u[np.argmax(np.abs(u))]<0:u=-u
            for bi,boundary in enumerate(record['boundaries']):
                direction=np.asarray(boundary['line_direction']);origin=np.asarray(boundary['line_origin_mm'])
                denom=float(direction@u)
                if abs(denom)<.95:continue
                positions=np.asarray(boundary['endpoints_mm'])@u
                slope=float(direction[2]/denom);intercept=float(origin[2]-slope*(origin@u))
                per_face[group].append(dict(role=role,section_axis=u.tolist(),section_interval_mm=[float(positions.min()),float(positions.max())],
                    z_slope=slope,z_intercept_mm=intercept,record_index=record_index,boundary_index=bi,
                    edge_error_estimate_mm=boundary['estimated_geometric_error_mm']))
    hypotheses=[]
    for group,edges in per_face.items():
        bottoms=[e for e in edges if e['role']=='bottom'];tops=[e for e in edges if e['role']=='top']
        for bottom in bottoms:
            for top in tops:
                low=max(bottom['section_interval_mm'][0],top['section_interval_mm'][0])
                high=min(bottom['section_interval_mm'][1],top['section_interval_mm'][1])
                if high-low<=.15:continue
                slope=top['z_slope']-bottom['z_slope'];intercept=top['z_intercept_mm']-bottom['z_intercept_mm']
                heights=np.array([low,high])*slope+intercept
                if heights.min()<=.15:continue
                estimated=bottom['edge_error_estimate_mm']+top['edge_error_estimate_mm']
                hypotheses.append(dict(upright_alias_group=group,upright_face=groups[group]['representative']['face_id'],
                    section_axis=bottom['section_axis'],finite_section_interval_mm=[low,high],
                    height_linear_slope=slope,height_linear_intercept_mm=intercept,
                    endpoint_heights_mm=heights.tolist(),bottom=bottom,top=top,
                    estimated_geometric_error_mm=estimated,estimated_within3mm=bool(estimated<=3.),
                    field_valid=False,field_endpoint_association_unresolved=True,
                    certified_boundary_error_bound_mm=None,hole_status='UNRESOLVED_HOLE_SUPPORT',
                    derived_from_same_upright_opposed_finite_boundaries=True))
    return hypotheses
