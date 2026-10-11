"""A source-only surface span and its two frozen local boundary slots.

All original float UV triangles are retained at their exact rank.
Exact original XYZ projection residuals remain explicit observation uncertainty.
This proves one selected span, not complete parent/child instance assignment.
"""
from fractions import Fraction as F
from hashlib import sha256
import json
import math

import numpy as np

from .decoder_output_slots_247 import validate_slot_capacity,local_region_triangles
from .concrete_curve_certificate_250 import (
    ConcreteCurve,UnprovenCurve,_fraction_bounds,_sqrt_fraction_bounds,_sin_cos_bounds,
)
from .mixed_curve_domain_253 import TriangleDomain,certify_curve_domain

VERSION='FROZEN_SOURCE_MIXED_DIMENSION_SURFACE_SLOT_WORLD_BINDING_253_v1'


def _f(x):return F.from_float(float(x))
def _digest(array):
    value=np.ascontiguousarray(array)
    return sha256(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


class SurfaceCompiler:
    def __init__(self,connection):
        self.db=connection
        if connection.execute('PRAGMA query_only').fetchone()!=(1,):
            raise ValueError('Read-only input manifest required')
        self.global_anchor_count=connection.execute('SELECT count(*) FROM anchors').fetchone()[0]

    def face(self,face_id):
        if type(face_id) is not int or face_id<0:raise ValueError('Original integer face identity required')
        row=self.db.execute('SELECT descriptor_json FROM domains WHERE face_id=?',(face_id,)).fetchone()
        if row is None:raise ValueError('Face not in frozen source manifest')
        face=json.loads(row[0]);basis=np.asarray(face['basis_uv'],float);origin=np.asarray(face['origin_mm'],float)
        normal=np.asarray(face['normal'],float)
        if (basis.shape!=(2,3) or origin.shape!=(3,) or normal.shape!=(3,)
                or not np.isfinite(basis).all() or not np.isfinite(origin).all() or not np.isfinite(normal).all()
                or np.max(np.abs(basis@basis.T-np.eye(2)))>1e-9
                or np.max(np.abs(basis@normal))>1e-9 or abs(origin@normal-face['offset_mm'])>1e-7):
            raise ValueError('Invalid frozen computational face frame')
        triangles=self.db.execute('''SELECT triangle_ordinal,raw_a,raw_b,raw_c,centroid_anchor_id
            FROM triangles WHERE face_id=? ORDER BY triangle_ordinal''',(face_id,)).fetchall()
        corners=self.db.execute('''SELECT r.triangle_ordinal,r.source_role,a.raw_a,a.seed_x,a.seed_y,a.seed_z
            FROM regions r JOIN anchors a ON a.anchor_id=r.anchor_id
            WHERE r.face_id=? AND r.source_role<3 AND a.kind='RAW_REPRESENTATIVE'
            ORDER BY r.triangle_ordinal,r.source_role''',(face_id,)).fetchall()
        if len(triangles)!=face['triangle_count'] or len(corners)!=3*len(triangles):
            raise ValueError('Complete original triangle/corner rows required')
        raw_tri=np.asarray([row[1:4] for row in triangles],dtype=np.int64).reshape(-1,3)
        raw_corners=np.asarray([row[2] for row in corners],dtype=np.int64)
        if (not np.array_equal(raw_tri.reshape(-1),raw_corners)
                or [row[0] for row in triangles]!=list(range(len(triangles)))
                or [(row[0],row[1]) for row in corners]!=[(i,j) for i in range(len(triangles)) for j in range(3)]):
            raise ValueError('Original source triangle/corner order changed')
        unique,first=np.unique(raw_corners,return_index=True)
        seeds=np.asarray([row[3:] for row in corners],float).reshape(-1,3)[first]
        uv=(seeds-origin)@basis.T
        compiled_tri=np.searchsorted(unique,raw_tri)
        domain=TriangleDomain(uv,compiled_tri,'source_face:'+str(face_id))
        exact_basis=tuple(tuple(map(_f,row)) for row in basis)
        exact_origin=tuple(map(_f,origin))
        exact_projected=[];maximum_square=F(0)
        for xyz,stored in zip(seeds,uv):
            original=tuple(map(_f,xyz))
            exact=tuple(sum((original[k]-exact_origin[k])*exact_basis[j][k] for k in range(3)) for j in range(2))
            exact_projected.append(exact)
            maximum_square=max(maximum_square,sum((exact[j]-_f(stored[j]))**2 for j in range(2)))
        projection_witnesses=[]
        for ordinal,rank in enumerate(domain.support_rank):
            if rank==2:continue
            a,b,c=(exact_projected[int(i)] for i in compiled_tri[ordinal])
            determinant=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            projection_witnesses.append(dict(original_triangle_ordinal=ordinal,stored_float_uv_rank=rank,
                original_exact_projection_determinant=dict(numerator=str(determinant.numerator),denominator=str(determinant.denominator))))
        projection_rounding=dict(all_original_vertex_projections_enclosed=True,
            maximum_uv_rounding_upper_mm=_sqrt_fraction_bounds(maximum_square)[1],
            collapsed_original_float_uv_triangle_witnesses=projection_witnesses,
            original_float_uv_domain_expanded_or_vertices_changed=False,
            physical_material_identity_or_filled_solid_verified=False)
        hashes={name:_digest(array) for name,array in (('basis',basis),('origin',origin),('uv',uv),('triangles',compiled_tri))}
        for array in (basis,origin,uv,compiled_tri,unique,raw_tri):array.flags.writeable=False
        return dict(face_id=face_id,descriptor=face,basis=basis,origin=origin,uv=uv,triangles=compiled_tri,
                    original_raw_ids=unique,original_raw_triangle_rows=raw_tri,
                    centroid_anchor_ids=[row[4] for row in triangles],domain=domain,
                    all_original_accepted_triangle_and_raw_corner_rows_retained=True,compiled_array_sha256=hashes,
                    source_projection_rounding=projection_rounding)


def _endpoint_intervals(curve):
    value=curve.record
    if curve.kind!='arc':
        vertices=curve.vertices()
        return [(tuple(map(_f,p)),tuple(map(_f,p))) for p in (vertices[0],vertices[-1])]
    center,u,v=map(lambda x:tuple(map(_f,x)),(value['center_mm'],value['u_axis'],value['v_axis']))
    radius=_f(value['radius_mm'])
    start=tuple(c+radius*x for c,x in zip(center,u))
    if value['is_closed']:return [(start,start),(start,start)]
    sine,cosine=_sin_cos_bounds((_f(value['sweep_rad']),_f(value['sweep_rad'])))
    intervals=[]
    for c,x,y in zip(center,u,v):
        xs=sorted((_f(cosine[0])*x,_f(cosine[1])*x));ys=sorted((_f(sine[0])*y,_f(sine[1])*y))
        intervals.append((c+radius*(xs[0]+ys[0]),c+radius*(xs[1]+ys[1])))
    return [(start,start),(tuple(x[0] for x in intervals),tuple(x[1] for x in intervals))]


def _local_endpoint_proven(context,ordinals,point_interval):
    domain=context['domain'];lo,hi=point_interval
    # A closed original triangle is convex even at rank 0/1. All four
    # interval-box corners in ONE original support prove its whole box.
    corners=((lo[0],lo[1]),(lo[0],hi[1]),(hi[0],lo[1]),(hi[0],hi[1]))
    return any(all(domain.original_triangle_member(int(ordinal),point) for point in corners)
               for ordinal in ordinals)


def _lift_exact(value,basis,origin=None):
    values=list(map(_f,value));o=[F(0)]*3 if origin is None else list(map(_f,origin))
    exact=[o[j]+sum(values[i]*_f(basis[i,j]) for i in range(2)) for j in range(3)]
    exported=[float(x) for x in exact]
    if not all(map(math.isfinite,exported)):raise UnprovenCurve('Nonfinite world-coordinate export')
    error=sum((_f(x)-y)**2 for x,y in zip(exported,exact))
    return exported,_sqrt_fraction_bounds(error)[1]


def bind_surface_span(compiler,context,local_curve,assignments):
    if not isinstance(local_curve,ConcreteCurve):local_curve=ConcreteCurve(local_curve)
    if local_curve.record['dimensions']!=2:raise ValueError('Actual local UV curve required')
    frozen=compiler.db.execute('SELECT descriptor_json FROM domains WHERE face_id=?',(context['face_id'],)).fetchone()
    if (frozen is None or json.loads(frozen[0])!=context['descriptor']
            or any(_digest(context[key])!=value for key,value in context['compiled_array_sha256'].items())
            or not np.array_equal(context['basis'],context['descriptor']['basis_uv'])
            or not np.array_equal(context['origin'],context['descriptor']['origin_mm'])):
        raise ValueError('Compiled original source frame was changed')
    if set(assignments)!= {'geometry_span','start_boundary','end_boundary'}:
        raise ValueError('One explicit span and both logical boundary definitions required')
    for key,row in assignments.items():
        if row['role']!=('geometry_span' if key=='geometry_span' else 'boundary'):
            raise ValueError('Frozen role disagrees with output field')
    capacity=validate_slot_capacity(compiler.db,list(assignments.values()))
    endpoints=_endpoint_intervals(local_curve)
    result=dict(version=VERSION,scope='ONE_SOURCE_SURFACE_SPAN_AND_TWO_BOUNDARIES_ONLY',
        face_id=context['face_id'],assignments=assignments,selected_slot_capacity=capacity,
        source_projection_rounding=context['source_projection_rounding'],
        complete_internal_geometry_node_slot_definitions_verified=False,
        complete_parent_child_order_or_global_assignment_verified=False,
        actual_all111_or_all1605_joint_expression_run=False,source_only_query_not_weld_prediction=True,
        certified_filled_material=False,computational_frame_is_reviewed_material_normal=False,
        model_inference_calls=0,actual_weld_optimizer_updates=0,full_contract_pass=False)
    if local_curve.record['geometry_uncertainty_mm']!=0:
        return {**result,'status':'UNPROVEN_LOCAL_CURVE_UNCERTAINTY_AND_SLOT_PERMISSIONS'}
    if local_curve.kind=='arc' and local_curve.record['turn_count']>compiler.global_anchor_count:
        return {**result,'status':'INVALID_UNWRAPPED_WINDING_EXCEEDS_FROZEN_GLOBAL_SPAN_CAPACITY'}
    for key,point in zip(('start_boundary','end_boundary'),endpoints):
        regions=local_region_triangles(compiler.db,assignments[key]['anchor_id'],context['face_id'])
        if not regions or not _local_endpoint_proven(context,regions,point):
            return {**result,'status':'UNPROVEN_SELECTED_BOUNDARY_LOCAL_PERMISSION','boundary_field':key}
    start_regions=local_region_triangles(compiler.db,assignments['geometry_span']['anchor_id'],context['face_id'])
    if not start_regions or not _local_endpoint_proven(context,start_regions,endpoints[0]):
        return {**result,'status':'UNPROVEN_SELECTED_SPAN_ANCHOR_LOCAL_START_PERMISSION'}
    own=certify_curve_domain(local_curve,context['uv'],context['triangles'],domain_id=context['domain'].domain_id,compiled_domain=context['domain'])
    result['own_domain_certificate']=own
    if not own['status'].startswith('PASS'):return {**result,'status':'UNPROVEN_OR_OUTSIDE_SELECTED_OWN_SURFACE_DOMAIN'}
    basis,origin=context['basis'],context['origin'];record=dict(local_curve.record)
    uu=sum(_f(x)**2 for x in basis[0]);vv=sum(_f(x)**2 for x in basis[1]);uv=sum(_f(x)*_f(y) for x,y in zip(*basis))
    sigma=_sqrt_fraction_bounds(max(uu+abs(uv),vv+abs(uv)))[1]
    if local_curve.kind!='arc':
        vertices=[_lift_exact(p,basis,origin) for p in record['points']]
        record['points']=[p for p,_ in vertices];quantization=max(error for _,error in vertices)
    else:
        center,ce=_lift_exact(record['center_mm'],basis,origin)
        u,ue=_lift_exact(record['u_axis'],basis);v,ve=_lift_exact(record['v_axis'],basis)
        record.update(center_mm=center,u_axis=u,v_axis=v,
                      orientation_frame_id=context['domain'].domain_id,orientation_frame_basis=basis.tolist())
        quantization=_fraction_bounds(_f(ce)+_f(record['radius_mm'])*(_f(ue)+_f(ve)))[1]
    record['geometry_uncertainty_mm']=_fraction_bounds(_f(quantization)+_f(sigma)*_f(local_curve.record['geometry_uncertainty_mm']))[1]
    try:world=ConcreteCurve(record)
    except ValueError as error:return {**result,'status':'UNPROVEN_FLOAT_WORLD_CURVE_EXPORT','reason':str(error)}
    return {**result,'status':'PASS_BOUND_SOURCE_SURFACE_CURVE_AND_LOCAL_BOUNDARY_SLOTS',
        'local_curve':local_curve.record,'world_curve':world.record,
        'world_curve_numeric_quantization_upper_mm':quantization,
        'world_curve_total_geometry_uncertainty_upper_mm':record['geometry_uncertainty_mm'],
        'world_frame_binding_verified_for_declared_uv_patch':True,
        'local_boundary_permissions_verified':True,'source_face_basis':basis.tolist(),'source_face_origin_mm':origin.tolist()}
