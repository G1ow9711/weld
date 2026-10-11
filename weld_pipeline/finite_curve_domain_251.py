"""Conditional UV coverage of ONE explicitly allowed finite triangle domain.

Exact rational clipping covers line spans including positive gaps and holes.
Arc neighborhoods use a conservative boundary superset, never a convex hull.
World-coordinate binding and legal runtime slot membership are separate gates.
"""
from fractions import Fraction as F
import math

import numpy as np

from .concrete_curve_certificate_250 import (
    ConcreteCurve,UnprovenCurve,point_polyline_distance_bounds,POLICY,_up,
)

VERSION='CONDITIONAL_CONCRETE_CURVE_OWN_FINITE_UV_DOMAIN_251_v1'


def _f(x):return F.from_float(float(x))
def _cross(a,b):return a[0]*b[1]-a[1]*b[0]
def _minus(a,b):return tuple(x-y for x,y in zip(a,b))
def _record(value):return dict(numerator=str(value.numerator),denominator=str(value.denominator))


class TriangleDomain:
    def __init__(self,uv_mm,triangles,domain_id):
        self.uv=np.asarray(uv_mm,float).copy()
        self.triangles=np.asarray(triangles).copy()
        if (not isinstance(domain_id,str) or not domain_id or self.uv.ndim!=2 or self.uv.shape[1]!=2
                or not np.isfinite(self.uv).all() or self.triangles.ndim!=2 or self.triangles.shape[1]!=3
                or not np.issubdtype(self.triangles.dtype,np.integer)
                or (self.triangles<0).any() or (self.triangles>=len(self.uv)).any()):
            raise ValueError('Original finite UV rows, integer triangles and an explicit domain identity required')
        self.domain_id=domain_id
        self.points=[tuple(map(_f,point)) for point in self.uv]
        self.orientation=[]
        edge_sides={}
        for ordinal,indices in enumerate(self.triangles):
            a,b,c=(self.points[int(index)] for index in indices)
            area=_cross(_minus(b,a),_minus(c,a))
            if area==0:
                raise UnprovenCurve('Degenerate original triangle '+str(ordinal)+' remains unresolved, not deleted')
            self.orientation.append(1 if area>0 else -1)
            for p,q,third in ((indices[0],indices[1],indices[2]),(indices[1],indices[2],indices[0]),(indices[2],indices[0],indices[1])):
                p,q=sorted((int(p),int(q)))
                side=_cross(_minus(self.points[q],self.points[p]),_minus(self.points[int(third)],self.points[p]))
                edge_sides.setdefault((p,q),[]).append((ordinal,1 if side>0 else -1))
        self.boundary_superset=[];self.removed_two_sided_internal_edges=[]
        for edge,records in sorted(edge_sides.items()):
            if {side for _,side in records}=={-1,1}:
                self.removed_two_sided_internal_edges.append(dict(edge=list(edge),original_triangle_ordinals=[i for i,_ in records]))
            else:
                self.boundary_superset.append(edge)
        if len(self.triangles) and not self.boundary_superset:
            raise UnprovenCurve('No finite boundary superset could be verified')
        vertices=self.uv[self.triangles]
        self.box_lo=vertices.min(axis=1) if len(vertices) else np.empty((0,2))
        self.box_hi=vertices.max(axis=1) if len(vertices) else np.empty((0,2))
        self.uv.flags.writeable=self.triangles.flags.writeable=False

    def point_member(self,point):
        value=tuple(map(_f,point))
        if len(value)!=2:return False,None
        candidates=np.flatnonzero(((self.box_lo<=point)&(point<=self.box_hi)).all(axis=1))
        for ordinal in candidates:
            indices=self.triangles[ordinal]
            vertices=[self.points[int(index)] for index in indices]
            sign=self.orientation[ordinal]
            if all(sign*_cross(_minus(b,a),_minus(value,a))>=0 for a,b in zip(vertices,vertices[1:]+vertices[:1])):
                return True,int(ordinal)
        return False,None

    def exact_span(self,start,end):
        a,b=tuple(map(_f,start)),tuple(map(_f,end))
        if a==b:
            inside,ordinal=self.point_member(start)
            return dict(covered=inside,zero_original_span_retained=True,
                        intersections=[] if ordinal is None else [dict(original_triangle_ordinal=ordinal,
                            lo=_record(F(0)),hi=_record(F(1)))],union_intervals=[] if not inside else [[_record(F(0)),_record(F(1))]])
        delta=_minus(b,a)
        segment_lo,segment_hi=np.minimum(start,end),np.maximum(start,end)
        candidates=np.flatnonzero(((self.box_lo<=segment_hi)&(segment_lo<=self.box_hi)).all(axis=1))
        intervals=[]
        for ordinal in candidates:
            vertices=[self.points[int(index)] for index in self.triangles[ordinal]]
            sign=self.orientation[ordinal];lower,upper=F(0),F(1)
            allowed=True
            for p,q in zip(vertices,vertices[1:]+vertices[:1]):
                edge=_minus(q,p)
                value=sign*_cross(edge,_minus(a,p))
                slope=sign*_cross(edge,delta)
                if slope>0:lower=max(lower,-value/slope)
                elif slope<0:upper=min(upper,-value/slope)
                elif value<0:allowed=False;break
            if allowed and lower<=upper:
                intervals.append((lower,upper,int(ordinal)))
        union=[]
        for lower,upper,_ in sorted(intervals):
            if union and lower<=union[-1][1]:
                union[-1][1]=max(union[-1][1],upper)
            else:union.append([lower,upper])
        covered=bool(union) and union[0][0]==0 and union[-1][1]==1 and len(union)==1
        return dict(covered=covered,zero_original_span_retained=False,
            intersections=[dict(original_triangle_ordinal=i,lo=_record(lo),hi=_record(hi)) for lo,hi,i in intervals],
            union_intervals=[[_record(lo),_record(hi)] for lo,hi in union],positive_gap_or_hole_merged=False)

    def boundary_distance_bounds(self,points):
        if not self.boundary_superset:
            return np.full(len(points),np.inf),np.full(len(points),np.inf)
        edges=self.uv[np.asarray(self.boundary_superset,dtype=np.int64)]
        # One independent 2-point polyline per edge; never connect separate domain boundaries.
        lower,upper=np.full(len(points),np.inf),np.full(len(points),np.inf)
        for edge in edges:
            lo,hi=point_polyline_distance_bounds(points,edge)
            lower=np.minimum(lower,lo);upper=np.minimum(upper,hi)
        return lower,upper


def certify_curve_domain(curve,uv_mm,triangles,*,domain_id,policy=None):
    if not isinstance(curve,ConcreteCurve):curve=ConcreteCurve(curve)
    if curve.record['dimensions']!=2:
        raise ValueError('This certificate requires an explicitly declared UV curve; world binding is separate')
    policy={**POLICY,**(policy or {})}
    if (not math.isfinite(policy['maximum_step_mm']) or not 0<policy['maximum_step_mm']<=1.
            or not math.isfinite(policy['maximum_chord_mm']) or not 0<policy['maximum_chord_mm']<=.1
            or type(policy['max_refinements']) is not int or policy['max_refinements']<0):
        raise ValueError('Frozen contract sampling and nonnegative refinement budget required')
    result=dict(version=VERSION,scope='CONDITIONAL_ONE_DECLARED_ORIGINAL_UV_TRIANGLE_DOMAIN',domain_id=domain_id,
        global_surface_union_substituted_for_allowed_domain=False,world_curve_to_original_domain_binding_verified=False,
        actual_runtime_slot_permission_and_capacity_verified=False,global_parent_order_and_identity_verified=False,
        certified_filled_material=False,material_or_weld_identity_verified=False,convex_hull_or_hole_fill_used=False,
        model_inference_calls=0,actual_weld_optimizer_updates=0,
        actual_all111_or_all1605_joint_expression_run=False,full_contract_pass=False)
    try:
        domain=TriangleDomain(uv_mm,triangles,domain_id)
    except UnprovenCurve as error:
        return {**result,'status':'UNPROVEN_ORIGINAL_TRIANGLE_DOMAIN','reason':str(error)}
    result.update(original_triangle_count=len(domain.triangles),original_uv_row_count=len(domain.uv),
                  boundary_superset_edges=len(domain.boundary_superset),
                  two_sided_internal_edges_removed=len(domain.removed_two_sided_internal_edges))
    if not len(domain.triangles):
        return {**result,'status':'PROVEN_CONCRETE_CURVE_OUTSIDE_EMPTY_DECLARED_DOMAIN'}
    uncertainty=curve.record['geometry_uncertainty_mm']
    if curve.kind!='arc' and uncertainty==0:
        vertices=curve.vertices()
        spans=[domain.exact_span(a,b) for a,b in zip(vertices[:-1],vertices[1:])]
        return {**result,'status':'PASS_COMPLETE_EXACT_UV_CURVE_DOMAIN_COVERAGE' if all(x['covered'] for x in spans)
                else 'PROVEN_CONCRETE_CURVE_OUTSIDE_DECLARED_TRIANGLE_DOMAIN',
                'original_geometry_span_count':len(spans),'zero_original_spans_retained':sum(x['zero_original_span_retained'] for x in spans),
                'per_original_span':spans,'proof':'EXACT_RATIONAL_HALFPLANES_AND_CLOSED_INTERVAL_UNION_NO_GAP_MERGE'}
    rounds=[]
    for refinement in range(policy['max_refinements']+1):
        try:
            samples=curve.samples(policy['maximum_step_mm']/2**refinement,
                                  policy['maximum_chord_mm']/4**refinement,policy['point_budget_per_curve'])
            points=samples['points'];errors=_up(samples['point_error_upper_mm']+uncertainty)
            total_chord=_up(samples['analytic_chord_error_upper_mm']+errors.max())
            if total_chord>.1:
                raise UnprovenCurve('Actual interpolation chord plus numeric enclosure not certified at 0.1mm')
            members=[domain.point_member(point) for point in points]
            lower,upper=domain.boundary_distance_bounds(points)
            radius=_up(errors+_up(samples['maximum_step_upper_mm']/2))
            membership=np.array([row[0] for row in members])
            proven_outside=(~membership)&(lower>errors)
            proven_inside=membership&(lower>radius)
            row=dict(refinement=refinement,point_count=len(points),sample_members_inside=int(membership.sum()),
                neighborhoods_proven_inside=int(proven_inside.sum()),maximum_step_upper_mm=samples['maximum_step_upper_mm'],
                total_interpolation_chord_error_upper_mm=float(total_chord),
                minimum_boundary_distance_lower_mm=float(lower.min()),
                boundary_edges_form_superset_of_true_union_boundary=True)
            rounds.append(row)
            if proven_outside.any():
                index=int(np.flatnonzero(proven_outside)[0])
                return {**result,'status':'PROVEN_CONCRETE_CURVE_POINT_OUTSIDE_DECLARED_DOMAIN','rounds':rounds,
                    'outside_witness':dict(sample_index=index,point_mm=points[index].tolist(),point_error_upper_mm=float(errors[index]),
                        boundary_distance_lower_mm=float(lower[index]),exact_sample_center_not_in_any_original_triangle=True)}
            if proven_inside.all():
                return {**result,'status':'PASS_COMPLETE_CONSERVATIVE_CURVE_UV_DOMAIN_COVERAGE','rounds':rounds,
                    'proof':'EXACT_SAMPLE_MEMBERSHIP_BOUNDARY_SUPERSET_DISTANCE_AND_CONTINUOUS_HALF_STEP_NEIGHBORHOODS'}
        except UnprovenCurve as error:
            return {**result,'status':'UNPROVEN_CURVE_DOMAIN_RESOURCE_OR_NUMERIC_LIMIT','reason':str(error),'rounds':rounds}
    return {**result,'status':'UNPROVEN_NEAR_DECLARED_DOMAIN_BOUNDARY','rounds':rounds,
            'unproven_is_not_deleted_or_scored_as_success':True}
