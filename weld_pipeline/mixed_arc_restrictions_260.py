"""Source-bound restrictions of a single explicit world-circle authority.

The circle prototype occupies an existing geometry-span role. Its actual body
is the selected parameter interval; endpoints retain symbolic angle identity.
"""
from fractions import Fraction as F
import math

import numpy as np

from .concrete_curve_certificate_250 import (
    ConcreteCurve, UnprovenCurve, _fraction_bounds, _sqrt_fraction_bounds,
    _sin_cos_bounds, pi_bounds,
)
from .raw_triangle_world_charts_258 import exact, dot, certify_exact_affine_circle_image


def _f(value):
    return exact(float(value))


def matching_points(first, second):
    a, b = first.get('point_exact'), second.get('point_exact')
    return (a is not None and b is not None and a == b) or first['world_expression'] == second['world_expression']


class RestrictedSourceArc(ConcreteCurve):
    def __init__(self, prototype, parameter_interval, *, context=None):
        if not isinstance(prototype, ConcreteCurve):
            prototype = ConcreteCurve(prototype)
        if prototype.kind != 'arc':
            raise ValueError('Analytic circle prototype required')
        first, last = parameter_interval
        if not 0 <= first < last <= 1:
            raise ValueError('Positive restriction inside the prototype range required')
        value = dict(prototype.record)
        center, u, v = [tuple(map(_f, value[key])) for key in ('center_mm', 'u_axis', 'v_axis')]
        if context is not None:
            basis = [tuple(map(_f, row)) for row in context['basis']]
            origin = tuple(map(_f, context['origin']))
            center = tuple(origin[j] + sum(center[i]*basis[i][j] for i in range(2)) for j in range(3))
            u, v = [tuple(sum(vector[i]*basis[i][j] for i in range(2)) for j in range(3)) for vector in (u, v)]
            value.update(center_mm=list(map(float, center)),u_axis=list(map(float,u)),v_axis=list(map(float,v)),dimensions=3)
        object.__setattr__(self, 'record', value)
        object.__setattr__(self, 'prototype', prototype)
        object.__setattr__(self, 'parameter_interval', (first, last))
        object.__setattr__(self, 'coefficients', (center, _f(value['radius_mm']), u, v))
        prototype_record=dict(value)
        if prototype.record['is_closed']:
            cycles=(last-first)*prototype.record['turn_count']
            value['is_closed']=cycles.denominator==1
            value['turn_count']=cycles.numerator//cycles.denominator
            value['sweep_rad']=(1 if prototype.record['sweep_rad']>0 else -1)*float(cycles)*math.tau
        else:
            slo,shi=self.sweep_bounds();plo,phi=pi_bounds()
            low,high=(last-first)*slo/(2*phi),(last-first)*shi/(2*plo)
            lower_floor=low.numerator//low.denominator;upper_floor=high.numerator//high.denominator
            if lower_floor!=upper_floor:
                raise UnprovenCurve('Restricted open arc complete-turn count is numerically unproven')
            value.update(is_closed=False,turn_count=lower_floor,sweep_rad=float(last-first)*prototype.record['sweep_rad'])
        value.update(prototype_arc=prototype_record,source_parameter_interval=(first,last),
                     source_exact_coefficients=self.coefficients,
                     evaluation_uses_exact_source_restriction=True)
        if context is not None:
            value.update(orientation_frame_id=context['domain'].domain_id,orientation_frame_basis=context['basis'].tolist())

    def sweep_bounds(self):
        value = self.prototype.record
        if value['is_closed']:
            lo, hi = pi_bounds()
            return 2*value['turn_count']*lo, 2*value['turn_count']*hi
        sweep = _f(abs(value['sweep_rad']))
        return sweep, sweep

    def phase(self, parameter):
        value = self.prototype.record
        sign = 1 if value['sweep_rad'] > 0 else -1
        if value['is_closed']:
            return 'TAU', (sign*value['turn_count']*parameter) % 1
        return 'RAD', _f(value['sweep_rad'])*parameter

    def point_bounds(self, parameter):
        phase_kind, phase = self.phase(parameter)
        if phase_kind == 'TAU' and (4*phase).denominator == 1:
            index = int(4*phase) % 4
            sine, cosine = ((0, 1, 0, -1)[index], (1, 0, -1, 0)[index])
            sine, cosine = (F(sine),F(sine)),(F(cosine),F(cosine))
        elif phase == 0:
            sine, cosine = (F(0),F(0)),(F(1),F(1))
        else:
            if phase_kind == 'TAU':
                lo, hi = pi_bounds(); angle = 2*phase*lo, 2*phase*hi
            else:
                angle = phase, phase
            sine, cosine = [tuple(map(_f, bounds)) for bounds in _sin_cos_bounds(angle)]
        center, radius, u, v = self.coefficients
        lower, upper = [], []
        for c, x, y in zip(center, u, v):
            xs, ys = sorted((x*cosine[0],x*cosine[1])),sorted((y*sine[0],y*sine[1]))
            lower.append(c+radius*(xs[0]+ys[0]));upper.append(c+radius*(xs[1]+ys[1]))
        return tuple(lower),tuple(upper)

    def endpoint(self, index):
        parameter = self.parameter_interval[index]
        lower, upper = self.point_bounds(parameter)
        point_exact = lower if lower == upper else None
        expression = ('POINT', point_exact) if point_exact is not None else ('WORLD_ARC_POINT',self.coefficients,self.phase(parameter))
        exported = [float((lo+hi)/2) for lo,hi in zip(lower,upper)]
        error = sum(max(abs(_f(x)-lo),abs(_f(x)-hi))**2 for x,lo,hi in zip(exported,lower,upper))
        return dict(point_exact=point_exact,point_mm=exported,world_expression=expression,
                    world_bounds=(lower,upper),world_export_upper_mm=_sqrt_fraction_bounds(error)[1],
                    authority_parameter=parameter)

    def circle_proof(self):
        center,radius,u,v=self.coefficients
        proof=certify_exact_affine_circle_image(u,v)
        return dict(status='PASS_EXACT_SOURCE_WORLD_CIRCLE_AUTHORITY',basis=proof,
                    world_radius_squared_mm=radius*radius*dot(u,u),
                    exact_authority_coefficients=self.coefficients,
                    numerical_axis_export_is_not_the_circle_definition=True)

    def signature(self):
        return self.coefficients, self.prototype.record['is_closed'], self.prototype.record['turn_count'], _f(self.prototype.record['sweep_rad'])

    def metric(self):
        _,radius,u,v=self.coefficients
        # Circle metric uses exact equal Gram values; an ellipse cannot use this.
        self.circle_proof()
        lo,hi=_sqrt_fraction_bounds(dot(u,u));slo,shi=self.sweep_bounds()
        delta=self.parameter_interval[1]-self.parameter_interval[0]
        return dict(physical_lower_mm=radius*_f(lo)*slo*delta,
                    physical_upper_mm=radius*_f(hi)*shi*delta,
                    analytic_arc=True,parameter_interval=self.parameter_interval,
                    complete_continuity_verified=True,positive_export_verified=False)

    def samples(self, step_mm=1., chord_mm=.1, point_budget=200001):
        if not (0<float(step_mm)<=1. and 0<float(chord_mm)<=.1 and type(point_budget) is int and point_budget>=2):
            raise ValueError('Frozen physical step/chord and finite evaluation budget required')
        _,radius,u,v=self.coefficients
        uu,vv,uv=dot(u,u),dot(v,v),dot(u,v)
        sigma=_f(_sqrt_fraction_bounds(max(uu+abs(uv),vv+abs(uv)))[1])
        _,sweep=self.sweep_bounds()
        delta=self.parameter_interval[1]-self.parameter_interval[0]
        angle=sweep*delta
        length=radius*sigma*angle
        pi_lower,_=pi_bounds()
        count=max(1,math.ceil(_fraction_bounds(length)[1]/step_mm),math.ceil(angle/(pi_lower/2)))
        while True:
            step=_fraction_bounds(length/count)[1]
            chord=_fraction_bounds(radius*sigma*(angle/count)**2/8)[1]
            if step<=step_mm and chord<=chord_mm:
                break
            count=max(count+1,math.ceil(count*max(step/step_mm,math.sqrt(chord/chord_mm))))
        if count+1>point_budget:
            raise UnprovenCurve('Complete arc restriction exceeds evaluation budget; no truncation')
        points,lowers,uppers,errors=[],[],[],[]
        start,last=self.parameter_interval
        for index in range(count+1):
            parameter=start+F(index,count)*(last-start)
            lower,upper=self.point_bounds(parameter)
            exported=[float((a+b)/2) for a,b in zip(lower,upper)]
            if not all(map(math.isfinite,exported)):
                raise UnprovenCurve('Analytic source endpoint is not numerically exportable')
            squared=sum(max(abs(_f(x)-a),abs(_f(x)-b))**2 for x,a,b in zip(exported,lower,upper))
            errors.append(_sqrt_fraction_bounds(squared)[1])
            if points and points[-1]==exported:
                raise UnprovenCurve('Positive analytic interval collapses between adjacent exported evaluation points')
            points.append(exported)
            lowers.append([_fraction_bounds(x)[0] for x in lower])
            uppers.append([_fraction_bounds(x)[1] for x in upper])
        return dict(points=np.asarray(points),point_enclosure_lower=np.asarray(lowers),point_enclosure_upper=np.asarray(uppers),
                    point_error_upper_mm=np.asarray(errors),maximum_step_upper_mm=step,
                    analytic_chord_error_upper_mm=chord,mandatory_vertex_sample_indices=[0,count],
                    original_zero_edges_retained=0,original_kind='arc',positive_length_cutoff_applied=False,
                    point_count=count+1,source_parameter_interval=self.parameter_interval,
                    complete_intrinsic_restriction_verified=True)


def certify_arc_chain(arcs, *, authority_ref, maximum_turns, is_closed=None, turn_count=None, direction=None):
    if not arcs:
        raise ValueError('Complete ordered arc restriction chain required')
    authority=arcs[0]
    signature=authority.signature()
    first=authority.parameter_interval[0]
    previous=first
    for arc in arcs:
        arc.circle_proof()
        if arc.signature()!=signature or arc.parameter_interval[0]!=previous:
            raise UnprovenCurve('Whole arc chain changes circle authority, skips phase or backtracks')
        previous=arc.parameter_interval[1]
    prototype=authority.prototype.record
    sign=1 if prototype['sweep_rad']>0 else -1
    if direction is not None and direction!=sign:
        raise UnprovenCurve('Whole arc traversal contradicts declared authoritative direction')
    delta=previous-first
    if prototype['is_closed']:
        cycles=delta*prototype['turn_count']
        complete=cycles.numerator//cycles.denominator
        closed=cycles.denominator==1
        if cycles>maximum_turns:
            raise UnprovenCurve('Whole parent arc traversal exceeds frozen global source capacity')
    else:
        slo,shi=authority.sweep_bounds();plo,phi=pi_bounds()
        low,high=delta*slo/(2*phi),delta*shi/(2*plo)
        lower_floor=low.numerator//low.denominator
        upper_floor=high.numerator//high.denominator
        if lower_floor!=upper_floor:
            raise UnprovenCurve('Open authoritative arc complete-turn count cannot be certified')
        complete=lower_floor
        closed=False  # A nonzero rational angle cannot equal a nonzero integer times true2pi.
        if high>maximum_turns:
            raise UnprovenCurve('Whole parent arc traversal exceeds frozen global source capacity')
    if is_closed is not None and is_closed!=closed:
        raise UnprovenCurve('Arc closure disagrees with complete authoritative phase range')
    if turn_count is not None and turn_count!=complete:
        raise UnprovenCurve('Whole arc complete-turn count disagrees with declared parent')
    return dict(status='PASS_COMPLETE_SAME_WORLD_CIRCLE_ORDERED_ARC_RESTRICTIONS_AND_PHASE',
                authority_ref=authority_ref,restriction_count=len(arcs),parameter_range=(first,previous),
                direction=sign,is_closed=closed,complete_turn_count=complete,
                maximum_complete_turns=maximum_turns,world_circle=authority.circle_proof(),
                source_domain_support_proof_is_separate=True,GT_geometry_verified=False)
