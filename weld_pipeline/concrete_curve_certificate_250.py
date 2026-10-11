"""Conditional concrete-curve certificates, separate from instance selection.

Every numeric interval is rounded outward. Circular samples use rational
Taylor bounds and a rational Machin bound for pi, not an assumed libm error.
This module never creates runtime identities or supplies material attributes.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import math

import numpy as np

VERSION = 'CONDITIONAL_CONCRETE_LINE_ARC_POLYLINE_CERTIFICATE_250_v1'
POLICY = dict(maximum_step_mm=1., maximum_chord_mm=.1, max_refinements=8,
              point_budget_per_curve=200001, query_block=32, segment_block=128)


class UnprovenCurve(ValueError):
    """A resource/numeric limit does not prove geometric infeasibility."""


def _f(x):
    return F.from_float(float(x))


def _down(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


def _exact(x):
    x = np.asarray(x, dtype=float)
    return x, x


def _add(a, b):
    return _down(a[0]+b[0]), _up(a[1]+b[1])


def _sub(a, b):
    return _down(a[0]-b[1]), _up(a[1]-b[0])


def _mul(a, b):
    products = np.stack(np.broadcast_arrays(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]))
    return _down(products.min(axis=0)), _up(products.max(axis=0))


def _square(a):
    left, right = a[0]*a[0], a[1]*a[1]
    lower = np.where((a[0] <= 0) & (a[1] >= 0), 0., np.minimum(_down(left), _down(right)))
    return np.maximum(0., lower), _up(np.maximum(left, right))


def _sum(a, axis=-1):
    lo, hi = np.moveaxis(a[0],axis,-1), np.moveaxis(a[1],axis,-1)
    result = _exact(np.zeros(lo.shape[:-1]))
    for index in range(lo.shape[-1]):
        result = _add(result,(lo[...,index],hi[...,index]))
    return result


def _norm(a):
    squared = _sum(_square(a))
    return np.maximum(0., _down(np.sqrt(np.maximum(0.,squared[0])))), _up(np.sqrt(np.maximum(0.,squared[1])))


def _fraction_bounds(value):
    result = float(value)
    if not math.isfinite(result):
        raise UnprovenCurve('Fraction cannot be represented in finite float64')
    exact = _f(result)
    return (float(_down(result)) if exact > value else result,
            float(_up(result)) if exact < value else result)


def _sqrt_fraction_bounds(value):
    if value < 0:
        raise ValueError('Nonnegative square required')
    lower, upper = _fraction_bounds(value)
    lo = math.sqrt(max(0., lower)); hi = math.sqrt(max(0., upper))
    while _f(lo)**2 > value:
        lo = math.nextafter(lo, -math.inf)
    while _f(hi)**2 < value:
        hi = math.nextafter(hi, math.inf)
    return lo, hi


def _squared_exact(a, b):
    return sum((_f(x)-_f(y))**2 for x,y in zip(a,b))


@lru_cache(maxsize=1)
def pi_bounds():
    def atan_bounds(inverse):
        total = F(0)
        for index in range(64):
            total += F((-1)**index, (2*index+1)*inverse**(2*index+1))
        following = F(1,129*inverse**129)
        # 64 terms end with a negative term; the next positive term bounds the remainder.
        return total,total+following
    a,b = atan_bounds(5),atan_bounds(239)
    return 16*a[0]-4*b[1],16*a[1]-4*b[0]


@lru_cache(maxsize=8192)
def _sin_cos_bounds(angle):
    if angle[0] == angle[1] == 0:
        return (0.,0.),(1.,1.)
    pi_lo, pi_hi = pi_bounds()
    tau_mid = pi_lo+pi_hi
    midpoint = (angle[0]+angle[1])/2
    multiple = (midpoint/tau_mid+F(1,2)).numerator//(midpoint/tau_mid+F(1,2)).denominator
    products = sorted((2*multiple*pi_lo,2*multiple*pi_hi))
    reduced = angle[0]-products[1],angle[1]-products[0]
    midpoint = sum(reduced)/2
    uncertainty = (reduced[1]-reduced[0])/2
    if abs(midpoint) > 4 or uncertainty > F(1,10**20):
        raise UnprovenCurve('Finite angular reduction not certified at this precision')
    sine,cosine = midpoint,F(1)
    sin_term,cos_term = midpoint,F(1)
    for number in range(1,80):
        sin_term *= -midpoint*midpoint/F((2*number)*(2*number+1))
        cos_term *= -midpoint*midpoint/F((2*number-1)*(2*number))
        sine += sin_term; cosine += cos_term
        next_sin = sin_term*(-midpoint*midpoint)/F((2*number+2)*(2*number+3))
        next_cos = cos_term*(-midpoint*midpoint)/F((2*number+1)*(2*number+2))
        if max(abs(next_sin),abs(next_cos)) < F(1,10**45):
            # The remaining alternating terms decrease here. Trig is 1-Lipschitz in angle.
            sin_error,cos_error = abs(next_sin)+uncertainty,abs(next_cos)+uncertainty
            sin_range = max(F(-1),sine-sin_error),min(F(1),sine+sin_error)
            cos_range = max(F(-1),cosine-cos_error),min(F(1),cosine+cos_error)
            return (_fraction_bounds(sin_range[0])[0],_fraction_bounds(sin_range[1])[1]),(
                _fraction_bounds(cos_range[0])[0],_fraction_bounds(cos_range[1])[1])
    raise UnprovenCurve('Rational trigonometric remainder did not converge')


@dataclass(frozen=True)
class ConcreteCurve:
    record: dict

    def __post_init__(self):
        value = dict(self.record)
        kind = value.get('kind')
        if kind not in ('line','arc','polyline') or type(value.get('is_closed')) is not bool:
            raise ValueError('Explicit curve kind and boolean closure required')
        uncertainty = float(value.get('geometry_uncertainty_mm',0.))
        if not math.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError('Finite nonnegative declared geometry uncertainty required')
        value['geometry_uncertainty_mm'] = uncertainty
        if kind in ('line','polyline'):
            points = np.asarray(value['points'],float)
            if (points.ndim != 2 or points.shape[1] not in (2,3) or len(points)<2
                    or not np.isfinite(points).all() or (kind == 'line' and len(points)!=2)):
                raise ValueError('Finite original 2D/3D ordered points required')
            if kind == 'line' and value['is_closed']:
                raise ValueError('A positive straight child is not a closed loop')
            if all(np.array_equal(points[0],point) for point in points):
                raise ValueError('Positive curve geometry required; zero edges are otherwise retained')
            value['points'] = points.tolist()
            dimensions = points.shape[1]
        else:
            center,u,v = (np.asarray(value[key],float) for key in ('center_mm','u_axis','v_axis'))
            dimensions = len(center) if center.ndim == 1 else 0
            if dimensions not in (2,3) or any(a.shape!=(dimensions,) or not np.isfinite(a).all() for a in (center,u,v)):
                raise ValueError('Finite arc center and numerical orthonormal basis required')
            gram = np.array([[u@u,u@v],[v@u,v@v]])
            if np.max(np.abs(gram-np.eye(2))) > 1e-9:
                raise ValueError('Declared numerical circle basis is not orthonormal')
            radius,sweep = float(value['radius_mm']),float(value['sweep_rad'])
            turns = value['turn_count']
            if (not math.isfinite(radius) or radius<=0 or not math.isfinite(sweep) or sweep==0
                    or type(turns) is not int or turns<0):
                raise ValueError('Positive finite radius, explicit signed sweep and integer turn count required')
            if value['is_closed']:
                if turns<1 or abs(sweep) != turns*math.tau:
                    raise ValueError('Closed arc needs an explicit complete number of turns')
            elif not turns*math.tau <= abs(sweep) < (turns+1)*math.tau:
                raise ValueError('Open unwrapped sweep disagrees with its declared turn count')
            for key,array in (('center_mm',center),('u_axis',u),('v_axis',v)):
                value[key] = array.tolist()
            value['radius_mm'],value['sweep_rad'] = radius,sweep
        value['dimensions'] = dimensions
        object.__setattr__(self,'record',value)

    @property
    def kind(self):
        return self.record['kind']

    def vertices(self):
        if self.kind == 'arc':
            raise ValueError('Arc has analytic parameters, not stored polyline vertices')
        points = np.asarray(self.record['points'],float)
        if self.record['is_closed'] and not np.array_equal(points[-1],points[0]):
            points = np.vstack((points,points[0]))
        return points

    def _arc_scales(self):
        value = self.record
        u,v = map(lambda x:list(map(_f,x)),(value['u_axis'],value['v_axis']))
        uu,vv,uv = sum(x*x for x in u),sum(x*x for x in v),sum(x*y for x,y in zip(u,v))
        sigma_upper = _sqrt_fraction_bounds(max(uu+abs(uv),vv+abs(uv)))[1]
        if value['is_closed']:
            pi_lo,pi_hi = pi_bounds()
            sweep = 2*value['turn_count']*pi_lo,2*value['turn_count']*pi_hi
        else:
            sweep = _f(abs(value['sweep_rad'])),_f(abs(value['sweep_rad']))
        return sweep,_f(sigma_upper)

    def samples(self, step_mm=1., chord_mm=.1, point_budget=200001):
        if not (0<float(step_mm)<=1. and 0<float(chord_mm)<=.1 and type(point_budget) is int and point_budget>=2):
            raise ValueError('Frozen physical sampling limits required')
        points, lowers, uppers, span_lengths, mandatory = [],[],[],[],[]
        zero_edges = 0
        if self.kind != 'arc':
            vertices = self.vertices()
            for edge,(a,b) in enumerate(zip(vertices[:-1],vertices[1:])):
                length_hi = _sqrt_fraction_bounds(_squared_exact(a,b))[1]
                if length_hi == 0:
                    zero_edges += 1
                count = max(1,int(math.ceil(length_hi/step_mm)))
                while length_hi/count > step_mm:
                    count += 1
                if len(points)+count+1 > point_budget:
                    raise UnprovenCurve('Declared point budget exceeded; original span was not deleted')
                mandatory.append(len(points))
                # Keep shared vertices and every zero edge explicitly, including duplicate samples.
                for index in range(count+1):
                    t = F(index,count)
                    exact = [_f(x)+t*(_f(y)-_f(x)) for x,y in zip(a,b)]
                    bounds = [_fraction_bounds(x) for x in exact]
                    points.append([float(x) for x in exact])
                    lowers.append([x[0] for x in bounds]); uppers.append([x[1] for x in bounds])
                    if index:
                        span_lengths.append(_fraction_bounds(_f(length_hi)/count)[1])
                mandatory.append(len(points)-1)
            chord_error = 0.
        else:
            value = self.record
            sweep,sigma = self._arc_scales()
            radius = _f(value['radius_mm'])
            length_hi = _fraction_bounds(radius*sigma*sweep[1])[1]
            if not math.isfinite(length_hi) or length_hi<=0:
                raise UnprovenCurve('Arc physical range not representable')
            pi_lo,_ = pi_bounds()
            count = max(1,int(math.ceil(length_hi/step_mm)),int(math.ceil(float(sweep[1]/(pi_lo/2)))))
            while True:
                maximum_span = _fraction_bounds(radius*sigma*sweep[1]/count)[1]
                chord_error = _fraction_bounds(radius*sigma*(sweep[1]/count)**2/8)[1]
                if maximum_span<=step_mm and chord_error<=chord_mm:
                    break
                count = max(count+1, int(math.ceil(count*max(maximum_span/step_mm,math.sqrt(chord_error/chord_mm)))))
            if count+1>point_budget:
                raise UnprovenCurve('Arc point budget exceeded; full winding retained as unproven')
            sign = 1 if value['sweep_rad']>0 else -1
            for index in range(count+1):
                if index == count and value['is_closed']:
                    points.append(list(points[0])); lowers.append(list(lowers[0])); uppers.append(list(uppers[0])); continue
                t = F(index,count)
                angle = tuple(sorted((sign*t*sweep[0],sign*t*sweep[1])))
                sine,cosine = _sin_cos_bounds(angle)
                basis = _add(_mul(cosine,_exact(value['u_axis'])),_mul(sine,_exact(value['v_axis'])))
                world = _add(_exact(value['center_mm']),_mul(_exact(value['radius_mm']),basis))
                if not np.isfinite(world).all():
                    raise UnprovenCurve('Arc point enclosure not finite')
                center = world[0]+(world[1]-world[0])/2
                points.append(center.tolist()); lowers.append(world[0].tolist()); uppers.append(world[1].tolist())
            mandatory = [0,count]
            span_lengths = [maximum_span]*count
        points,lowers,uppers = map(np.asarray,(points,lowers,uppers))
        error_axes = np.maximum(np.abs(_down(lowers-points)),np.abs(_up(uppers-points)))
        point_errors = _norm(_exact(error_axes))[1]
        return dict(points=points, point_enclosure_lower=lowers, point_enclosure_upper=uppers,
                    point_error_upper_mm=point_errors, maximum_step_upper_mm=max(span_lengths),
                    analytic_chord_error_upper_mm=chord_error, mandatory_vertex_sample_indices=mandatory,
                    original_zero_edges_retained=zero_edges, original_kind=self.kind,
                    positive_length_cutoff_applied=False, point_count=len(points))


def point_polyline_distance_bounds(points,vertices,query_block=32,segment_block=128):
    """Full finite segment union; outward interval projection, including zero edges."""
    q,p = np.asarray(points,float),np.asarray(vertices,float)
    if q.ndim!=2 or p.ndim!=2 or len(p)<2 or q.shape[1]!=p.shape[1] or not np.isfinite(q).all() or not np.isfinite(p).all():
        raise ValueError('Finite matching-dimensional queries and original segment vertices required')
    result_lo,result_hi = np.full(len(q),np.inf),np.full(len(q),np.inf)
    for qi in range(0,len(q),query_block):
        query = _exact(q[qi:qi+query_block,None,:])
        lo,hi = np.full(len(query[0]),np.inf),np.full(len(query[0]),np.inf)
        for si in range(0,len(p)-1,segment_block):
            stop=min(len(p)-1,si+segment_block)
            a,b = _exact(p[None,si:stop,:]),_exact(p[None,si+1:stop+1,:])
            delta,relative = _sub(b,a),_sub(query,a)
            numerator,denominator = _sum(_mul(relative,delta)),_sum(_square(delta))
            # Intervals that include zero denominator get [0,1]; they are not removed or regularized.
            safe = denominator[0]>0
            divisor_lo = np.where(safe,denominator[0],1.)
            divisor_hi = np.where(safe,denominator[1],1.)
            quotients = np.stack(np.broadcast_arrays(numerator[0]/divisor_lo,numerator[0]/divisor_hi,
                                                     numerator[1]/divisor_lo,numerator[1]/divisor_hi))
            t = (np.where(safe,np.clip(_down(quotients.min(axis=0)),0.,1.),0.),
                 np.where(safe,np.clip(_up(quotients.max(axis=0)),0.,1.),1.))
            residual = _sub(relative,_mul((t[0][...,None],t[1][...,None]),delta))
            distance = _norm(residual)
            if not np.isfinite(distance).all():
                raise UnprovenCurve('Finite segment distance interval overflow')
            lo = np.minimum(lo,distance[0].min(axis=1)); hi=np.minimum(hi,distance[1].min(axis=1))
        result_lo[qi:qi+len(lo)],result_hi[qi:qi+len(hi)] = lo,hi
    return result_lo,result_hi


def intrinsic_semantics(predicted,reference):
    a,b = predicted.record,reference.record
    reasons = []
    if a['kind']!=b['kind']:
        reasons.append('CURVE_KIND_MISMATCH')
    if a['is_closed']!=b['is_closed']:
        reasons.append('EXPLICIT_CLOSURE_MISMATCH')
    status = 'MATCH_CONDITIONAL_ON_OUTPUT_FRAME_AND_PARENT_ORDER'
    if a['kind']==b['kind']=='arc':
        if a['turn_count']!=b['turn_count']:
            reasons.append('UNWRAPPED_TURN_COUNT_MISMATCH')
        if a.get('orientation_frame_id') is None or b.get('orientation_frame_id') is None:
            status = 'UNPROVEN_ORIENTED_WINDING_FRAME'
        elif a['orientation_frame_id']!=b['orientation_frame_id']:
            status = 'UNPROVEN_DIFFERENT_ORIENTATION_FRAMES'
        elif a.get('orientation_frame_basis') is None or b.get('orientation_frame_basis') is None:
            status='UNPROVEN_ORIENTATION_FRAME_BASIS'
        else:
            frame_a,frame_b=map(lambda value:np.asarray(value['orientation_frame_basis'],float),(a,b))
            if (frame_a.shape!=(2,a['dimensions']) or frame_b.shape!=frame_a.shape
                    or not np.isfinite(frame_a).all() or not np.isfinite(frame_b).all()):
                raise ValueError('Two finite frozen orientation-frame axes required')
            if not np.array_equal(frame_a,frame_b):
                status='UNPROVEN_INCONSISTENT_FROZEN_FRAME_BASIS'
            else:
                def rotation(value):
                    u,v=map(lambda x:list(map(_f,x)),(value['u_axis'],value['v_axis']))
                    fu,fv=map(lambda x:list(map(_f,x)),frame_a)
                    if a['dimensions']==2:
                        determinant=(u[0]*v[1]-u[1]*v[0])*(fu[0]*fv[1]-fu[1]*fv[0])
                    else:
                        def cross(x,y):
                            return [x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]]
                        determinant=sum(x*y for x,y in zip(cross(u,v),cross(fu,fv)))
                    return ((determinant>0)-(determinant<0))*(1 if value['sweep_rad']>0 else -1)
                ra,rb=rotation(a),rotation(b)
                if not ra or not rb:
                    status='UNPROVEN_ORIENTED_ROTATION'
                elif ra!=rb:
                    reasons.append('SIGNED_ROTATION_IN_FIXED_FRAME_MISMATCH')
    return dict(status='MISMATCH' if reasons else status,reasons=reasons,
                actual_frozen_frame_binding_verified=False,
                global_parent_order_and_instance_identity_verified=False,
                polyline_closed_order_requires_separate_span_correspondence=True)


def _direct_polygon_bound(predicted,reference):
    if predicted.kind=='arc' or reference.kind=='arc':
        return None
    a,b = predicted.vertices(),reference.vertices()
    if a.shape!=b.shape:
        return None
    squared = max(_squared_exact(x,y) for x,y in zip(a,b))
    uncertainty = _f(predicted.record['geometry_uncertainty_mm'])+_f(reference.record['geometry_uncertainty_mm'])
    threshold = F(3)-uncertainty
    if threshold>=0 and squared<=threshold*threshold:
        return dict(lower_mm=0.,upper_mm=min(3.,_fraction_bounds(_f(_sqrt_fraction_bounds(squared)[1])+uncertainty)[1]),
                    proof='EXACT_CORRESPONDING_ORDERED_VERTICES_AND_CONVEX_INTERPOLATION',
                    exact_maximum_vertex_squared_error=dict(numerator=str(squared.numerator),denominator=str(squared.denominator)))
    return None


def certify_concrete_pair(predicted,reference,policy=None):
    """Ordered endpoint and symmetric continuous distance; no global selection proof."""
    if not isinstance(predicted,ConcreteCurve): predicted=ConcreteCurve(predicted)
    if not isinstance(reference,ConcreteCurve): reference=ConcreteCurve(reference)
    if predicted.record['dimensions']!=reference.record['dimensions']:
        raise ValueError('Both concrete curves need the same physical coordinate frame dimension')
    policy = {**POLICY,**(policy or {})}
    if policy['maximum_step_mm']>1 or policy['maximum_chord_mm']>.1 or policy['max_refinements']<0:
        raise ValueError('Contract sampling cannot be relaxed')
    result = dict(version=VERSION,limit_mm=3.,scope='CONDITIONAL_CONCRETE_CURVE_GEOMETRY_ONLY',
                  intrinsic_semantics=intrinsic_semantics(predicted,reference),
                  global_runtime_slot_capacity_and_parent_order_verified=False,
                  own_finite_domain_coverage_verified=False, material_or_weld_identity_verified=False,
                  actual_all111_or_all1605_joint_expression_run=False,
                  model_inference_calls=0,actual_weld_optimizer_updates=0,full_contract_pass=False)
    rounds = []
    direct = _direct_polygon_bound(predicted,reference)
    uncertainty = _fraction_bounds(_f(predicted.record['geometry_uncertainty_mm'])+_f(reference.record['geometry_uncertainty_mm']))[1]
    for refinement in range(policy['max_refinements']+1):
        try:
            step=policy['maximum_step_mm']/2**refinement
            chord=policy['maximum_chord_mm']/4**refinement
            a,b = (curve.samples(step,chord,policy['point_budget_per_curve']) for curve in (predicted,reference))
            ea,eb = a['point_error_upper_mm'],b['point_error_upper_mm']
            endpoints = _norm(_sub((a['point_enclosure_lower'][[0,-1]],a['point_enclosure_upper'][[0,-1]]),
                                   (b['point_enclosure_lower'][[0,-1]],b['point_enclosure_upper'][[0,-1]])))
            endpoint_lo = np.maximum(0.,_down(endpoints[0]-uncertainty))
            endpoint_hi = _up(endpoints[1]+uncertainty)
            if direct is not None:
                # Direct proof also covers exact endpoints at the 3mm boundary without numeric widening.
                endpoint_hi = np.minimum(endpoint_hi,direct['upper_mm'])
                metric = direct
                directed=[]
            else:
                directed=[]
                for source,target in ((a,b),(b,a)):
                    lo,hi = point_polyline_distance_bounds(source['points'],target['points'],
                                                          policy['query_block'],policy['segment_block'])
                    target_error = _up(target['analytic_chord_error_upper_mm']+target['point_error_upper_mm'].max())
                    allowance = _up(target_error+uncertainty)
                    lower = max(0.,float(_down(_down(lo-source['point_error_upper_mm'])-allowance).max()))
                    upper = float(_up(_up(_up(hi+source['point_error_upper_mm']).max()+allowance)+_up(source['maximum_step_upper_mm']/2)))
                    directed.append(dict(lower_mm=lower,upper_mm=upper))
                metric=dict(lower_mm=max(x['lower_mm'] for x in directed),upper_mm=max(x['upper_mm'] for x in directed),
                            proof='OUTWARD_SEGMENT_DISTANCES_TARGET_CHORD_BOUND_AND_SOURCE_HALF_STEP')
            row=dict(refinement=refinement,maximum_step_upper_mm=max(a['maximum_step_upper_mm'],b['maximum_step_upper_mm']),
                     maximum_analytic_chord_error_upper_mm=max(a['analytic_chord_error_upper_mm'],b['analytic_chord_error_upper_mm']),
                     points=[a['point_count'],b['point_count']],ordered_endpoint_lower_mm=endpoint_lo.tolist(),
                     ordered_endpoint_upper_mm=endpoint_hi.tolist(),bidirectional_whole_curve=metric,directed=directed,
                     retained_zero_edges=[a['original_zero_edges_retained'],b['original_zero_edges_retained']])
            rounds.append(row)
            if max(endpoint_hi)<=3. and metric['upper_mm']<=3.:
                geometry='PASS_CONSERVATIVE_ORDERED_ENDPOINTS_AND_BIDIRECTIONAL_WHOLE_CURVE'
                break
            if max(endpoint_lo)>3. or metric['lower_mm']>3.:
                geometry='PROVEN_GEOMETRIC_MISMATCH'
                break
            if direct is not None:
                raise AssertionError('An exact direct proof should settle the geometry')
        except UnprovenCurve as error:
            geometry='UNPROVEN_RESOURCE_OR_NUMERIC_LIMIT'
            result['unproven_reason']=str(error)
            break
    else:
        geometry='UNPROVEN_AT_FROZEN_REFINEMENT_BUDGET'
    semantic=result['intrinsic_semantics']['status']
    combined = ('CONDITIONAL_GEOMETRY_AND_INTRINSIC_TYPE_CLOSURE_MATCH'
                if geometry.startswith('PASS_') and semantic.startswith('MATCH_') else
                'INTRINSIC_SEMANTIC_MISMATCH' if semantic=='MISMATCH' else
                'UNPROVEN_INTRINSIC_SEMANTICS' if geometry.startswith('PASS_') else geometry)
    return {**result,'status':combined,'geometry_status':geometry,'rounds':rounds,
            'unproven_is_not_deleted_or_scored_as_success':True}


def from_reference_237(reference_metadata,domain_frame_id=None,orientation_frame_basis=None):
    """Use the previously checked analytic reference, only for offline scoring."""
    analytic=reference_metadata['analytic']
    if analytic['kind']=='line':
        return ConcreteCurve(dict(kind='line',is_closed=False,points=[analytic['start_mm'],analytic['end_mm']]))
    parameters=analytic['parameters']
    normal,u = np.asarray(parameters['plane_normal']),np.asarray(parameters['start_direction'])
    return ConcreteCurve(dict(kind='arc',center_mm=parameters['center_mm'],radius_mm=parameters['radius_mm'],
        u_axis=u.tolist(),v_axis=np.cross(normal,u).tolist(),sweep_rad=parameters['sweep_rad'],
        is_closed=parameters['is_closed'],turn_count=1 if parameters['is_closed'] else 0,
        orientation_frame_id=domain_frame_id,orientation_frame_basis=orientation_frame_basis,
        geometry_uncertainty_mm=analytic['reference_geometry_uncertainty_allowance_mm']))
