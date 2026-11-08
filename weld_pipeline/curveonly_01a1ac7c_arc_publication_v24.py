"""Endpoint-preserving local arcs with continuous conversion-error bounds."""
import math,time
import numpy as np
from . import curveonly_01a1ac7c as cad
from . import curveonly_01a1ac7c_arc_publication_v6 as old
MAX_ERROR=old.MAX_CONVERSION_ERROR_MM
TURN_LIMIT=old.TURN_LIMIT
QUERY_SPACING=.05
MAX_PIECES=64


def candidate(points):
    p=np.asarray(points,float);delta=p[-1]-p[0];chord=np.linalg.norm(delta)
    if len(p)<3 or chord<1e-7:return None
    u=delta/chord;middle=(p[0]+p[-1])/2;x=p-middle
    perp=x-np.outer(x@u,u)
    if np.linalg.norm(perp,axis=1).max()<.05:return None
    _,_,vh=np.linalg.svd(perp,full_matrices=False);v=vh[0];v-=u*(v@u);v/=np.linalg.norm(v)
    a=2*x@v;b=(x*x).sum(1)-(chord/2)**2;den=float(a@a)
    if den<1e-12:return None
    height=float(a@b/den);center=middle+height*v;radius=float(np.linalg.norm(p[0]-center))
    if radius<=0 or not np.isfinite(radius):return None
    normal=np.cross(u,v);normal/=np.linalg.norm(normal);start=(p[0]-center)/radius;cross=np.cross(normal,start)
    angles=np.unwrap(np.arctan2((p-center)@cross,(p-center)@start));sweep=float(angles[-1])
    if abs(sweep)<1e-7 or abs(sweep)>TURN_LIMIT:return None
    if np.min(np.diff(angles)*np.sign(sweep)) < -1e-8:return None
    plane=abs((p-center)@normal);radial=abs(np.linalg.norm(p-center,axis=1)-radius)
    if max(plane.max(),radial.max())>MAX_ERROR:return None
    n=max(3,int(math.ceil(abs(sweep)*radius))+1,int(math.ceil(abs(sweep)/.02))+1)
    while radius*(1-math.cos(abs(sweep)/(n-1)/2))>.001:n=2*n-1
    grid=np.linspace(0.,sweep,n);arc=center+radius*(np.cos(grid)[:,None]*start+np.sin(grid)[:,None]*cross)
    if max(np.linalg.norm(arc[0]-p[0]),np.linalg.norm(arc[-1]-p[-1]))>1e-6:return None
    # Along either curve, distance to the other set is1-Lipschitz in arclength.
    # Endpoint-inclusive samples leave at most half a sample gap unobserved.
    L=float(cad.arclength(p)[-1]);source=cad.resample(p,QUERY_SPACING)
    qn=max(3,int(math.ceil(radius*abs(sweep)/QUERY_SPACING))+1);qg=np.linspace(0.,sweep,qn)
    actual=center+radius*(np.cos(qg)[:,None]*start+np.sin(qg)[:,None]*cross)
    chord_error=radius*(1-math.cos(abs(sweep)/(n-1)/2))
    bound_source=float(old.distance_upper(source,arc).max()+L/(len(source)-1)/2+chord_error)
    bound_arc=float(old.distance_upper(actual,p).max()+radius*abs(sweep)/(qn-1)/2)
    bound=max(bound_source,bound_arc)
    if bound>MAX_ERROR:return None
    parameters=dict(schema_version='contract_3d_circular_arc_chord_v1',center_mm=center.tolist(),radius_mm=radius,
        plane_normal=normal.tolist(),start_direction=start.tolist(),sweep_rad=sweep,is_closed=False,
        units='mm',angle_units='rad',parameterization='C + R*(cos(sign(sweep)*s/R)*U + sin(sign(sweep)*s/R)*(N cross U))')
    return dict(points=arc.tolist(),length_mm=radius*abs(sweep),geometry_type='arc',arc_parameters=parameters,
        source_conversion_continuous_error_upper_mm=bound,endpoint_preserving=True,
        status='NOMINAL_LOCAL_ARC_GEOMETRY_AND_TURN_CERTIFIED',reason='BOUNDED_LOCAL_GEOMETRY_REPRESENTATION')


def convert(points):
    stack=[(np.asarray(points,float),0)];out=[]
    while stack:
        p,depth=stack.pop();arc=candidate(p)
        if arc is not None:out.append(arc);continue
        if depth>=10 or len(out)+len(stack)+2>MAX_PIECES:return None
        # A genuinely straight span must stay a line/general curve; it cannot
        # receive an invented large-radius circular certificate.
        chord=p[-1]-p[0];size=np.linalg.norm(chord)
        if size>1e-8 and np.linalg.norm(np.cross(p-p[0],chord/size),axis=1).max()<.05:return None
        L=float(cad.arclength(p)[-1])
        if L<.1:return None
        stack.append((cad.slice_curve(p,L/2,L),depth+1));stack.append((cad.slice_curve(p,0,L/2),depth+1))
    return out


def publish(decoded):
    start=time.perf_counter();parents=[];records=[]
    for parent in decoded['parents']:
        pieces=[];changed=False;bounds=[]
        for source_index,s in enumerate(parent['segments']):
            arcs=None if 'arc_parameters'in s else convert(s['points'])
            if arcs is None:pieces.append(dict(s));continue
            changed=True
            for a in arcs:
                a.update(source_segment_index=source_index,source_representation_reason=s.get('reason'))
                pieces.append(a);bounds.append(a['source_conversion_continuous_error_upper_mm'])
        if changed and any(np.linalg.norm(np.asarray(a['points'])[-1]-np.asarray(b['points'])[0])>1e-6 for a,b in zip(pieces[:-1],pieces[1:])):
            changed=False;bounds=[];pieces=[dict(s)for s in parent['segments']]
        if changed:
            for j,s in enumerate(pieces):s['segment_index']=j
            points=np.vstack([np.asarray(pieces[0]['points']),*[np.asarray(s['points'])[1:]for s in pieces[1:]]])
            p=dict(parent,points=points.tolist(),segments=pieces,arc_publication_status='BOUNDED_ENDPOINT_PRESERVING_LOCAL_ARCS',
                nominal_source_points=parent['points'],source_conversion_continuous_error_upper_mm=max(bounds))
        else:p=parent
        parents.append(p);records.append(dict(changed=changed,segments=len(pieces),maximum_continuous_conversion_bound_mm=max(bounds)if bounds else None))
    return dict(decoded,parents=parents,published_segments=sum(len(p['segments'])for p in parents),
        local_arc_publication=dict(records=records,maximum_conversion_error_mm=MAX_ERROR,maximum_turn_rad=TURN_LIMIT,
            maximum_pieces_per_source_segment=MAX_PIECES,query_spacing_mm=QUERY_SPACING,
            continuous_Lipschitz_conversion_certificate=True,labels_or_references_read=False,seconds=time.perf_counter()-start))
