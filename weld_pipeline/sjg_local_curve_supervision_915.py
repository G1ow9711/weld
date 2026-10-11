"""Typed local line/arc event labels for TRAIN; no target reads at inference."""
import numpy as np
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier


def compatible_interval(profile,representation,tolerance_mm=3.,_candidate=None,_target=None,_points=None):
    if profile['representation']['kind']!=representation['kind']:return None
    candidate=_candidate if _candidate is not None else CurveCarrier(profile['representation'])
    target=_target if _target is not None else CurveCarrier(representation)
    if candidate.length+2*tolerance_mm<target.length:return None
    # For two line segments, distance to a convex segment is convex along
    # the other segment: endpoints give the exact directed maximum.
    points=_points if _points is not None else target.at(np.array([0.,target.length]) if target.kind=='line' else target.grid(1.))
    q,distance=candidate.project(points)
    maximum=float(distance.max())
    if maximum>tolerance_mm:return None
    closed=candidate.kind=='arc' and candidate.arc.is_closed
    if closed:q=np.unwrap(q*2*np.pi/candidate.length)*candidate.length/(2*np.pi)
    change=np.diff(q);direction=1 if np.median(change)>=0 else -1
    if np.any(direction*change < -1e-5):return None
    lo,hi=float(q.min()),float(q.max());span=hi-lo
    if span<=1e-8 or span>candidate.length+1e-6:return None
    whole=bool(target.kind=='arc' and target.arc.is_closed and abs(span-candidate.length)<1e-5)
    if closed:
        lo%=candidate.length
        if candidate.length-lo<1e-7:lo=0.
        hi=lo+span;wrap=bool(hi>candidate.length+1e-7)
        intervals=[[lo,candidate.length],[0.,hi-candidate.length]] if wrap else [[lo,min(hi,candidate.length)]]
        if whole:intervals=[[0.,candidate.length]]
        start,end=lo,hi%candidate.length
    else:
        lo=max(0.,lo);hi=min(candidate.length,hi);wrap=False;intervals=[[lo,hi]];start,end=lo,hi
    # Check the reverse distance for the fitted subarc/line, not the whole
    # longer carrier. This rejects a nearest-point match with the wrong path.
    reverse_error=0.
    for left,right in intervals:
        if right-left<=1e-8:continue
        sub=CurveCarrier(candidate.segment(left,right))
        sample=np.array([0.,sub.length]) if sub.kind=='line' else sub.grid(1.)
        _,d=target.project(sub.at(sample))
        reverse_error=max(reverse_error,float(d.max()))
    maximum=max(maximum,reverse_error)
    if maximum>tolerance_mm:return None
    return dict(lo=start,hi=end,span_mm=span,intervals=intervals,wrap=wrap,closed_whole=whole,
                target_direction_forward=bool(direction>0),maximum_bidirectional_sample_error_mm=maximum)


def supervise(profiles,target):
    if target.get('coordinate_frame')!='workpiece_local' or target.get('unit')!='mm':raise ValueError('Local millimetre targets required')
    by_profile={};children=[];missing=[]
    carriers=[CurveCarrier(p['representation']) for p in profiles]
    bounds=[]
    for c in carriers:
        if c.kind=='line':bounds.append([c.ends.min(0),c.ends.max(0)])
        else:
            # The complete supporting circle bounds every possible subarc;
            # deliberately conservative, never an approximate sampled box.
            first=c.arc.start_direction;second=np.cross(c.arc.plane_normal,first)
            radius=c.arc.radius_mm*np.sqrt(first**2+second**2)
            bounds.append([c.arc.center_mm-radius,c.arc.center_mm+radius])
    bounds=np.asarray(bounds,float).reshape(-1,2,3)
    lengths=np.asarray([c.length for c in carriers]);kinds=np.asarray([c.kind for c in carriers])
    for pi,parent in enumerate(target['parents']):
        for ci,child in enumerate(parent['children']):
            options=[]
            curve=CurveCarrier(child['representation'])
            points=curve.at(np.array([0.,curve.length]) if curve.kind=='line' else curve.grid(1.))
            possible=(kinds==curve.kind)&(lengths+6>=curve.length)
            possible &= np.all(points.min(0)>=bounds[:,0]-3.-1e-8,axis=1)
            possible &= np.all(points.max(0)<=bounds[:,1]+3.+1e-8,axis=1)
            for k in np.flatnonzero(possible):
                k=int(k);profile=profiles[k]
                event=compatible_interval(profile,child['representation'],_candidate=carriers[k],_target=curve,_points=points)
                if event is not None:options.append(dict(profile=k,**event))
            identity=dict(parent=pi,child=ci,original_child_id=child['original_child_id'],kind=child['representation']['kind'])
            if not options:
                missing.append(dict(**identity,reason='NO_TYPED_CARRIER_WITHIN_3MM'));continue
            # All candidates are retained as alternative geometric supervision;
            # they are not claimed to be distinct true material contacts.
            for event in options:by_profile.setdefault(str(event['profile']),[]).append(dict(**identity,**event))
            children.append(dict(**identity,profiles=[e['profile'] for e in options],
                                 best_error_mm=min(e['maximum_bidirectional_sample_error_mm'] for e in options)))
    assert len(children)+len(missing)==target['original_children']
    return dict(original_G=target['original_children'],covered_children=len(children),children=children,missing=missing,
        by_profile=by_profile,labels_are_geometric_compatibility_not_material_identity=True,
        original_parent_count=len(target['parents']),not_extraction_accuracy=True)


def event_targets(profile,events,q):
    curve=CurveCarrier(profile['representation']);q=np.asarray(q,float)
    target=np.zeros((len(q),4),np.float32);offset=np.zeros((len(q),2),np.float32);mask=np.zeros((len(q),2),bool)
    if not events:return target,offset,mask
    for event in events:
        for left,right in event['intervals']:target[:,0]=np.maximum(target[:,0],(q>=left)&(q<=right))
        if event['closed_whole']:target[:,3]=1
    for head,name in enumerate(('lo','hi')):
        delta=np.asarray([e[name] for e in events if not e['closed_whole']])[None]-q[:,None]
        if not delta.shape[1]:continue
        if curve.kind=='arc' and curve.arc.is_closed:delta=(delta+curve.length/2)%curve.length-curve.length/2
        nearest=delta[np.arange(len(q)),np.abs(delta).argmin(1)]
        target[:,head+1]=np.exp(-.5*(nearest/2.)**2)
        offset[:,head]=nearest/10.;mask[:,head]=abs(nearest)<=8.
    return target,offset,mask


def decode_interval(representation,lo,hi,wrap=False,whole=False):
    curve=CurveCarrier(representation)
    if whole:
        if curve.kind!='arc' or not curve.arc.is_closed:raise ValueError('Whole-loop output requires an observed closed carrier')
        return representation
    if not wrap:return curve.segment(lo,hi)
    if curve.kind!='arc' or not curve.arc.is_closed or not 0<=hi<lo<curve.length:raise ValueError('Invalid wrapped interval')
    from weld_pipeline.contract_curve_geometry import CircularArc
    a=curve.arc;length=curve.length-lo+hi;start=(curve.at(lo)-a.center_mm)/a.radius_mm
    arc=CircularArc(a.center_mm,a.radius_mm,a.plane_normal,start,np.sign(a.sweep_rad)*length/a.radius_mm,False)
    return dict(kind='arc',arc_parameters=arc.to_parameters())


def controls():
    from weld_pipeline.contract_curve_geometry import CircularArc
    line=dict(representation=dict(kind='line',endpoints_mm=[[0.,0.,0.],[100.,0.,0.]]))
    event=compatible_interval(line,dict(kind='line',endpoints_mm=[[20.,0.,0.],[40.,0.,0.]]))
    assert event is not None and event['lo']==20 and event['hi']==40 and not event['wrap']
    target,offset,mask=event_targets(line,[event],np.array([10.,20.,30.,40.,50.]))
    assert np.array_equal(target[:,0],[0,1,1,1,0]) and target[1,1]==1 and target[3,2]==1
    assert not compatible_interval(line,dict(kind='line',endpoints_mm=[[20.,4.,0.],[40.,4.,0.]]))
    circle=CircularArc([0.,0.,0.],100.,[0.,0.,1.],[1.,0.,0.],2*np.pi,True)
    profile=dict(representation=dict(kind='arc',arc_parameters=circle.to_parameters()))
    checked=[]
    for forward in (True,False):
        angle=np.deg2rad(350 if forward else 10)
        arc=CircularArc([0.,0.,0.],100.,[0.,0.,1.],[np.cos(angle),np.sin(angle),0.],np.deg2rad(20 if forward else -20))
        rep=dict(kind='arc',arc_parameters=arc.to_parameters());event=compatible_interval(profile,rep)
        assert event and event['wrap'] and event['target_direction_forward']==forward
        decoded=CurveCarrier(decode_interval(profile['representation'],event['lo'],event['hi'],wrap=True))
        truth=CurveCarrier(rep);assert truth.project(decoded.at(decoded.grid()))[1].max()<1e-8
        t,_,_=event_targets(profile,[event],np.array([0.,circle.length_mm/2]))
        assert np.array_equal(t[:,0],[1,0]);checked.append(dict(wrap=True,target_forward=forward))
    whole=compatible_interval(profile,profile['representation']);assert whole and whole['closed_whole']
    q=CurveCarrier(profile['representation']).grid();target,_,_=event_targets(profile,[whole],q)
    assert np.all(target[:,0]==1) and np.all(target[:,3]==1) and not np.any(target[:,1:3])
    assert compatible_interval(line,profile['representation']) is None
    return dict(line_occupancy_endpoints_and_off_curve_rejection=True,wrapped_circle_directions=checked,
                whole_circle_occupancy_without_false_endpoint=True,kind_mismatch_rejected=True)
