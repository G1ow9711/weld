"""Additional curve-shape checks, separate from the frozen endpoint scorer."""
import numpy as np


def _points(points):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<2 or not np.isfinite(p).all():
        raise ValueError('A finite 3D polyline with at least two vertices is required')
    return p


def _samples(p,step):
    result=[];max_interval=0.
    for a,b in zip(p[:-1],p[1:]):
        length=float(np.linalg.norm(b-a));count=max(1,int(np.ceil(length/step)))
        result.append(np.linspace(a,b,count+1));max_interval=max(max_interval,length/count)
    return np.concatenate(result),max_interval


def _distance_to_polyline(q,p):
    a=p[:-1];delta=np.diff(p,axis=0);length2=np.sum(delta*delta,axis=1)
    out=[]
    for block in np.array_split(q,max(1,int(np.ceil(len(q)/256)))):
        v=block[:,None]-a[None]
        t=np.sum(v*delta[None],axis=2)/np.maximum(length2,1e-30)
        nearest=a[None]+np.clip(t,0,1)[:,:,None]*delta[None]
        out.extend(np.sqrt(np.sum((block[:,None]-nearest)**2,axis=2)).min(1))
    return np.asarray(out)


def polyline_hausdorff_bounds(first,second,step_mm=.5):
    if not np.isfinite(step_mm) or step_mm<=0:raise ValueError('Positive finite step required')
    a,b=_points(first),_points(second);qa,da=_samples(a,step_mm);qb,db=_samples(b,step_mm)
    lower=float(max(_distance_to_polyline(qa,b).max(),_distance_to_polyline(qb,a).max()))
    # Distance to a fixed set is 1-Lipschitz. A sample is at most half an
    # interval from every point of the represented polyline.
    upper=lower+max(da,db)/2
    return {'lower_mm':lower,'upper_mm':upper,'sampling_step_mm':step_mm,
            'length_difference_mm':float(abs(np.linalg.norm(np.diff(a,axis=0),axis=1).sum()-
                                           np.linalg.norm(np.diff(b,axis=0),axis=1).sum()))}


def check_matched_curves(predictions,truths,endpoint_score,limit_mm=3.):
    rows=[];uncertain_or_failed=set()
    for pair in endpoint_score['per_truth']:
        if not pair['accepted']:continue
        pi,ti=pair['prediction_index'],pair['truth_index'];pred=predictions[pi]['segments'];truth=truths[ti]['segments']
        if pair['reversed']:pred=[{**s,'points':s['points'][::-1]} for s in pred[::-1]]
        for si,(p,t) in enumerate(zip(pred,truth)):
            if t['kind']!='arc':continue
            metric=polyline_hausdorff_bounds(p['points'],t['points'])
            if metric['lower_mm']<=limit_mm<metric['upper_mm']:
                metric=polyline_hausdorff_bounds(p['points'],t['points'],step_mm=.05)
            status='PASS' if metric['upper_mm']<=limit_mm else ('FAIL' if metric['lower_mm']>limit_mm else 'BOUNDARY_UNCERTAIN')
            if status!='PASS':uncertain_or_failed.add(ti)
            rows.append({'prediction_index':pi,'truth_index':ti,'segment_index':si,'status':status,**metric})
    return {'scope':'Additional symmetric polyline-geometry check on endpoint-matched arcs; not a replacement for one-to-one or segmentation scoring.',
            'limit_mm':limit_mm,'endpoint_valid_welds':endpoint_score['valid_welds'],
            'valid_welds_after_curve_check':endpoint_score['valid_welds']-len(uncertain_or_failed),
            'arc_segments_checked':len(rows),'all_checked_arcs_verified':not uncertain_or_failed,'per_arc':rows}
