"""One-to-one weld acceptance scoring. Never used by the inference pipeline."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def welding_type(value):
    value=str(value).lower()
    if value in ('horizontal','h','平焊缝') or 'horizontal' in value: return 'horizontal'
    if value in ('vertical','v','立焊缝') or 'vertical' in value: return 'vertical'
    raise ValueError(f'Unknown welding type: {value}')


def endpoints(weld):
    segments=weld['segments']
    if not segments: raise ValueError('Empty weld')
    out=np.asarray([segments[0]['points'][0],segments[-1]['points'][-1]],dtype=float)
    if out.shape!=(2,3) or not np.isfinite(out).all():raise ValueError('Invalid endpoints')
    return out


def topology(weld):
    """Count and ordered curve kinds are mandatory."""
    return tuple(s['kind'] for s in weld['segments'])


def pair_errors(prediction,truth,allow_reverse=True):
    p,t=endpoints(prediction),endpoints(truth)
    direct=np.linalg.norm(p-t,axis=1)
    flipped=np.linalg.norm(p[::-1]-t,axis=1)
    reverse=allow_reverse and (flipped.max(),flipped.sum())<(direct.max(),direct.sum())
    errors=flipped if reverse else direct
    pt=topology(prediction)
    if reverse: pt=pt[::-1]
    segment_error=None
    if pt==topology(truth):
        predicted=prediction['segments']
        if reverse:predicted=[{**s,'points':s['points'][::-1]} for s in predicted[::-1]]
        pe=np.asarray([p for s in predicted for p in [s['points'][0],s['points'][-1]]],float)
        te=np.asarray([p for s in truth['segments'] for p in [s['points'][0],s['points'][-1]]],float)
        if not np.isfinite(pe).all() or not np.isfinite(te).all():raise ValueError('Invalid segment endpoints')
        segment_error=float(np.linalg.norm(pe-te,axis=1).max())
    return {'start_mm':float(errors[0]),'end_mm':float(errors[1]),'reversed':bool(reverse),
        'type_correct':welding_type(prediction['type'])==welding_type(truth['type']),
        'topology_correct':pt==topology(truth),'segment_endpoint_max_mm':segment_error}


def score_welds(predictions,truths,endpoint_mm=3.,allow_reverse=True,segment_endpoint_mm=3.):
    n,m=len(predictions),len(truths)
    if not m: raise ValueError('An empty truth set requires separate negative-sample scoring')
    pairs={};valid=np.zeros((n,m),dtype=bool);cost=np.full((n,m),1e9)
    for i,p in enumerate(predictions):
        for j,t in enumerate(truths):
            e=pair_errors(p,t,allow_reverse)
            ok=e['type_correct'] and e['topology_correct'] and max(e['start_mm'],e['end_mm'])<=endpoint_mm
            ok=ok and e['segment_endpoint_max_mm'] is not None and e['segment_endpoint_max_mm']<=segment_endpoint_mm
            valid[i,j]=ok;pairs[i,j]=e
            # The large invalid-pair penalty maximizes valid matching cardinality
            # before minimizing endpoint error, rather than greedily taking neighbors.
            error=min(e['start_mm']+e['end_mm'],1e5)
            cost[i,j]=(0 if ok else 1e9)+error
    rows=[];matched_p=set();matched_t=set()
    if n:
        ii,jj=linear_sum_assignment(cost)
        for i,j in zip(ii,jj):
            if valid[i,j]:
                matched_p.add(int(i));matched_t.add(int(j))
                rows.append({'prediction_index':int(i),'truth_index':int(j),
                    'prediction_id':predictions[i].get('id',str(i)),'truth_id':truths[j].get('id',str(j)),
                    'accepted':True,**pairs[i,j]})
    for j,t in enumerate(truths):
        if j in matched_t:continue
        if n:
            candidates=[i for i,p in enumerate(predictions) if welding_type(p['type'])==welding_type(t['type'])]
            candidates=candidates or list(range(n))
            i=min(candidates,key=lambda i:max(pairs[i,j]['start_mm'],pairs[i,j]['end_mm']))
            rows.append({'truth_index':j,'truth_id':t.get('id',str(j)),'accepted':False,
                'nearest_prediction_index':int(i),**pairs[i,j],
                'reason':'unmatched or endpoint/type/segment relationship failed'})
        else:
            rows.append({'truth_index':j,'truth_id':t.get('id',str(j)),'accepted':False,'reason':'no_predictions'})
    count=len(matched_t);false=n-count
    return {'truth_count':m,'prediction_count':n,'valid_welds':count,
        'valid_extraction_rate':count/m,'false_or_duplicate_predictions':false,
        'false_or_duplicate_rate_per_truth':false/m,'endpoint_limit_mm':endpoint_mm,
        'segment_endpoint_limit_mm':segment_endpoint_mm,'evaluation_version':'weld_segments_v2',
        'rate_gate_pass':count/m>.95,'false_prediction_gate_pass':false/m<=.05,
        'all_gates_pass':count/m>.95 and false/m<=.05,
        'unmatched_prediction_indices':sorted(set(range(n))-matched_p),'per_truth':rows,
        'scope':'weld endpoints, welding type, ordered segment kinds/count and every segment endpoint; arc interior errors reported separately'}
