"""TRAIN-only child validity, type and oriented parent-continuation labels.

This module consumes targets only to create labels. Native continuation input
features live separately in 1004 and never call this code at inference.
"""
import numpy as np
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier


def training_targets(children,pairs,target):
    if target.get('coordinate_frame')!='workpiece_local' or target.get('unit')!='mm':
        raise ValueError('Native workpiece-local millimetre targets required')
    curves=[CurveCarrier(c['representation']) for c in children]
    ends=np.asarray([c.at(np.array([0.,c.length])) for c in curves]).reshape(-1,2,3)
    kinds=np.array([c.kind for c in curves]);matches=[[] for _ in children]
    for pi,parent in enumerate(target['parents']):
        for ci,child in enumerate(parent['children']):
            truth=CurveCarrier(child['representation']);boundary=truth.at(np.array([0.,truth.length]))
            direct=np.linalg.norm(ends-boundary,axis=2).max(1)
            reverse=np.linalg.norm(ends[:,::-1]-boundary,axis=2).max(1)
            closed=truth.kind=='arc' and truth.arc.is_closed
            if closed:
                possible=[j for j,c in enumerate(curves) if c.kind=='arc' and c.arc.is_closed
                    and np.linalg.norm(c.arc.center_mm-truth.arc.center_mm)<=3.
                    and abs(c.arc.radius_mm-truth.arc.radius_mm)<=3.]
            else:possible=np.flatnonzero((kinds==truth.kind)&(np.minimum(direct,reverse)<=3.))
            samples=truth.at(truth.grid(1.))
            for j in possible:
                j=int(j);prediction=curves[j]
                error=max(float(prediction.project(samples)[1].max()),float(truth.project(prediction.at(prediction.grid(1.)))[1].max()))
                if error>3.:continue
                if closed:
                    same=bool((prediction.arc.plane_normal*np.sign(prediction.arc.sweep_rad))@
                        (truth.arc.plane_normal*np.sign(truth.arc.sweep_rad))>0.)
                else:same=bool(direct[j]<=reverse[j])
                matches[j].append(dict(parent=pi,child=ci,same_direction=same,
                    original_child_id=child['original_child_id'],weld_type=child['weld_type']))
    valid=np.asarray([bool(v) for v in matches],np.float32)
    types=np.zeros(len(children),np.int64);known=np.zeros(len(children),bool)
    for i,rows in enumerate(matches):
        names={r['weld_type'] for r in rows}
        if len(names)==1:
            name=next(iter(names))
            if name not in ('horizontal','vertical'):raise ValueError('Unsupported known weld type '+str(name))
            types[i]=int(name=='vertical');known[i]=True
    link=np.zeros((len(pairs),2),np.float32)
    for k,(i,ri,j,rj) in enumerate(pairs):
        for a in matches[int(i)]:
            for b in matches[int(j)]:
                if a['parent']!=b['parent']:continue
                forward_a=a['same_direction']!=bool(ri);forward_b=b['same_direction']!=bool(rj)
                if forward_a!=forward_b:continue
                parent=target['parents'][a['parent']];count=len(parent['children']);ci,cj=a['child'],b['child']
                adjacent=cj==ci+1 if forward_a else cj==ci-1
                closure=parent['is_closed'] and ((ci==count-1 and cj==0) if forward_a else (ci==0 and cj==count-1))
                if adjacent or closure:
                    link[k,0]=1.
                    if closure:link[k,1]=1.
                    else:
                        gap=parent['gaps'][min(ci,cj)]
                        if gap not in ('SHARED','POSITIVE_GAP'):raise ValueError('Unknown gap label')
                        if gap=='SHARED':link[k,1]=1.
    return dict(valid=valid,weld_type=types,weld_type_known=known,link=link),matches


def controls():
    from weld_pipeline.sjg_native_continuation_1004 import continuation_inputs
    from weld_pipeline.contract_curve_geometry import CircularArc
    def line(a,b):return dict(kind='line',endpoints_mm=[a,b])
    representations=[line([0,0,0],[10,0,0]),line([10,0,0],[20,0,0]),line([50,0,0],[60,0,0])]
    target=dict(coordinate_frame='workpiece_local',unit='mm',parents=[dict(is_closed=False,gaps=['SHARED','POSITIVE_GAP'],
        children=[dict(representation=r,original_child_id='p:'+str(i),weld_type='horizontal') for i,r in enumerate(representations)])])
    children=[dict(representation=r,profile=0) for r in representations]+[dict(representation=line([0,30,0],[10,30,0]),profile=0)]
    pairs,_=continuation_inputs(children,[dict(face_indices=[0,1])]);labels,matches=training_targets(children,pairs,target)
    lookup={tuple(p):labels['link'][i] for i,p in enumerate(pairs)}
    assert np.array_equal(labels['valid'],[1,1,1,0]) and np.array_equal(labels['weld_type_known'],[True,True,True,False])
    assert np.array_equal(lookup[0,0,1,0],[1,1]) and np.array_equal(lookup[1,0,2,0],[1,0])
    assert np.array_equal(lookup[1,1,0,1],[1,1])
    a=CircularArc([0,0,0],10,[0,0,1],[1,0,0],2*np.pi,True)
    b=CircularArc([0,0,0],10,[0,0,1],[0,1,0],2*np.pi,True)
    truth=dict(coordinate_frame='workpiece_local',unit='mm',parents=[dict(is_closed=True,gaps=[],children=[
        dict(representation=dict(kind='arc',arc_parameters=a.to_parameters()),original_child_id='circle:0',weld_type='horizontal')])])
    labels,_=training_targets([dict(representation=dict(kind='arc',arc_parameters=b.to_parameters()),profile=0)],np.empty((0,4),np.int64),truth)
    assert labels['valid'][0]==1
    return dict(native_local_target_required=True,original_parent_child_identities_labels_only=True,
        shared_and_positive_gaps_distinguished=True,whole_parent_reverse_labels_consistent=True,
        background_type_not_invented=True,closed_contour_no_arbitrary_endpoint_anchor=True,optimizer_updates=0)
