"""Contract D3 development geometry: whole reversal, never child-wise flipping.

No runtime prediction is edited. Direction/side conventions remain unreviewed,
so these development metrics do not certify formal contract acceptance.
"""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from weld_pipeline.world_set_evaluation_285 import CATEGORIES,category_keys,reference_segment,distance_to_primitive
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment

VERSION='CONTRACT_D3_WHOLE_PARENT_REVERSAL_CLOSED_CONTOUR_DEVELOPMENT_v1'

def variants(parent,allow_reverse):
    original=list(enumerate(parent['segments']))
    orders=[(False,original,list(parent['gaps']))]
    if allow_reverse:orders.append((True,[(i,dict(s,**reverse_segment(s))) for i,s in original[::-1]],list(parent['gaps'])[::-1]))
    for reverse,segments,gaps in orders:
        shifts=range(len(segments)) if parent['is_closed'] and len(segments)>1 else [0]
        for shift in shifts:
            selected=segments[shift:]+segments[:shift]
            if shift:
                ring=gaps+['SHARED'];rotated=ring[shift:]+ring[:shift];selected_gaps=rotated[:-1]
            else:selected_gaps=gaps
            yield dict(reversed_whole_parent=reverse,cyclic_child_shift=shift,segments=selected,gaps=selected_gaps)

def child_error(prediction,truth,reference,closed_without_breakpoints,direction_specified):
    if prediction['weld_type']!=truth['weld_type'] or prediction['kind']!=truth['original_kind']:return None
    a=np.asarray(prediction['points'],np.float64);b=np.asarray(reference['points'],np.float64)
    if closed_without_breakpoints:
        # D3: a complete closed contour has no arbitrary endpoint anchor.
        if prediction['kind']!='arc':return None
        pa,ta=[CircularArc.from_parameters(s['arc_parameters']) for s in (prediction,reference)]
        if not pa.is_closed or not ta.is_closed:return None
        if direction_specified and float((math.copysign(1.,pa.sweep_rad)*pa.plane_normal)@
                                         (math.copysign(1.,ta.sweep_rad)*ta.plane_normal))<=0:return None
        errors=None
    else:
        errors=np.linalg.norm(a[[0,-1]]-b[[0,-1]],axis=1)
        if float(errors.max())>3.:return None
        if prediction['kind']=='arc':
            pa,ta=[CircularArc.from_parameters(s['arc_parameters']) for s in (prediction,reference)]
            pt=math.copysign(1.,pa.sweep_rad)*np.cross(pa.plane_normal,pa.start_direction)
            tt=math.copysign(1.,ta.sweep_rad)*np.cross(ta.plane_normal,ta.start_direction)
            if float(pt@tt)<=0.:return None
    whole=float(max(distance_to_primitive(a,reference).max(),distance_to_primitive(b,prediction).max()))
    if whole>3.:return None
    return dict(ordered_endpoint_errors_after_allowed_whole_correspondence_mm=None if errors is None else errors.tolist(),
        bidirectional_sampled_maximum_mm=whole,sampling_maximum_step_mm=1.,
        arbitrary_closed_anchor_not_an_endpoint_gate=closed_without_breakpoints)

def score_source(decoded,targets,direction_semantics=None):
    parents=decoded['parents'];truths=targets['parents']
    assert sum(len(p['children']) for p in truths)==targets['original_children']
    if direction_semantics is None:direction_semantics=[None]*len(truths)
    assert len(direction_semantics)==len(truths) and all(v in (None,False,True) for v in direction_semantics)
    references=[[reference_segment(c) for c in t['children']] for t in truths]
    cost=np.full((len(parents),len(truths)),1e12,np.float64);accepted={}
    for pi,parent in enumerate(parents):
        for ti,truth in enumerate(truths):
            if parent['child_count']!=truth['original_child_count'] or parent['is_closed']!=truth['is_closed']:continue
            specified=direction_semantics[ti] is True
            for view in variants(parent,not specified):
                if view['gaps']!=truth['gaps']:continue
                rows={}
                for ci,(predicted_child,segment) in enumerate(view['segments']):
                    error=child_error(segment,truth['children'][ci],references[ti][ci],
                        truth['is_closed'] and len(truth['children'])==1,specified)
                    if error is not None:rows[ci]=dict(predicted_child_index=predicted_child,
                        reversed_whole_parent=view['reversed_whole_parent'],cyclic_child_shift=view['cyclic_child_shift'],**error)
                if not rows:continue
                value=-1e6*len(rows)+sum(sum(row['ordered_endpoint_errors_after_allowed_whole_correspondence_mm'] or [])+
                    row['bidirectional_sampled_maximum_mm'] for row in rows.values())
                if value<cost[pi,ti]:cost[pi,ti]=value;accepted[(pi,ti)]=rows
    matches=[]
    if cost.size:
        pp,tt=linear_sum_assignment(cost)
        for pi,ti in zip(pp.tolist(),tt.tolist()):
            for ci,error in accepted.get((pi,ti),{}).items():
                matches.append(dict(predicted_parent_index=pi,original_parent_index=ti,child_index=ci,
                    original_child_id=truths[ti]['children'][ci]['original_child_id'],**error))
    categories={key:dict(G=0,predicted=0,TP=0,FP=0) for key in CATEGORIES}
    for truth in truths:
        for child in truth['children']:
            for key in category_keys(child['weld_type'],child['original_kind']):categories[key]['G']+=1
    for parent in parents:
        for child in parent['segments']:
            for key in category_keys(child['weld_type'],child['kind']):categories[key]['predicted']+=1
    for invalid in decoded['invalid_predictions']:
        for ci in range(invalid['counted_false_children']):
            weld=invalid['predicted_weld_types'][min(ci,len(invalid['predicted_weld_types'])-1)]
            kind=invalid['predicted_kinds'][min(ci,len(invalid['predicted_kinds'])-1)]
            keys=category_keys('horizontal' if weld==0 else 'vertical','line' if kind==1 else 'arc') if kind in (1,2) else ('overall','horizontal' if weld==0 else 'vertical')
            for key in keys:categories[key]['predicted']+=1
    for match in matches:
        child=truths[match['original_parent_index']]['children'][match['child_index']]
        for key in category_keys(child['weld_type'],child['original_kind']):categories[key]['TP']+=1
    for value in categories.values():value['FP']=value['predicted']-value['TP']
    assert categories['overall']['predicted']==decoded['total_prediction_children']
    return dict(version=VERSION,categories=categories,matches=matches,original_children=targets['original_children'],
        original_parents=len(truths),allowed_operation='ONE_WHOLE_PARENT_REVERSAL_AND_CLOSED_CYCLIC_CORRESPONDENCE_D3',
        independent_child_reversal_forbidden=True,predictions_or_GT_coordinates_changed=False,
        direction_convention_review='UNCONFIRMED_DEVELOPMENT_ALLOWANCE' if any(v is None for v in direction_semantics) else 'EXPLICIT_SUPPLIED_POLICY',
        all_original_denominators_retained=True,formal_signed_side_and_E16_acceptance=False,full_contract_pass=False)

def synchronized_endpoint_fields(fields,reversed_parent):
    result=dict(fields)
    if reversed_parent:
        for a,z in [('front_wall','back_wall'),('start_height','end_height'),('start_hole_height','end_hole_height'),('start_base_edge','end_base_edge')]:
            result[a],result[z]=fields[z],fields[a]
    # Fixed physical plate slots and signs are retained; unreviewed side/normal
    # conventions cannot gain formal approval through this correspondence.
    return result
