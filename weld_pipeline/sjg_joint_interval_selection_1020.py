"""Select mutually compatible curve hypotheses within one observed joint.

Only identical face associations and numerically coincident infinite curves
compete. Confidence is the existing network score. Coordinates, traversal,
gaps and child representations are never fitted to reference labels here.
"""
from bisect import bisect_right
import numpy as np
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier


def linear_schedule(rows):
    """Maximum summed confidence on nonoverlapping open interval interiors."""
    ordered = sorted(rows, key=lambda r:(r['hi'],r['lo'],r['index']))
    ends = [row['hi'] for row in ordered];value=[0.];paths=[()]
    for i,row in enumerate(ordered):
        previous = bisect_right(ends,row['lo']+1e-9,0,i)
        take = value[previous]+row['weight'];path=paths[previous]+(row['index'],)
        if take>value[-1]+1e-12:
            value.append(take);paths.append(path)
        else:
            value.append(value[-1]);paths.append(paths[-1])
    return value[-1],list(paths[-1])


def circular_schedule(rows, circumference):
    # At most one selected interval crosses the fixed angular origin.
    plain=[r for r in rows if not r['wrap'] and not r['whole']]
    best,chosen=linear_schedule(plain)
    for crossing in rows:
        if not crossing['wrap'] and not crossing['whole']:continue
        if crossing['whole']:
            score=crossing['weight'];ids=[crossing['index']]
        else:
            compatible=[r for r in plain if r['lo']>=crossing['hi']-1e-9 and r['hi']<=crossing['lo']+1e-9]
            score,ids=linear_schedule(compatible);score+=crossing['weight'];ids.append(crossing['index'])
        if score>best+1e-12:best,chosen=score,ids
    return best,chosen


def same_infinite_curve(a,b,tolerance=1e-3):
    if a.kind!=b.kind:return False
    if a.kind=='line':
        delta=b.ends-a.ends[0]
        perpendicular=delta-(delta@a.direction)[:,None]*a.direction
        return bool(np.linalg.norm(perpendicular,axis=1).max()<=tolerance and abs(a.direction@b.direction)>1-1e-8)
    return bool(np.linalg.norm(a.arc.center_mm-b.arc.center_mm)<=tolerance
        and abs(a.arc.radius_mm-b.arc.radius_mm)<=tolerance
        and abs(a.arc.plane_normal@b.arc.plane_normal)>1-1e-8)


def select(predictions,profiles,endpoint_tolerance_mm=3.):
    if endpoint_tolerance_mm<0:raise ValueError('Nonnegative endpoint tolerance required')
    curves=[CurveCarrier(p['representation']) for p in predictions]
    groups={};isolated=[]
    for i,prediction in enumerate(predictions):
        profile=profiles[int(prediction['profile'])]
        faces=tuple(sorted(set(profile['face_indices'])))
        # A single-face boundary lacks a resolved pair. Only hypotheses on the
        # exact same input carrier compete; other unresolved contacts survive.
        key=('faces',faces) if len(faces)==2 else ('carrier',int(prediction['profile']))
        if not faces:
            isolated.append(i);continue
        groups.setdefault(key,[]).append(i)
    kept=list(isolated);decisions=[]
    for key,indices in groups.items():
        families=[]
        for i in indices:
            for family in families:
                if same_infinite_curve(curves[family[0]],curves[i]):family.append(i);break
            else:families.append([i])
        for family in families:
            if len(family)==1:kept.extend(family);continue
            reference=curves[family[0]];rows=[]
            for i in family:
                curve=curves[i];weight=float(predictions[i]['confidence'])
                if not np.isfinite(weight) or not 0<=weight<=1:raise ValueError('Invalid prediction confidence')
                # Shrinking each end by its allowed error preserves two
                # adjacent children whose independently predicted ends overlap
                # by up to twice the contract endpoint tolerance.
                trim=min(endpoint_tolerance_mm,max(0.,curve.length/2-1e-6))
                if curve.kind=='line':
                    endpoints=(curve.ends-reference.ends[0])@reference.direction
                    lo,hi=float(endpoints.min()+trim),float(endpoints.max()-trim)
                    rows.append(dict(index=i,lo=lo,hi=hi,weight=weight))
                else:
                    a=reference.arc;second=np.cross(a.plane_normal,a.start_direction)
                    radial=curve.at(0)-a.center_mm
                    phase=float(np.mod(np.arctan2(radial@second,radial@a.start_direction),2*np.pi)*a.radius_mm)
                    circumference=2*np.pi*a.radius_mm
                    forward=curve.arc.plane_normal@a.plane_normal*np.sign(curve.arc.sweep_rad)>0
                    if curve.arc.is_closed:
                        rows.append(dict(index=i,lo=0.,hi=circumference,weight=weight,wrap=False,whole=True))
                    else:
                        lo=(phase if forward else phase-curve.length)+trim
                        lo%=circumference;hi=lo+curve.length-2*trim
                        rows.append(dict(index=i,lo=lo,hi=hi%circumference if hi>circumference else hi,
                                         weight=weight,wrap=hi>circumference,whole=False))
            if reference.kind=='line':score,chosen=linear_schedule(rows)
            else:score,chosen=circular_schedule(rows,2*np.pi*reference.arc.radius_mm)
            kept.extend(chosen)
            decisions.append(dict(joint_key=key,kind=reference.kind,candidates=family,kept=chosen,
                suppressed=[i for i in family if i not in chosen],summed_confidence=score))
    kept=sorted(kept)
    return [dict(predictions[i],before_joint_selection_index=i) for i in kept],dict(
        input_predictions=len(predictions),output_predictions=len(kept),decisions=decisions,
        infinite_curve_coincidence_tolerance_mm=1e-3,endpoint_tolerance_mm=endpoint_tolerance_mm,
        target_reads=0,STEP_reads=0,coordinates_and_traversal_unchanged=True,
        predicted_face_associations_not_customer_material_truth=True,parent_structure_still_pending=True)


def controls():
    from weld_pipeline.contract_curve_geometry import CircularArc
    profiles=[dict(face_indices=[0,1]),dict(face_indices=[1,0]),dict(face_indices=[2,3])]
    def line(lo,hi,confidence,profile=0,y=0):
        return dict(profile=profile,confidence=confidence,representation=dict(kind='line',endpoints_mm=[[lo,y,0],[hi,y,0]]))
    predictions=[line(0,100,.9),line(0,50,.6),line(50,100,.6),line(0,100,.8,2),line(0,100,.7,0,1)]
    output,audit=select(predictions,profiles)
    assert {p['before_joint_selection_index'] for p in output}=={1,2,3,4}
    # Same physical line, reversed carrier orientation and differing lengths.
    output,_=select([line(0,100,.8),line(-20,100,.7,1)],profiles)
    assert len(output)==1 and output[0]['confidence']==.8
    adjacent=[line(0,53,.8),line(47,100,.7)]
    assert len(select(adjacent,profiles)[0])==2
    assert len(select([line(0,40,.8),line(60,100,.7)],profiles)[0])==2
    # Circular alternatives across the angular origin preserve disjoint arcs.
    def arc(start,sweep,confidence):
        a=CircularArc([0,0,0],100,[0,0,1],[np.cos(start),np.sin(start),0],sweep,abs(abs(sweep)-2*np.pi)<1e-8)
        return dict(profile=0,confidence=confidence,representation=dict(kind='arc',arc_parameters=a.to_parameters()))
    circle=[arc(0,2*np.pi,.9),arc(1.75*np.pi,.5*np.pi,.6),arc(.5*np.pi,.5*np.pi,.6)]
    result,_=select(circle,profiles)
    assert {p['before_joint_selection_index'] for p in result}=={1,2}
    reverse=arc(0,-np.pi/2,.8);same=arc(1.5*np.pi,np.pi/2,.7)
    result,_=select([reverse,same],profiles)
    assert len(result)==1 and result[0]['representation']==reverse['representation']
    assert select([],profiles)[0]==[]
    # Exhaustive verification against all subsets for a small line schedule.
    rows=[dict(index=i,lo=a,hi=b,weight=w) for i,(a,b,w) in enumerate([(0,6,.7),(0,2,.4),(2,4,.4),(4,6,.4),(3,8,.8)])]
    result,chosen=linear_schedule(rows);best=0.
    for mask in range(1<<len(rows)):
        part=[row for i,row in enumerate(rows) if mask&(1<<i)]
        if all(a['hi']<=b['lo'] or b['hi']<=a['lo'] for i,a in enumerate(part) for b in part[i+1:]):
            best=max(best,sum(r['weight'] for r in part))
    assert abs(result-best)<1e-12
    return dict(global_schedule_matches_exhaustive_search=True,adjacent_children_and_positive_gaps_preserved=True,
        different_faces_and_parallel_offset_lines_preserved=True,closed_and_wrapped_arcs_supported=True,
        direction_and_coordinates_preserved=True,confidence_decides_alternatives=True,optimizer_updates=0)
