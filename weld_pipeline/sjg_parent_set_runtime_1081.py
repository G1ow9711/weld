"""Target-free full parent-query decoding with one authoritative shared node."""
import math
import numpy as np
from weld_pipeline.contract_curve_geometry import CircularArc,fit_ordered_circle,sample_arc,verify_sampled_arc
from weld_pipeline.world_set_runtime_285 import line_segment


def child_geometry(curve,lo,hi,forward,closed=False,wrap=False):
    lo,hi=sorted((float(lo),float(hi)))
    if not 0<=lo<=hi<=curve.length:raise ValueError('Predicted child outside finite profile')
    if curve.kind=='line':
        if closed or wrap or not hi>lo:raise ValueError('Invalid line closure/wrap/extent')
        q=np.asarray([lo,(lo+hi)/2,hi])
        if not forward:q=q[::-1]
        return curve.at(q),None
    original=curve.arc
    if closed:
        if not original.is_closed:raise ValueError('Closed child requires complete circular input carrier')
        start=lo;length=curve.length
    elif wrap:
        if not original.is_closed or not hi>lo:raise ValueError('Wrapped child requires complete circular input carrier')
        start=hi;length=curve.length-hi+lo
    else:
        if not hi>lo:raise ValueError('Degenerate arc interval')
        start=lo;length=hi-lo
    direction=math.copysign(1.,original.sweep_rad)
    start_point=curve.at(start)
    arc=CircularArc(original.center_mm,original.radius_mm,original.plane_normal,
                    (start_point-original.center_mm)/original.radius_mm,
                    direction*length/original.radius_mm,bool(closed))
    if not forward:
        arc=CircularArc(arc.center_mm,arc.radius_mm,arc.plane_normal,
                        (arc.point_at(arc.length_mm)-arc.center_mm)/arc.radius_mm,-arc.sweep_rad,arc.is_closed)
    return arc.point_at(np.asarray([0.,arc.length_mm/2,arc.length_mm])),arc


def decode_choices(choices,bank):
    """All positive query objects are emitted or counted invalid; no GT count."""
    parents=[];invalid=[];declared=0
    for parent in choices:
        if float(parent['probability'])<.5:continue
        children=parent['children'];count=int(parent['child_count']);declared+=count
        types=[int(c.get('weld_type',0)) for c in children]
        kinds=[]
        for c in children:
            try:kinds.append(1 if bank.get(c['profile'])['curve'].kind=='line' else 2)
            except (ValueError,IndexError,KeyError):kinds.append(0)
        try:
            # The neural head has five slots; analytic path decoding itself has
            # no five-child limit, so the finite graph can reuse this geometry.
            if count<1 or len(children)!=count:raise ValueError('Invalid declared child count')
            nodes=[];segments=[];links=[];gaps=[];previous_end=None
            for index,c in enumerate(children):
                item=bank.get(c['profile']);curve=item['curve']
                lo=bank.coordinate(c['profile'],c['intervals'][0],c['fractions'][0])
                hi=bank.coordinate(c['profile'],c['intervals'][1],c['fractions'][1])
                points,arc=child_geometry(curve,lo,hi,bool(c['forward']),bool(c['closed']),bool(c.get('wrap',False)))
                connection=c.get('connection','UNKNOWN')
                if index and connection=='SHARED':
                    start_id=previous_end
                    points[0]=nodes[start_id]
                else:
                    start_id=len(nodes);nodes.append(points[0].tolist())
                if index:
                    if connection not in ('SHARED','POSITIVE_GAP','UNKNOWN'):raise ValueError('Invalid connection state')
                    gaps.append(connection)
                    if connection=='POSITIVE_GAP' and np.array_equal(nodes[previous_end],nodes[start_id]):
                        raise ValueError('Positive gap declared at coincident nodes')
                if parent['closed'] and index==count-1:
                    end_id=links[0][0] if links else start_id;points[-1]=nodes[end_id]
                elif c['closed']:
                    if count!=1 or not parent['closed']:raise ValueError('Inconsistent parent/child closure')
                    end_id=start_id;points[-1]=nodes[start_id]
                else:
                    end_id=len(nodes);nodes.append(points[-1].tolist())
                if curve.kind=='line':
                    segment=line_segment(points[0],points[-1])
                else:
                    original_ends=arc.point_at(np.asarray([0.,arc.length_mm]))
                    if not np.array_equal(points[[0,-1]],original_ends):
                        if arc.is_closed:raise ValueError('Cannot distort a closed circle to attach a different node')
                        # Three predicted control points define an analytic circle.
                        # The shared start is owned by the preceding child only.
                        arc=fit_ordered_circle(points,tolerance_mm=1e-6,is_closed=False)
                    segment=sample_arc(arc)
                    segment['points'][0]=list(nodes[start_id]);segment['points'][-1]=list(nodes[end_id])
                    verify_sampled_arc(segment)
                wt=int(c['weld_type'])
                if wt not in (0,1):raise ValueError('Unknown predicted weld category')
                segment.update(weld_type=('horizontal','vertical')[wt],profile=int(c['profile']),
                    relation=item['relation'],side_hypothesis=item.get('side_hypothesis'),
                    predicted_side=c.get('side','UNKNOWN'),material_identity='UNKNOWN',process_attributes=None,
                    prediction_id=f"{parent['query']}:{index}",node_ids=[start_id,end_id])
                segments.append(segment);links.append([start_id,end_id]);previous_end=end_id
            parents.append(dict(child_count=count,segments=segments,gaps=gaps,is_closed=bool(parent['closed']),
                predicted_parent_query=int(parent['query']),nodes_mm=nodes,child_node_ids=links,
                parent_scope='one complete predicted query; children may select distinct profiles and relations'))
        except (ValueError,ArithmeticError,IndexError) as error:
            invalid.append(dict(counted_false_children=count,predicted_weld_types=types or [0],
                predicted_kinds=kinds or [0],reason=str(error),parent_query=int(parent['query'])))
    assert sum(p['child_count'] for p in parents)+sum(p['counted_false_children'] for p in invalid)==declared
    return dict(parents=parents,invalid_predictions=invalid,total_prediction_children=declared,
                presence_threshold=.5,GT_reads=0,NMS=False,output_union=False,GT_count_used=False)


def network_choices(model,result,bank,chunk=4096):
    import torch
    probability=result['existence'].sigmoid().detach().cpu().numpy()
    selected=np.flatnonzero(probability>=.5)
    choices=[]
    for query in selected:
        count=int(result['count'][query].argmax())+1
        vectors=result['profile_query'][query,:count]
        best=vectors.new_full((count,),-float('inf'));profile_ids=torch.zeros(count,dtype=torch.long,device=vectors.device)
        # Complete categorical argmax per child. This is not candidate TopK pruning.
        for first in range(0,len(bank),chunk):
            logits=vectors@result['profile_key'][first:first+chunk].T/math.sqrt(model.width)
            value,index=logits.max(-1);improved=value>best
            profile_ids=torch.where(improved,index+first,profile_ids);best=torch.maximum(best,value)
        children=[]
        for child,pid in enumerate(profile_ids.tolist()):
            item=bank.get(pid)
            features=torch.as_tensor(item['interval_features'],device=vectors.device)
            logits=model.endpoint_logits(result['endpoint_query'][query,child],features)
            intervals=logits.argmax(-1).tolist()
            children.append(dict(profile=pid,intervals=intervals,fractions=result['fraction'][query,child].tolist(),
                forward=bool(result['direction'][query,child]>=0),closed=bool(result['closed'][query,child]>=0),
                wrap=bool(result['wrap'][query,child]>=0),weld_type=int(result['type'][query,child].argmax()),
                side=('UNKNOWN','A','B')[int(result['side'][query,child].argmax())],
                connection=('UNKNOWN','SHARED','POSITIVE_GAP')[int(result['connection'][query,child].argmax())]))
        choices.append(dict(query=int(query),probability=float(probability[query]),child_count=count,
                            closed=bool(result['parent_closed'][query]>=0),children=children))
    return choices
