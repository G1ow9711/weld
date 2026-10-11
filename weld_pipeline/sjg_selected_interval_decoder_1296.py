"""Geometry decoding for upstream-selected intervals, without a second score gate.

Original geometry implementation is preserved verbatim apart from admission
validation and its receipt. Historical decoder and probability fields stay intact.
"""
import math
import numpy as np
from weld_pipeline.sjg_parent_set_runtime_1081 import child_geometry,line_segment,fit_ordered_circle,sample_arc,verify_sampled_arc

def manifest_choices(manifest):
    if manifest.get('schema')!='FROZEN_UNIFIED_DP_SELECTION_V1':raise ValueError('Missing selected manifest')
    if not all(manifest.get(k) for k in ('model_sha256','input_binding','DP_implementation_sha256','selection_sha256')):
        raise ValueError('Unbound selected manifest')
    entries=manifest['choices'];keys=[v['query'] for v in entries]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate manifest query')
    return dict(zip(keys,entries))

def decode_selected_choices(choices,bank,selected_manifest):
    """All positive query objects are emitted or counted invalid; no GT count."""
    parents=[];invalid=[];declared=0;permitted=manifest_choices(selected_manifest)
    for parent in choices:
        if parent['query'] not in permitted or parent!=permitted[parent['query']]:
            raise ValueError('Choice identity or fields differ from sealed DP selection')
        score=float(parent['unified_score'])
        if not math.isfinite(score) or score<=0:raise ValueError('Expected a finite positive upstream unified score')
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
                presence_threshold=None,admission_rule='upstream_unified_score_strict_positive',GT_reads=0,NMS=False,output_union=False,GT_count_used=False)

def decode_selected_shared(choices,bank,selected_manifest):
    permitted=manifest_choices(selected_manifest)
    if {v['query'] for v in choices}!=set(permitted):raise ValueError('Missing or extra selected query')
    result=dict(parents=[],invalid_predictions=[],total_prediction_children=0,
        presence_threshold=None,admission_rule='upstream_unified_score_strict_positive',
        GT_reads=0,STEP_reads=0,NMS=False,output_union=False,GT_count_used=False)
    cache={};queries=[]
    for choice in choices:
        if choice!=permitted[choice['query']]:raise ValueError('Changed selected query fields')
        if not math.isfinite(float(choice['unified_score'])) or choice['unified_score']<=0:
            raise ValueError('Expected a finite positive upstream unified score')
        if choice['child_count']!=1 or len(choice['children'])!=1:
            raise ValueError('Shared interval decoder requires a single child')
        queries.append(choice['query']);c=choice['children'][0]
        key=(c['profile'],*c['intervals'],*c['fractions'],c['closed'],c.get('wrap',False),c['forward'],c['weld_type'])
        if key not in cache:cache[key]=decode_selected_choices([choice],bank,selected_manifest)
        base=cache[key]
        for parent in base['parents']:
            p=dict(parent,predicted_parent_query=choice['query'])
            p['segments']=[dict(parent['segments'][0],prediction_id=f"{choice['query']}:0")]
            result['parents'].append(p)
        for invalid in base['invalid_predictions']:
            result['invalid_predictions'].append(dict(invalid,parent_query=choice['query']))
        result['total_prediction_children']+=choice['child_count']
    actual=[p['predicted_parent_query'] for p in result['parents']]+[p['parent_query'] for p in result['invalid_predictions']]
    assert len(set(queries))==len(queries) and sorted(actual)==sorted(queries)
    assert sum(p['child_count'] for p in result['parents'])+sum(p['counted_false_children'] for p in result['invalid_predictions'])==result['total_prediction_children']
    return result
