"""Label-free conversion of frozen native predictions for complete D3 scoring.

Supports lines and arcs. Invalid geometry remains counted false output; no GT
identity, coordinate snapping, side reassignment or candidate suppression.
"""
import copy,math
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc,verify_sampled_arc

def decoded_parents(predictions):
    groups=[];current=None;previous=None
    for i,p in enumerate(predictions):
        join=current is not None and p['source_sequence']==previous['source_sequence'] and p['parent_connection_at_start'] and p['forward']==previous['forward']
        if not join:current=[];groups.append(current)
        current.append((i,p));previous=p
    parents=[];invalid=[]
    for group in groups:
        forward=group[0][1]['forward'];gaps=['SHARED' if abs(b['lo']-a['hi'])<=1e-6 else 'POSITIVE_GAP' for (_,a),(_,b) in zip(group,group[1:])]
        ordered=group if forward else list(reversed(group))
        if not forward:gaps=gaps[::-1]
        try:
            segments=[]
            for i,p in ordered:
                rep=p['representation']
                if rep['kind']=='line':segment=line_segment(*rep['endpoints_mm'])
                elif rep['kind']=='arc':segment=sample_arc(CircularArc.from_parameters(rep['arc_parameters']));verify_sampled_arc(segment)
                else:raise ValueError('Unknown predicted geometry kind')
                segments.append(dict(segment,weld_type=p['weld_type'],prediction_id=i,process_attributes=copy.deepcopy(p.get('process_attributes'))))
            parents.append(dict(child_count=len(ordered),segments=segments,gaps=gaps,is_closed=bool(len(ordered)==1 and ordered[0][1].get('closed',False)),predicted_source_sequence=group[0][1]['source_sequence'],parent_scope='within predicted source sequence; cross-sequence parent connection unresolved'))
        except (ValueError,ArithmeticError) as error:
            invalid.append(dict(counted_false_children=len(group),predicted_weld_types=[int(p['weld_type']=='vertical') for _,p in group],predicted_kinds=[{'line':1,'arc':2}.get(p['representation'].get('kind'),0) for _,p in group],reason=str(error),prediction_ids=[i for i,_ in group]))
    assert sum(p['child_count'] for p in parents)+sum(p['counted_false_children'] for p in invalid)==len(predictions)
    return dict(parents=parents,invalid_predictions=invalid,total_prediction_children=len(predictions))

def controls():
    from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
    def line(i,a,b,forward=True,connect=False,seq=0):return dict(source_sequence=seq,family=seq,parent_connection_at_start=connect,forward=forward,lo=i*10.,hi=(i+1)*10.,weld_type='horizontal',representation=dict(kind='line',endpoints_mm=[a,b]),closed=False)
    p=[line(0,[10,0,0],[0,0,0],False),line(1,[20,0,0],[10,0,0],False,True)]
    d=decoded_parents(p);assert d['parents'][0]['segments'][0]['endpoints_mm'][0]==[20.,0.,0.]
    assert d['parents'][0]['gaps']==['SHARED'] and d['parents'][0]['child_count']==2
    doubles=decoded_parents([line(0,[0,0,0],[10,0,0],seq=0),line(0,[0,0,0],[10,0,0],seq=1)]);assert len(doubles['parents'])==2
    arc=CircularArc([0,0,0],10,[0,0,1],[1,0,0],2*math.pi,True)
    pred=dict(source_sequence=2,family=2,parent_connection_at_start=False,forward=True,lo=0.,hi=arc.length_mm,weld_type='horizontal',representation=dict(kind='arc',arc_parameters=arc.to_parameters()),closed=True)
    targets=dict(original_children=1,parents=[dict(original_child_count=1,is_closed=True,gaps=[],children=[dict(original_child_id='a',original_kind='arc',weld_type='horizontal',representation=copy.deepcopy(pred['representation']))])])
    score=score_native_parents(decoded_parents([pred]),targets,[False]);assert score['categories']['overall']['TP']==1
    bad=line(0,[0,0,0],[0,0,0]);invalid=decoded_parents([bad]);assert invalid['invalid_predictions'][0]['counted_false_children']==1
    return dict(arc_full_circle_D3_scored=True,reverse_parent_order_from_predicted_direction=True,same_geometry_distinct_sides_retained=True,invalid_output_not_dropped=True,GT_reads_in_adapter=0)
