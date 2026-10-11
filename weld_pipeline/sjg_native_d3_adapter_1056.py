"""Evaluation-only adapter: native coordinates into the existing parent D3 scorer.

The legacy field name world_representation is populated by a deep copy of the
native representation. No coordinate transform is applied or inferred. Runtime
prediction modules must not import this target-reading adapter.
"""
import copy
import numpy as np
from weld_pipeline.contract_geometry_d3_409 import score_source
from weld_pipeline.world_set_runtime_285 import line_segment

def native_targets(target):
    result=copy.deepcopy(target)
    for p in result['parents']:
        for c in p['children']:
            if 'representation' not in c:raise ValueError('Explicit native representation required')
            c['world_representation']=copy.deepcopy(c['representation'])
    return result

def score_native_parents(decoded,target,direction_semantics=None):
    result=score_source(decoded,native_targets(target),direction_semantics)
    result.update(coordinate_adapter='legacy field alias only; original native XYZ unchanged',
                  native_coordinate_transform_applied=False,full_contract_pass=False)
    return result

def controls():
    def segment(a,b):return dict(line_segment(a,b),weld_type='horizontal')
    a=segment([0,0,0],[20,0,0]);b=segment([20,0,0],[20,30,0])
    target=dict(original_children=2,parents=[dict(original_child_count=2,is_closed=False,gaps=['SHARED'],children=[
        dict(original_child_id=str(i),original_kind='line',weld_type='horizontal',
             representation=dict(kind='line',endpoints_mm=s['endpoints_mm'])) for i,s in enumerate((a,b))])])
    def decoded(segments,gaps=['SHARED']):return dict(parents=[dict(child_count=len(segments),is_closed=False,gaps=gaps,segments=segments)],
        invalid_predictions=[],total_prediction_children=len(segments))
    good=score_native_parents(decoded([a,b]),target,[False]);assert good['categories']['overall']['TP']==2
    ar=segment(a['endpoints_mm'][1],a['endpoints_mm'][0]);br=segment(b['endpoints_mm'][1],b['endpoints_mm'][0])
    whole=score_native_parents(decoded([br,ar]),target,[False]);assert whole['categories']['overall']['TP']==2
    assert all(m['reversed_whole_parent'] for m in whole['matches'])
    partial=score_native_parents(decoded([ar,b]),target,[False]);assert partial['categories']['overall']['TP']<2
    fixed=score_native_parents(decoded([br,ar]),target,[True]);assert fixed['categories']['overall']['TP']<2
    wronggap=score_native_parents(decoded([a,b],['POSITIVE_GAP']),target,[False]);assert wronggap['categories']['overall']['TP']==0
    assert all('world_representation' not in c for p in target['parents'] for c in p['children'])
    bad=segment([0,3.01,0],[20,3.01,0]);limit=score_native_parents(decoded([bad,b]),target,[False]);assert limit['categories']['overall']['TP']<2
    return dict(native_input_not_mutated=True,no_coordinate_transform=True,legal_whole_parent_reversal=True,
        independent_child_flip_not_accepted_as_all_TP=True,specified_direction_respected=True,
        gap_identity_preserved=True,above_3mm_not_accepted=True,formal_side_and_E16_not_certified=True,
        optimizer_updates=0)
