"""TRAIN-only complete-parent compatibility, using unchanged D3 primitives."""
import numpy as np
from weld_pipeline.contract_geometry_d3_409 import variants, child_error
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle


def compile_compatibility(decoded, target):
    truths=native_targets(target)['parents']
    references=[[reference_segment(c) for c in t['children']] for t in truths]
    edges=[[] for _ in decoded['parents']]
    for pi,parent in enumerate(decoded['parents']):
        views=list(variants(parent,True))
        for ti,truth in enumerate(truths):
            if parent['child_count']!=truth['original_child_count'] or parent['is_closed']!=truth['is_closed']:continue
            for view in views:
                if view['gaps']!=truth['gaps']:continue
                complete=True
                for ci,(_,segment) in enumerate(view['segments']):
                    if child_error(segment,truth['children'][ci],references[ti][ci],
                        truth['is_closed'] and len(truth['children'])==1,False) is None:
                        complete=False;break
                if complete:
                    edges[pi].append(ti);break
    oracle=ParentSetOracle(edges,len(truths))
    for i in range(len(edges)):oracle.select(i)
    matched=[i for i,v in enumerate(oracle.right_match) if v>=0]
    return dict(compatible_targets=edges,original_parents=len(truths),original_children=target['original_children'],
        maximum_matched_parents=len(matched),maximum_matched_children=sum(len(truths[i]['children']) for i in matched),
        uncovered_target_parents=[i for i in range(len(truths)) if i not in matched],
        all_parents_coverable=len(matched)==len(truths),is_extraction_result=False,
        matching_policy='Whole-parent D3 with unchanged 3mm gate; no independent child flips',
        direction_and_E16_still_unconfirmed=True)
