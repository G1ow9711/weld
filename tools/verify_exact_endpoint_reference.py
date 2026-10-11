"""Compare every compiled cost and global matching with unmodified D3."""
import argparse
import copy
import math
from pathlib import Path
import time

import numpy as np

from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.contract_curve_geometry import CircularArc, sample_arc
from weld_pipeline.server_compact_line_predictions import line_segment_array
from weld_pipeline.server_exact_endpoint_reference import compile_exact_endpoint_reference
from weld_pipeline.server_offline_parent_set_reference import compile_reference_table, evaluate_reference_subset
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents
from weld_pipeline.sjg_local_runtime_904 import write, digest

ROOT = Path(__file__).resolve().parents[1]


def line(a, b, weld='horizontal'):
    return dict(line_segment_array(a, b), weld_type=weld)


def arc(anchor=0., closed=False):
    geometry = CircularArc([0., 0., 0.], 8., [0., 0., 1.],
                           [math.cos(anchor), math.sin(anchor), 0.],
                           2.*math.pi if closed else math.pi/2., closed)
    return dict(sample_arc(geometry), weld_type='horizontal')


def parent(segments, gaps=(), closed=False):
    return dict(child_count=len(segments), segments=segments, gaps=list(gaps), is_closed=closed)


def target(parents):
    result = []
    for pi, p in enumerate(parents):
        children = []
        for ci, s in enumerate(p['segments']):
            representation = (dict(kind='line', endpoints_mm=s['endpoints_mm']) if s['kind']=='line'
                              else dict(kind='arc', arc_parameters=s['arc_parameters']))
            children.append(dict(original_child_id=f'{pi}_{ci}', original_kind=s['kind'],
                                 weld_type=s['weld_type'], representation=representation))
        result.append(dict(original_child_count=len(children), children=children,
                           is_closed=p['is_closed'], gaps=p['gaps']))
    return dict(parents=result, original_children=sum(len(p['children']) for p in result))


def decode(parents, indices):
    selected = [parents[int(i)] for i in indices]
    return dict(parents=selected, invalid_predictions=[],
                total_prediction_children=sum(p['child_count'] for p in selected))


def main(output_name):
    output = ROOT/'server_rebuild/full_dataset_v1'/output_name
    if output.exists() or Path(output_name).name != output_name:
        raise ValueError('Choose a new immutable local control file name')
    started = time.perf_counter()
    a, b = line([0,0,0],[10,0,0]), line([10,0,0],[20,0,0])
    ar, br = line([10,0,0],[0,0,0]), line([20,0,0],[10,0,0])
    wrong = line([0,20,0],[10,20,0])
    truth = target([parent([a,b], ['SHARED']), parent([a])])
    pool = [parent([a,b], ['SHARED']), parent([a,b], ['SHARED']),
            parent([wrong,b], ['SHARED']), parent([br,ar], ['SHARED']),
            parent([ar,b], ['SHARED']), parent([a,b], ['POSITIVE_GAP']),
            parent([wrong]), parent([line([0,0,0],[10,0,0], 'vertical')])]
    cases = [('partial_reversal_duplicates_gaps', pool, truth, [False,False]),
             ('specified_direction', pool, truth, [True,True]),
             ('empty_predictions', [], truth, [False,False]),
             ('empty_references', pool, target([]), [])]
    for y in [np.nextafter(3.,0.), 3., np.nextafter(3.,math.inf), 3.01]:
        cases.append((f'endpoint_boundary_{y.hex()}', [parent([line([0,y,0],[10,y,0])])], target([parent([a])]), [False]))
    for closed in [False, True]:
        cases.append((f'arc_anchor_{closed}', [parent([arc(math.pi/2.,closed)],closed=closed)],
                      target([parent([arc(0.,closed)],closed=closed)]), [False]))
    cases.append(('mixed_partial', [parent([wrong,arc()],['SHARED'])],
                  target([parent([a,arc()],['SHARED'])]), [False]))
    cases.append(('wrong_arc_kind_all_parent_rows_retained', [parent([arc()])],
                  target([parent([a])]), [False]))
    wrong_arc=dict(arc(),weld_type='vertical')
    cases.append(('mixed_wrong_arc_type_later_line_preserved',
                  [parent([wrong_arc,b],['SHARED'])],
                  target([parent([arc(),b],['SHARED'])]),[False]))
    c=line([20,0,0],[0,0,0])
    cases.append(('closed_cyclic', [parent([b,c,a],['SHARED','SHARED'],True)],
                  target([parent([a,b,c],['SHARED','SHARED'],True)]),[False]))
    rng = np.random.default_rng(1123)
    results, subset_checks = [], 0
    for name, parents, gt, directions in cases:
        original, _ = compile_reference_table(decode(parents,np.arange(len(parents))),gt,directions)
        for batch in [1,3,1024]:
            compiled, scored = compile_exact_endpoint_reference(parents,gt,decode,batch,directions)
            for key in ['costs','tp','child_counts','invalid_children']:
                np.testing.assert_array_equal(compiled[key],original[key],err_msg=f'{name}:{batch}:{key}')
            for size in sorted({0,1 if parents else 0,len(parents)}):
                ids=rng.choice(len(parents),size,replace=False)
                for order in [ids,ids[::-1]]:
                    fast=evaluate_reference_subset(compiled,order)
                    direct=score_native_parents(decode(parents,order),gt,directions)['categories']['overall']
                    assert all(fast[k]==direct[k] for k in ['predicted','TP','FP']),name
                    subset_checks+=1
        results.append(dict(case=name,parents=len(parents),reference_parents=len(gt['parents']),
                            scored_pairs=scored['original_scored_pairs'],skipped_pairs=scored['proven_unavailable_pairs']))
    assert original['costs'].shape==(1,1)
    partial=compile_exact_endpoint_reference(pool,truth,decode,3,[False,False])[0]
    assert partial['tp'][2,0]==1, 'A bad first child must not suppress a good later child'
    assert sum(r['skipped_pairs'] for r in results)>0
    bad=copy.deepcopy(pool)
    bad[0]['child_count']+=1
    try:
        compile_exact_endpoint_reference(bad,truth,decode)
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed complete ownership was accepted')
    write(output,dict(status='PASS_EVERY_ORIGINAL_COST_AND_GLOBAL_MATCHING_SYNTHETIC_CONTROL_NOT_MODEL_QUALITY',
        cases=results,subset_and_order_controls=subset_checks,all_costs_and_tp_bitwise_equal=True,
        partial_parent_later_child_preserved=True,closed_circle_anchor_not_rejected=True,
        malformed_ownership_rejected=True,all_candidates_and_reference_denominators_retained=True,
        source_sha256=digest(__file__),helper_sha256=digest(ROOT/'weld_pipeline/server_exact_endpoint_reference.py'),
        original_compiler_sha256=digest(ROOT/'weld_pipeline/server_offline_parent_set_reference.py'),
        seconds=time.perf_counter()-started,optimizer_updates=0,model_accuracy=False))
    print('EXACT_ENDPOINT_REFERENCE_CONTROLS_PASS',len(cases),subset_checks,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-name',required=True)
    main(parser.parse_args().output_name)
