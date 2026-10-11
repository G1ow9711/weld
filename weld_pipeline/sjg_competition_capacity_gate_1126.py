"""Freeze input request resources, then substitute unchanged TRAIN witnesses."""
import gc
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, check_resources
from weld_pipeline.sjg_joint_forest_train_1109 import load_case
from weld_pipeline.sjg_request_competition_1125 import freeze_requests, competition_resources, controls

KEYS = ('21', '4', '10', '25', '36', '42', 'BM79A_native_local')


def solver_controls():
    from scipy.optimize import milp  # Warm library import outside solver timing.
    from weld_pipeline.sjg_joint_link_runtime_1107 import select_paths
    nodes = [dict(probability=.9, fragment=i, side=0) for i in range(2)]
    links = np.empty((0, 2), np.int32)
    p = np.empty((0, 4)); legal = np.empty((0, 3), bool)
    one = select_paths(nodes, links, p, legal, competition_resources=[dict(nodes=[0, 1])], allow_node_selection=True)
    assert len(one['selected_nodes']) == 1 and one['unselected_explanations'] == 1
    two = select_paths(nodes, links, p, legal, competition_resources=[], allow_node_selection=True)
    assert set(two['selected_nodes']) == {0, 1}
    # Legacy caller still emits both admitted nodes, preserving old frozen tests.
    legacy = select_paths(nodes, links, p, legal)
    assert set(legacy['selected_nodes']) == {0, 1}
    instances=[dict(nodes[0],k=1),dict(nodes[0],k=2)]
    disjoint=select_paths(instances,links,p,legal,competition_resources=[],allow_node_selection=True)
    assert set(disjoint['selected_nodes'])=={0,1}
    overlapping=select_paths(instances,links,p,legal,competition_resources=[dict(nodes=[0,1])],allow_node_selection=True)
    assert len(overlapping['selected_nodes'])==1
    return dict(one_request_two_explanations_one_output=True,
                independent_resource_two_outputs=True, legacy_default_unchanged=True,
                same_fragment_disjoint_ordinal_instances_allowed=True,
                same_fragment_overlapping_instances_compete=True,
                one=one, two=two)


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run311');ap.add_argument('--reuse-input-run')
    args=ap.parse_args()
    start = time.perf_counter(); dest = OUT/args.run; dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json', dict(status='REGISTERED_BEFORE_EXECUTION',
        training_keys=list(KEYS), DEV='64', original_G=363,
        new_capacity_searches=0, new_fit_calls=0, input_only_freeze_before_GT=True,
        sources_sha256={name:digest(ROOT/'weld_pipeline'/name) for name in
                       ('sjg_request_competition_1125.py', 'sjg_competition_capacity_gate_1126.py', 'sjg_joint_link_runtime_1107.py')}))
    write(dest/'synthetic_controls.json', dict(intervals=controls(), solver=solver_controls()))
    bindings = []
    for key in KEYS+('64',):
        check_resources()
        if args.reuse_input_run:
            source = OUT/args.reuse_input_run/f'source_{key}_input_requests.json'
            registered=next(r for r in read(OUT/args.reuse_input_run/'all_input_requests_frozen_before_labels.json')['cases'] if r['source']==key)
            assert digest(source)==registered['sha256']
            frozen=read(source)
        else:
            case = load_case(key, supervision=False)
            frozen = freeze_requests(key, case['graph'])
            del case
        path = dest/f'source_{key}_input_requests.json'; write(path, frozen)
        bindings.append(dict(source=key, requests=len(frozen['requests']), path=path.name, sha256=digest(path)))
        print('INPUT_COMPETITION_REQUESTS_FROZEN', key, len(frozen['requests']), flush=True)
        gc.collect()
    write(dest/'all_input_requests_frozen_before_labels.json', dict(cases=bindings, GT_reads=0))
    pair_proofs = []
    for pair in read(OUT/'run309/conflict_provenance.json')['rows']:
        frozen = read(dest/f"source_{pair['source']}_input_requests.json")
        matches = [i for i, r in enumerate(frozen['requests']) if
                   set(r['members']) == {pair['negative_profile'], pair['positive_profile']}]
        pair_proofs.append(dict(source=pair['source'], profiles=[pair['negative_profile'], pair['positive_profile']],
                               matching_input_requests=matches, proven=len(matches) == 1))
    write(dest/'six_pairs_request_proof.json', pair_proofs)
    assert all(p['proven'] for p in pair_proofs), 'At least one pair lacks a single generation request'
    audits = []
    for key in KEYS:
        check_resources(); case = load_case(key, supervision=True)
        nodes = [case['teacher'][k] for k in sorted(case['teacher'])]
        frozen = read(dest/f'source_{key}_input_requests.json')
        uncertainty=[]
        resources = competition_resources(nodes, case['graph'], frozen,uncertainty)
        write(dest/f'source_{key}_arithmetic_enclosures.json', uncertainty)
        collisions = []
        for r in resources:
            collisions.append(dict(**r, teacher_object_rows=[nodes[i]['object_row'] for i in r['nodes']],
                                   profiles=[case['graph'].profile(nodes[i]['fragment']) for i in r['nodes']]))
        audit = dict(source=key, G=case['G'], parents=case['parents'], original_positive_fragments=len(nodes),
                     conflicting_occupied_atoms=collisions, direct_substitution_feasible=not collisions,
                     original_witnesses_changed=False, new_capacity_searches=0)
        audits.append(audit); write(dest/f'source_{key}_unchanged_witness_gate.json', audit)
        print('UNCHANGED_WITNESS_COMPETITION_GATE', key, 'COLLISIONS', len(collisions), flush=True)
        del case; gc.collect()
    passed = all(a['direct_substitution_feasible'] for a in audits)
    write(dest/'receipt.json', dict(status='INPUT_COMPETITION_AND_ORIGINAL_WITNESSES_PASS' if passed else 'INPUT_COMPETITION_CONTRADICTS_ORIGINAL_WITNESSES',
        cases=audits, G=sum(a['G'] for a in audits), parents=sum(a['parents'] for a in audits),
        six_pairs_request_proof_pass=all(p['proven'] for p in pair_proofs),
        new_fit_calls=0, new_capacity_searches=0, DEV_labels_read=0,
        extraction_correct_rate=None, extraction_precision_mm=None, seconds=time.perf_counter()-start))
    print('COMPETITION_CAPACITY_GATE_COMPLETE', passed, time.perf_counter()-start, flush=True)


if __name__ == '__main__':
    main()
