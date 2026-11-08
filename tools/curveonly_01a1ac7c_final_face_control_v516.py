"""Whole-dataset control of the final unchanged competing-face gate."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import copy, gc, multiprocessing, pickle, subprocess, sys, time, types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tools import curveonly_01a1ac7c_finite_curve_control_v506 as prior
from tools import curveonly_01a1ac7c_paired_face_control_v409 as guards
from weld_pipeline import curveonly_01a1ac7c_final_face_runtime_v514 as runtime

e, common = prior.e, prior.common
DEST = e.REC / 'final_face_control_v516'
OUTPUT = e.OUT / 'final_face_control_v516'


def check():
    p = e.read(DEST / 'FROZEN_PROTOCOL.json')
    for b in p['sources'] + [p['inputs'], p['baseline_scores'], p['baseline_outputs'], p['model_bundle']]:
        common.valid(b)
    return p


def prepare():
    assert not (DEST / 'FROZEN_PROTOCOL.json').exists()
    assert e.read(prior.DEST / 'full/CLOSURE.json')['adopted_research']
    assert e.read(e.REC / 'final_face_audit_v512/CLOSURE.json')['analysis_closed']
    controls = e.REC / 'final_face_controls_v515'
    assert e.read(controls / 'ACTUAL_TERMINAL.json')['exit_code'] == 0
    assert e.read(controls / 'CLOSURE.json')['count'] == 17
    p = prior.check()
    outputs = {c['source_id']: c for c in e.read(prior.DEST / 'full/OUTPUTS_FROZEN.json')['cases']}
    jobs = []
    for j in e.read(p['inputs']['path'])['cases']:
        old = outputs[j['source_id']]
        prepared = e.read(common.valid(j['prepared'])['path'])
        inventory = j['actual_inventory'] or prepared['inventory']
        jobs.append(dict(j, current_prediction=old['prediction'],
                         model_geometry=old['model_input_geometry'], model_features=old['features'],
                         source_geometry=old['source_geometry'], final_inventory=inventory))
    commit = subprocess.check_output(['git', 'rev-parse', 'refs/heads/codex/curve-only-01a1ac7c'], text=True).strip()
    sources = [Path(__file__).resolve(), Path(runtime.__file__).resolve(), Path(runtime.rules.__file__).resolve()]
    for source in sources:
        assert source.read_bytes() == subprocess.check_output(['git', 'show', commit + ':' + str(source.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    e.write(DEST / 'INPUTS.json', dict(cases=jobs, no_truth_or_scores=True))
    e.write(DEST / 'BASELINE_V506_SCORES.json', dict(original_cases=e.read(prior.DEST / 'full/RESULT.json')['cases'], additional_cases=[]))
    e.write(DEST / 'FROZEN_PROTOCOL.json', dict(p, source_commit=commit,
        sources=[e.bind(s) for s in sources] + p['sources'], inputs=e.bind(DEST / 'INPUTS.json'),
        baseline_scores=e.bind(DEST / 'BASELINE_V506_SCORES.json'), baseline_outputs=e.bind(prior.DEST / 'full/OUTPUTS_FROZEN.json'),
        controls=e.bind(controls / 'CLOSURE.json'), diagnosis=e.bind(e.REC / 'final_face_audit_v512/CLOSURE.json'),
        change='Repeat original V407 gate after every recovery; preserve early audit, exclude unmodeled derived sources as UNKNOWN without fabricated features.',
        no_new_fit_or_VALcalibration=True, quick_is_all22TRAIN=True,
        cached_precision_not_cold=True, epoch=time.time()))


def one(value):
    job, phase, model_binding = value
    guards.install_fence()
    packet = e.read(common.valid(job['model_geometry'])['path'])
    features = np.load(common.valid(job['model_features'])['path'])
    before = e.read(common.valid(job['current_prediction'])['path'])
    with Path(common.valid(model_binding)['path']).open('rb') as stream:
        model = pickle.load(stream)
    qualified = bool(job['STEP'] and packet.get('pose', {}).get('passed'))
    solids = e.core.load_material(common.valid(job['STEP'])['path']) if qualified else None
    # The original inspect() returns before constructing its context when no
    # selected model source is rejected. Avoid reading unused inventory bytes.
    modeled = [p for p in before['parents'] if p['source_candidate'] < len(packet['candidates'])]
    votes = runtime.rules.previous.expert.classify(features, packet, model.paired.source_expert) if qualified else []
    needs = qualified and any(votes[p['source_candidate']]['rejected'] for p in modeled)
    inventory = e.read(common.valid(job['final_inventory'])['path']) if needs else None
    result = runtime.rules.correct(before, features, packet, inventory, solids, model.paired.source_expert)
    audit = result['final_paired_face_source_rule']
    removed = set(audit['removed_parent_indices'])
    retained = [p for i, p in enumerate(before['parents']) if i not in removed]
    assert len(retained) == len(result['parents'])
    for a, b in zip(retained, result['parents']):
        assert {k:v for k,v in a.items() if k != 'parent_id'} == {k:v for k,v in b.items() if k != 'parent_id'}
    assert before.get('paired_face_source_rule') == result.get('paired_face_source_rule')
    target = OUTPUT / phase / 'outputs' / job['source_id']
    target.mkdir(parents=True, exist_ok=True)
    e.write(target / 'curves.json', result)
    common.shared.write_csv(target / 'curves.csv', job['source_id'], result)
    e.write(target / 'RULE_TRACE.json', dict(input=job, audit=audit,
        cached_actual_upstream_model_output=True, actual_frozen_source_expert_forward=True,
        fresh_finite_material_queries=bool(needs), no_labels_or_scores=True,
        retained_geometry_process_and_fields_exact=True, original_early_audit_preserved=True, epoch=time.time()))
    print('FINAL_FACE_OUTPUT', phase, job['source_id'], len(before['parents']), len(result['parents']), sorted(removed), flush=True)
    return dict(source_id=job['source_id'], split=job['split'], prediction=e.bind(target / 'curves.json'),
        csv=e.bind(target / 'curves.csv'), trace=e.bind(target / 'RULE_TRACE.json'),
        source_geometry=job['source_geometry'], model_input_geometry=job['model_geometry'],
        features=job['model_features'], input_failure=False)


def estimate(job):
    return .65 * 1024**3 + 8 * (job['model_geometry']['bytes'] + job['final_inventory']['bytes']) + job['model_features']['bytes']


def predict(phase):
    p = check()
    jobs = e.read(p['inputs']['path'])['cases']
    outputs = []
    if phase == 'quick':
        jobs = [j for j in jobs if j['split'] == 'TRAIN']
    else:
        assert phase == 'full' and e.read(DEST / 'quick/CLOSURE.json')['proceed_full']
        outputs = e.read(DEST / 'quick/OUTPUTS_FROZEN.json')['cases']
        jobs = [j for j in jobs if j['split'] == 'VAL']
    target = DEST / phase
    target.mkdir(exist_ok=True)
    assert not (target / 'OUTPUTS_FROZEN.json').exists()
    remaining = sorted(jobs, key=estimate, reverse=True)
    resources = []
    # Independent workpieces share at most two workers. Conservative actual
    # input sizes determine each admitted batch; every child exits per case.
    while remaining:
        state = common.shared.resources.resources()
        available = .9 * state['maximum'] - state['memory'] - .4 * 1024**3
        slots = min(2, int(.8 * state['quota'] - state['cpu']))
        chosen = []
        reserved = 0
        for job in remaining:
            if len(chosen) < slots and reserved + estimate(job) <= available:
                chosen.append(job)
                reserved += estimate(job)
        if not chosen:
            e.write(target / 'RESOURCE_WAIT.json', dict(resource=state, next_estimated_bytes=estimate(remaining[0]), remaining=len(remaining), epoch=time.time()))
            time.sleep(5)
            continue
        resources.append(dict(resource=state, admitted=[j['source_id'] for j in chosen], estimated_reserved_bytes=reserved))
        with ProcessPoolExecutor(max_workers=len(chosen), mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
            for future in as_completed([pool.submit(one, (j, phase, p['model_bundle'])) for j in chosen]):
                outputs.append(future.result())
                e.write(target / 'PROGRESS.json', dict(cases=outputs, remaining=len(remaining), resources=resources, epoch=time.time()))
        remaining = [j for j in remaining if j not in chosen]
        gc.collect()
    outputs.sort(key=lambda c: c['source_id'])
    lookup = {c['source_id']: c for c in outputs}
    alljobs = e.read(p['inputs']['path'])['cases']
    scoring = [dict(j, geometry=lookup[j['source_id']]['source_geometry']) for j in alljobs if j['source_id'] in lookup]
    e.write(target / 'SCORING_SOURCE_INPUTS.json', dict(cases=scoring, no_truth_or_scores=True))
    e.write(target / 'OUTPUTS_FROZEN.json', dict(cases=outputs, all_outputs_closed_before_scores=True,
        source_geometry_inputs=e.bind(target / 'SCORING_SOURCE_INPUTS.json'), resources=resources,
        cached_precision_not_cold=True, epoch=time.time()))


def score(phase):
    p = check()
    frozen = e.read(DEST / phase / 'OUTPUTS_FROZEN.json')
    def localcheck():
        return dict(p, inputs=frozen['source_geometry_inputs'])
    types.FunctionType(common.score.__code__, dict(common.__dict__, DEST=DEST, check=localcheck))(phase)
    result = e.read(DEST / phase / 'RESULT.json')
    result['baseline'] = 'V506_previous_full_and_frozen_historical_best'
    e.write(DEST / phase / 'RESULT.json', result)
    if phase == 'quick':
        now = result['totals']['TRAIN']
        old = e.read(prior.DEST / 'full/SCOPE_TOTALS.json')['TRAIN']
        proceed = now['TP'] >= old['TP'] and now['FP'] < old['FP'] and now['F1'] > old['F1']
        e.write(DEST / phase / 'CLOSURE.json', dict(analysis_closed=True, proceed_full=proceed,
            metrics=now, baseline=old, gained=sum(len(c['gained']) for c in result['cases']),
            lost=sum(len(c['lost']) for c in result['cases']), epoch=time.time()))
        print('FINAL_FACE_QUICK_PROCEED', proceed, now, flush=True)


def integration():
    p = check()
    assert e.read(DEST / 'FULL_SCORE_ACTUAL_TERMINAL.json')['exit_code'] == 0
    outputs = {c['source_id']: c for c in e.read(DEST / 'full/OUTPUTS_FROZEN.json')['cases']}
    job = next(c for c in e.read(p['inputs']['path'])['cases'] if c['source_id'] == 'curve_e51507fb1de03bf5')
    guards.install_fence()
    features = np.load(common.valid(job['model_features'])['path'])
    packet = e.read(common.valid(job['model_geometry'])['path'])
    inventory = e.read(common.valid(job['final_inventory'])['path'])
    solids = e.core.load_material(common.valid(job['STEP'])['path'])
    with Path(common.valid(p['model_bundle'])['path']).open('rb') as stream:
        model = pickle.load(stream)
    actual, before = runtime.decode_existing(features, packet, model, solids, inventory, return_before=True)
    frozen_before = e.read(common.valid(job['current_prediction'])['path'])
    frozen = e.read(common.valid(outputs[job['source_id']]['prediction'])['path'])
    assert before['parents'] == frozen_before['parents']
    assert actual['parents'] == frozen['parents']
    prep = e.read(common.valid(job['prepared'])['path'])
    original = e.read(common.valid(prep['original_forward'])['path'])
    assert model.last_states == original['source_decisions']
    from tools import curveonly_01a1ac7c_coplanar_source_report_v459 as compare
    diffs = compare.compare_geometry(actual['final_paired_face_source_rule'], frozen['final_paired_face_source_rule'])
    assert actual['paired_face_source_rule'] == frozen['paired_face_source_rule']
    e.write(DEST / 'full/ACTUAL_MODEL_RULE_INTEGRATION.json', dict(passed=True,
        cases=[dict(source_id=job['source_id'], actual_R35_and_experts_forward=True,
            upstream_V506_parents_exact=True, final_parents_exact=True, original_neural_states_exact=True,
            fresh_final_finite_geometry_queries=True, numeric_diffs=diffs,
            original_early_gate_audit_exact=True)],
        runtime=e.bind(runtime.__file__), model=p['model_bundle'],
        no_label_or_score_reads=True, cached_input_observations_not_cold=True, epoch=time.time()))
    print('FINAL_FACE_ACTUAL_MODEL_EQUAL', job['source_id'], len(diffs), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'integration':
        integration()
    else:
        {'predict': predict, 'score': score}[sys.argv[1]](sys.argv[2])
