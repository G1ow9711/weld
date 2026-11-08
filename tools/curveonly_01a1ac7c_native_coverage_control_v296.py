"""Authorized rule source-selection control: unchanged R35 model and proposals."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
import multiprocessing
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_direction_full_v262 as previous
from tools import curveonly_01a1ac7c_ordered_span_full_v238 as full
from tools import curveonly_01a1ac7c_normal_span_family_v285 as resources
from weld_pipeline import curveonly_01a1ac7c_native_coverage_selection_v294 as rules
from weld_pipeline import curveonly_01a1ac7c_native_coverage_runtime_v295 as runtime

DEST, OUTPUT = e.REC/'native_coverage_control_v296', e.OUT/'native_coverage_control_v296'
QUICK = ('curve_0e430b19e6c300a4', 'curve_dfea4a82bc9967c4', 'curve_e51507fb1de03bf5', 'curve_06599ca427fb5556', 'curve_4738a0b266814107')


def valid(binding):
    assert e.sha(binding['path']) == binding['sha256'], binding['path']
    return binding


def check():
    p = e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']:
        valid(b)
    valid(p['model']); valid(p['baseline_outputs']); valid(p['baseline_scores'])
    return p


def prepare():
    assert e.read(e.REC/'normal_span_family_v285/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'rule_source_control_v291/full/CLOSURE.json')['analysis_closed']
    assert e.read(previous.DEST/'run/CLOSURE.json')['analysis_closed']
    assert not (DEST/'FROZEN_PROTOCOL.json').exists(), 'Never overwrite an executed protocol'
    commit = subprocess.check_output(['git', 'rev-parse', 'refs/heads/codex/curve-only-01a1ac7c'], text=True).strip()
    sources = [Path(__file__).resolve(), Path(rules.__file__).resolve(), Path(runtime.__file__).resolve()]
    for path in sources:
        assert path.read_bytes() == subprocess.check_output(['git', 'show', commit+':'+str(path.relative_to(e.ROOT))])
    old = e.read(previous.DEST/'FROZEN_PROTOCOL.json')
    cases = e.read(valid(old['inputs'])['path'])['cases']
    assert len(cases) == 32 and sum(c['split'] == 'TRAIN' for c in cases) == 22
    outputs = {c['source_id']: c for c in e.read(previous.DEST/'run/OUTPUTS_FROZEN.json')['cases']}
    jobs = []
    for c in cases:
        sid = c['source_id']
        jobs.append(dict(source_id=sid, split=c['split'], family=c['family'],
            raw=c['raw'], STEP=c['STEP'],
            features=e.bind(previous.OUTPUT/'features'/sid/'features.npy'),
            geometry=e.bind(previous.OUTPUT/'features'/sid/'geometry.json'),
            baseline_prediction=outputs[sid]['prediction']))
    assert all(next(c for c in cases if c['source_id'] == sid)['split'] == 'TRAIN' for sid in QUICK)
    DEST.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)
    sources = [e.bind(s) for s in sources]+old['sources']+[e.bind(rules.original.__file__),e.bind(resources.__file__)]
    e.write(DEST/'INPUTS.json', dict(cases=jobs, no_truths_or_score_answers=True, cached_precision_not_cold=True))
    e.write(DEST/'FROZEN_PROTOCOL.json', dict(source_commit=commit, sources=sources,
        model=e.bind(previous.DEST/'run/model.pkl'), baseline_outputs=e.bind(previous.DEST/'run/OUTPUTS_FROZEN.json'),
        baseline_scores=e.bind(previous.DEST/'run/RESULT.json'), inputs=e.bind(DEST/'INPUTS.json'),
        previous_control=e.bind(e.REC/'normal_span_family_v285/CLOSURE.json'),
        user_request=e.bind(e.REC/'USER_RULE_INTEGRATION_REQUEST_SIDE_01a1b45d_20261107.md'),
        design=e.bind(e.REC/'NATIVE_COVERAGE_RULE_DESIGN_V294.md'),
        scope=dict(TRAIN_inputs=22, TRAIN_families=18, TRAIN_G=121, VAL_numeric_inputs=9,
            VAL_G=37, VAL_fileversions=10, VAL_fileversion_G=48, source_files=32),
        quick_TRAIN=list(QUICK),
        design_change='Correct demonstrated V289 fragmentation bias: integrate unchanged executed model support over unique original native arclength, exact global compatible selection; equal objective prefers fewer geometric representations. TP units, ranges, features and model unchanged.',
        failed_predecessor=e.bind(e.REC/'rule_source_control_v291/full/CLOSURE.json'),
        physical_integration_measure_not_acceptance_unit=True, no_always_longest=True, finite_support_not_should_weld=True,
        no_source_or_endpoint_or_model_or_feature_or_geometry_change=True,
        new_material_pair_side_and_process_certification=False,
        quick_proceed_criterion='Strict TRAIN final-output F1 increases and TP does not decrease; gains/losses/FP all reported. No validation answer selects a formula or threshold.',
        same3mm_true_gaps_process_cuts_ARC30=True, no_VALfit_mining_or_calibration=True,
        all_outputs_frozen_before_scoring=True, time_optimization_canceled=True,
        round35_model_reused_not_a_new_fit=True, full_contract_pass=False, epoch=time.time()))
    print('RULE_SOURCE_PROTOCOL_FROZEN', commit, flush=True)


def controls():
    check()
    # Verify an actual global alternative to greedy, including a preserved gap.
    chosen, receipt = rules.maximum_compatible([.9, .6, .6], [6, 1, 1])
    assert chosen == [1, 2] and abs(receipt['objective']-1.2) < 1e-12
    # Exact optimization is checked against independent exhaustive enumeration.
    rng = np.random.default_rng(20261106)
    for n in range(1, 11):
        w = rng.random(n).tolist(); adjacency = [0]*n
        for i in range(n):
            for j in range(i):
                if rng.random() < .4:
                    adjacency[i] |= 1 << j; adjacency[j] |= 1 << i
        chosen, receipt = rules.maximum_compatible(w, adjacency)
        optimum = max(sum(w[i] for i in range(n) if mask >> i & 1)
            for mask in range(1 << n)
            if all(not (adjacency[i] & mask) for i in range(n) if mask >> i & 1))
        assert abs(receipt['objective']-optimum) < 1e-12
    # A longer interval has no preference when it has weaker endpoint support.
    chosen, _ = rules.maximum_compatible([.2, .8], [2, 1]); assert chosen == [1]
    p = dict(candidate=0, points=[[0., 0., 0.], [10., 0., 0.]],
             interval=dict(start_mm=0., end_mm=10.))
    packet = dict(candidates=[dict(points=p['points'], material=0)])
    feasible, spans = rules.feasibility(p, packet, [10.])
    assert feasible['feasible'] and feasible['joint_pair_status'] == 'UNKNOWN'
    assert spans == [(0, 0., 10.)]
    feasible, _ = rules.feasibility(dict(p, interval=dict(start_mm=0., end_mm=11.)), packet, [10.])
    assert not feasible['feasible']
    assert rules.pair_identity(dict(joint_material_pair=[1, 0])) is None
    assert rules.pair_identity(dict(joint_material_pair=[1, 0], joint_material_pair_status='CONFIRMED_INPUT_GEOMETRY')) == (0, 1)
    assert rules.measure([(0, 0., 10.)]) == rules.measure([(0, 0., 3.), (0, 3., 7.), (0, 7., 10.)]) == 10.
    assert rules.measure([(0, 0., 7.), (0, 3., 10.)]) == 10.
    chosen, _ = rules.maximum_compatible([2., 2., 2., 6.], [8, 8, 8, 7]); assert chosen == [3]
    chosen, _ = rules.maximum_compatible([.2*10., .8*4.], [2, 1]); assert chosen == [1]
    assert rules.measure([(0, 0., .00001)]) > 0.
    e.write(DEST/'CONTROLS.json', dict(passed=True, exact_graph_optimum_matches_enumeration=True,
        whole_compatible_set_beats_greedy=True, no_always_longest=True,
        native_union_measure_segmentation_invariant=True, duplicated_source_measure_not_double_counted=True,
        equal_coverage_prefers_fewer_representation_records=True, positive_short_source_not_deleted=True,
        native_bounds_checked=True, UNKNOWN_not_certified=True, no_dataset_labels_read=True, epoch=time.time()))
    print('RULE_SOURCE_FUNCTIONAL_CONTROLS_PASSED', flush=True)


def write_csv(path, sid, decoded):
    with path.open('w', newline='', encoding='utf8') as stream:
        fields = ['workpiece_id', 'weld_id', 'parent_id', 'segment_index', 'weld_type', 'geometry_type',
            'is_closed', 'start_x', 'start_y', 'start_z', 'end_x', 'end_y', 'end_z',
            'length_mm', 'confidence', 'curve_points_file', 'status']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for parent in decoded['parents']:
            for segment in parent['segments']:
                a, b = segment['points'][0], segment['points'][-1]
                writer.writerow(dict(workpiece_id=sid, weld_id=parent['parent_id']+'_'+str(segment['segment_index']),
                    parent_id=parent['parent_id'], segment_index=segment['segment_index'],
                    weld_type=parent['weld_type'], geometry_type=segment['geometry_type'],
                    is_closed=parent['is_closed'], start_x=a[0], start_y=a[1], start_z=a[2],
                    end_x=b[0], end_y=b[1], end_z=b[2], length_mm=segment['length_mm'],
                    confidence=parent['confidence'], curve_points_file='curves.json', status=segment['status']))


def one(job):
    p = check(); sid = job['input']['source_id']; phase = job['phase']
    allowed = Path(p['model']['path']).resolve()
    def fence(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            path = Path(os.fsdecode(args[0])); value = str(path)
            if any(q in value for q in ('/truth/', '/targets/', '/review/', '/scores/', '/score_coordinate_maps/')):
                raise PermissionError('Rule forward cannot read answers '+value)
            if path.name in ('FIT_TARGETS.json', 'RESULT.json', 'SCORE.json', 'TRUTH_INDEX.json'):
                raise PermissionError('Rule forward cannot read scores or targets '+value)
            if path.name == 'model.pkl' and path.resolve() != allowed:
                raise PermissionError('Only frozen R35 model is authorized '+value)
        e.fence(event, args)
    sys.addaudithook(fence)
    observed = np.load(valid(job['input']['features'])['path'])
    packet = e.read(valid(job['input']['geometry'])['path'])
    with allowed.open('rb') as stream:
        model = pickle.load(stream)
    with threadpool_limits(limits=2):
        old = previous.runtime.decode(observed, packet, model)
        baseline = e.read(valid(job['input']['baseline_prediction'])['path'])
        assert full.detail.equal(old['parents'], baseline['parents'])
        old_states = json.dumps(model.last_states, sort_keys=True)
        old_normalization = json.dumps(model.last_representation_normalization, sort_keys=True)
        decoded = runtime.decode(observed, packet, model)
        assert json.dumps(model.last_states, sort_keys=True) == old_states
        assert json.dumps(model.last_representation_normalization, sort_keys=True) == old_normalization
        assert decoded['finite_source_selection']['input_proposals'] == old['finite_source_selection']['input_proposals']
    target = OUTPUT/phase/'outputs'/sid; target.mkdir(parents=True, exist_ok=True)
    e.write(target/'curves.json', decoded)
    write_csv(target/'curves.csv', sid, decoded)
    forward = dict(source_id=sid, same_R35_actual_neural_states_and_raw_normalization_exact=True,
        baseline_actual_published_parents_replay_exact=True,
        original_raw_count=old['finite_source_selection']['input_proposals'],
        source_decisions=model.last_states, primitive_representation_normalization=model.last_representation_normalization,
        no_labels_or_new_fitting_or_endpoint_change=True, model=p['model'])
    e.write(target/'FORWARD.json', forward)
    receipt = dict(source_id=sid, split=job['input']['split'], prediction=e.bind(target/'curves.json'),
        csv=e.bind(target/'curves.csv'), forward=e.bind(target/'FORWARD.json'), input_failure=False)
    print('RULE_FINAL_OUTPUT_FROZEN', phase, sid, old['geometry_units'], decoded['geometry_units'], flush=True)
    return receipt


def predict(phase):
    p = check(); assert e.read(DEST/'CONTROLS.json')['passed']
    assert phase in ('quick', 'full')
    if phase == 'full':
        assert e.read(DEST/'quick/DECISION.json')['proceed_full_scope']
        assert e.read(DEST/'quick/CLOSURE.json')['analysis_closed']
    cases = e.read(valid(p['inputs'])['path'])['cases']
    if phase == 'quick':
        cases = [c for c in cases if c['source_id'] in QUICK]
        assert len(cases) == len(QUICK) and all(c['split'] == 'TRAIN' for c in cases)
    target = DEST/phase
    target.mkdir(exist_ok=True)
    assert not (target/'OUTPUTS_FROZEN.json').exists()
    while True:
        resource = resources.resources()
        workers = min(3, int((.9*resource['maximum']-resource['memory']-1024**3)/(2*1024**3)),
                      int((.8*resource['quota']-resource['cpu'])/2))
        if workers >= 1:
            break
        print('RULE_WAIT_SHARED_RESOURCES', resource, flush=True); time.sleep(10)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        futures = [pool.submit(one, dict(phase=phase, input=c)) for c in cases]
        for future in as_completed(futures):
            rows.append(future.result())
            e.write(target/'PROGRESS.json', dict(completed=len(rows), total=len(cases), workers=workers, epoch=time.time()))
    assert len(rows) == len(cases)
    rows.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json', dict(cases=rows, complete_scope=len(rows), all_outputs_closed=True,
        all_frozen_before_scores=True, no_input_failure=True, model=p['model'], protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),
        cached_precision_not_cold=True, no_new_fit=True, epoch=time.time()))
    print('RULE_ALL_FINAL_OUTPUTS_FROZEN', phase, len(rows), flush=True)


def total(rows):
    result = {k:sum(c['metrics'][k] for c in rows) for k in ('G', 'P', 'TP', 'FP', 'FN')}
    result.update(inputs=len(rows), extraction=result['TP']/result['G'] if result['G'] else None,
        FP_per_G=result['FP']/result['G'] if result['G'] else None,
        F1=2*result['TP']/(result['G']+result['P']) if result['G']+result['P'] else None)
    return result


def score(phase):
    p = check(); target = DEST/phase
    assert e.read(target/'PREDICT_ACTUAL_TERMINAL.json')['exit_code'] == 0
    frozen = e.read(target/'OUTPUTS_FROZEN.json'); assert frozen['all_frozen_before_scores']
    baseline = e.read(p['baseline_scores']['path'])
    scores = {c['source_id']:c for c in baseline['original_cases']+baseline['additional_cases']}
    inputs = {c['source_id']:c for c in e.read(p['inputs']['path'])['cases']}
    rows = []
    for output in frozen['cases']:
        sid = output['source_id']; old = scores[sid]
        if phase == 'quick': assert inputs[sid]['split'] == 'TRAIN'
        decoded = e.read(valid(output['prediction'])['path'])
        packet = e.read(valid(inputs[sid]['geometry'])['path'])
        truth = e.read(valid(old['truth_binding'])['path'])['units']
        exported = [full.raw_score.exported(c) for c in decoded['parents']]
        assert all(c['_valid_export'] for c in exported)
        chains, groups = full.grouping.source_groups(packet); parts = []
        for curve in exported:
            ci = curve['source_candidate']; gi = chains[ci]
            parts.append(full.evaluation.annotate(curve, groups[gi], gi, packet['candidates'][ci].get('material'), ci))
        units = full.geometry.units(parts)
        metrics = full.grouping.matching(units, truth)
        categories = {name:full.grouping.matching([c for c in units if bool(c['closed']) == closed],
            [c for c in truth if bool(c['closed']) == closed]) for name, closed in [('open', False), ('closed', True)]}
        before = {m['truth'] for m in old['metrics']['matches']}; after = {m['truth'] for m in metrics['matches']}
        legal = e.read(previous.DEST/'review/cases'/sid/'TRACE.json')['traces']
        observed = np.load(inputs[sid]['features']['path'])
        forward = e.read(valid(output['forward'])['path'])
        grid = previous.runtime.candidate_rows(observed, packet)
        generator = __import__('weld_pipeline.curveonly_01a1ac7c_direction_spans_v247', fromlist=['x'])
        raw, normalization = generator.normalize(generator.codec.proposals(packet, grid,
            [c['codes'] for c in forward['source_decisions']]), packet)
        assert normalization == forward['primitive_representation_normalization']
        assert len(raw) == decoded['finite_source_selection']['input_proposals']
        counts, traces = {}, []
        for ti, reference in enumerate(truth):
            options = legal[ti]['legal_intervals']; ids = {c['candidate'] for c in options}
            correct = [i for i, c in enumerate(raw) if c['candidate'] in ids and full.raw_score.geometry_error(c, reference) is not None]
            if ti in after: reason = 'COMPLETE_GEOMETRY_TP'
            elif not ids: reason = 'NO_LEGAL_COMPLETE_INPUT_CARRIER'
            elif correct: reason = 'SOURCE_SELECTION_EXPORT_OR_GROUPING_REJECTS_COMPLETE_RAW_INTERVAL'
            elif not any(forward['source_decisions'][ci]['codes'] for ci in ids): reason = 'MODEL_REJECTS_ALL_LEGAL_SOURCES'
            else: reason = 'RAW_COMPLETE_ENDPOINT_PAIR_RANGE_OR_TYPE_MISMATCH'
            counts[reason] = counts.get(reason, 0)+1
            traces.append(dict(truth=ti, reason=reason, legal_intervals=options, correct_raw=correct,
                               physical_process_cause='UNKNOWN'))
        csv_receipt = full.detail.exports(output)
        turns = [float(v)*180/np.pi for c in exported for v in c['_segment_turns'] if v is not None]
        row = dict(source_id=sid, name=old.get('name', sid), split=inputs[sid]['split'], metrics=metrics,
            previous_metrics=old['metrics'], categories=categories, gained=sorted(after-before), lost=sorted(before-after),
            truth_binding=old['truth_binding'], output=output, counts=counts, traces=traces,
            csv=csv_receipt, maximum_ARC_child_degrees=max(turns) if turns else None,
            actual_R35_forward_raw_exact=True, input_geometry_unchanged=True)
        e.write(target/'cases'/sid/'SCORE.json', row); rows.append(row)
        print('RULE_FINAL_SCORE', phase, sid, metrics['TP'], metrics['G'], metrics['FP'], flush=True)
    totals = dict(TRAIN=total([c for c in rows if c['split'] == 'TRAIN']))
    canonical = {c['canonical_source_id'] for c in e.read(previous.evaluation.DEST/'FROZEN_PROTOCOL.json')['numeric_input_choices']}
    added = {c['source_id'] for c in baseline['additional_cases']}
    canonical |= added
    if phase == 'full':
        totals.update(old4_VAL=total([c for c in rows if c['split'] == 'VAL' and c['source_id'] in canonical-added]),
            additional5_VAL=total([c for c in rows if c['source_id'] in added]),
            all9_VAL=total([c for c in rows if c['split'] == 'VAL' and c['source_id'] in canonical]),
            all10_labelversion_VAL=total([c for c in rows if c['split'] == 'VAL']))
        assert totals['TRAIN']['G'] == 121 and totals['all9_VAL']['G'] == 37
        assert totals['all10_labelversion_VAL']['G'] == 48
    selected = [c for c in rows if phase == 'quick' or c['split'] == 'TRAIN' or c['source_id'] in canonical]
    gained = [(c['source_id'], ti) for c in selected for ti in c['gained']]
    lost = [(c['source_id'], ti) for c in selected for ti in c['lost']]
    categories = {split:{name:total([dict(metrics=c['categories'][name]) for c in selected if c['split'] == split])
        for name in ('open', 'closed')} for split in ('TRAIN', 'VAL')}
    e.write(target/'RESULT.json', dict(complete=True, phase=phase, cases=rows, totals=totals,
        categories=categories, gained=gained, lost=lost, frozen_outputs=e.bind(target/'OUTPUTS_FROZEN.json'),
        no_VALfit_mining_or_calibration=True, current_model_weights_unchanged=True,
        cached_precision_not_cold=True, full_contract_pass=False, epoch=time.time()))
    if phase == 'quick':
        old_total = total([dict(metrics=c['previous_metrics']) for c in rows]); new_total = totals['TRAIN']
        proceed = new_total['F1'] > old_total['F1'] and new_total['TP'] >= old_total['TP']
        e.write(target/'DECISION.json', dict(proceed_full_scope=proceed, previous=old_total, current=new_total,
            gained=gained, lost=lost, frozen_criterion_unchanged=True,
            reason='TRAIN final-output criterion passed' if proceed else 'TRAIN criterion failed; analyze before any full VAL forward',
            no_new_VAL_forward_or_score=True, epoch=time.time()))
    # Display is diagnostic only; actual strict3D matching above decides scores.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen = [c for c in rows if c['source_id'] in QUICK] if phase == 'quick' else [c for c in rows if c['source_id'] in ('curve_71166e99b1f46203', 'curve_0ccbe50eaf3cb217', 'curve_08dcd4a0601b8674')]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(14, 4*len(chosen)), squeeze=False)
    old_outputs = {c['source_id']:c for c in e.read(p['baseline_outputs']['path'])['cases']}
    for i, c in enumerate(chosen):
        sid = c['source_id']; truth = e.read(c['truth_binding']['path'])['units']
        cloud = np.concatenate([np.asarray(q['points']) for q in truth]); center = cloud.mean(0)
        basis = np.linalg.svd(cloud-center, full_matrices=False)[2][:2]
        for j, (binding, metric, title) in enumerate([(old_outputs[sid]['prediction'], c['previous_metrics'], 'R35'),
                (c['output']['prediction'], c['metrics'], 'RuleV294')]):
            ax = axes[i, j]
            for q in truth:
                xy = (np.asarray(q['points'])-center)@basis.T
                ax.plot(xy[:, 0], xy[:, 1], color='black', linewidth=3, alpha=.35)
            for q in e.read(binding['path'])['parents']:
                xy = (np.asarray(q['points'])-center)@basis.T
                ax.plot(xy[:, 0], xy[:, 1], color='royalblue', linewidth=1)
            ax.set_title(f"{sid} {title}: {metric['TP']}/{metric['G']} FP{metric['FP']} FN{metric['FN']}")
            ax.set_aspect('equal', adjustable='datalim')
    fig.suptitle('Final curves: black GT, blue published. Strict3D score retained.'); fig.tight_layout()
    path = OUTPUT/phase/'FINAL_COMPARISON.png'; fig.savefig(path, dpi=120); plt.close(fig)
    e.write(target/'VISUAL.json', dict(figure=e.bind(path), actual_main_inspection_required=True))
    print('RULE_FINAL_TOTALS', phase, totals, 'gained', len(gained), 'lost', len(lost), flush=True)


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare': prepare()
    elif action == 'controls': controls()
    elif action == 'predict': predict(sys.argv[2])
    elif action == 'score': score(sys.argv[2])
    else: raise ValueError(action)
