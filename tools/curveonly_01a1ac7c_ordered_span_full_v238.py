"""R33: fresh ordered span generator on all22TRAIN, unchanged observations and V60."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import multiprocessing
import pickle
import subprocess
import sys
import time
import types
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_ordered_span_train_v236 as training
from tools import curveonly_01a1ac7c_section_round_v146 as original
from tools import curveonly_01a1ac7c_supplement_predictions_v129 as supplement
from tools import curveonly_01a1ac7c_supplement_score_v130 as new_score
from tools import curveonly_01a1ac7c_geometry_eval_v27 as grouping
from tools import curveonly_01a1ac7c_complete_eval_v60 as evaluation
from tools import curveonly_01a1ac7c_score_v2 as raw_score
from tools import curveonly_01a1ac7c_local_review_v101 as detail
from weld_pipeline import curveonly_01a1ac7c_ordered_spans_v235 as runtime
from weld_pipeline import curveonly_01a1ac7c_complete_units_v60 as geometry

NAME = 'round33'
DEST, OUTPUT = e.REC/NAME, e.OUT/NAME


def valid(binding):
    assert e.sha(binding['path']) == binding['sha256'], binding['path']
    return binding


def prepare():
    closed=e.REC/'ordered_span_controls_v237/CLOSURE.json'
    decision=e.REC/'ordered_span_controls_v237/DECISION.json'
    assert e.read(closed)['passed'] and e.read(decision)['proceed_full22_fit_and_all9_VAL']
    assert e.read(e.REC/'round32/review/VISUAL_INSPECTION.json')['actually_viewed']
    for b in e.read(closed)['bindings'].values():valid(b)
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    inputs=e.read(e.REC/'round32/INPUTS.json');targets=e.read(e.REC/'round32/FIT_TARGETS.json')
    provenance=e.read(e.REC/'round32/INPUT_PROVENANCE.json')
    assert len(inputs['cases'])==32 and len(targets['cases'])==22
    assert all(q['split']=='TRAIN' for q in targets['cases'])
    for q in provenance['cases']:
        valid(q['features']);valid(q['geometry']);valid(q['receipt'])
    e.write(DEST/'INPUTS.json',inputs);e.write(DEST/'FIT_TARGETS.json',targets)
    e.write(DEST/'INPUT_PROVENANCE.json',provenance)
    if not (OUTPUT/'features').exists():(OUTPUT/'features').symlink_to(e.OUT/'round32/features',target_is_directory=True)
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    for module in (runtime,training,sys.modules[__name__]):
        path=Path(module.__file__).resolve()
        assert path.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(path.relative_to(e.ROOT))])
    old=e.read(e.REC/'round32/FROZEN_PROTOCOL.json')
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,
        sources=old['sources']+[e.bind(q.__file__) for q in (runtime,training,sys.modules[__name__])],
        closure=e.bind(closed),decision=e.bind(decision),inputs=e.bind(DEST/'INPUTS.json'),
        targets=e.bind(DEST/'FIT_TARGETS.json'),provenance=e.bind(DEST/'INPUT_PROVENANCE.json'),
        reference=e.bind(e.REC/'round32/run/RESULT.json'),reference_comparison=e.bind(e.REC/'round32/run/COMPARISON.json'),
        backbone_settings=training.SETTINGS,feature_names=list(runtime.NAMES),
        scope=dict(TRAIN_inputs=22,TRAIN_families=18,TRAIN_G=121,VAL_numeric_inputs=9,VAL_fileversions=10,old4_G=27,new5_G=10,all9_G=37),
        design='Fresh95476parameter recurrent span pointer. Same111observations/169literal-alias labels/128epochs/seed/globalnormalizer logic. Generate complete ordered endpoint pairs with bounded local offsets and learned continuation; full closed-cycle action. No independent foreground gate, binary peak pairing, fixed count classes, old weights, minimum weld/gap. Two orientations merge before unchanged219identity/56selection/export/V60.',
        model_field_semantics='3diagnostics: emitted-range support confidence, maximum executed start attention, maximum executed end attention; these are notR32independent binary fields. Actual recurrent steps retained.',
        literature='https://arxiv.org/abs/1506.03134 ; general pointer mechanism, task-specific grammar/efficacy unproven',
        all32_outputs_freeze_before_scores=True,no_VALfit_mining_or_threshold_calibration=True,
        full_contract_acceptance_not_claimed=True,cached_precision_not_cold_time=True,
        reserve_fit_bytes=6*1024**3,reserve_predict_bytes=2*1024**3,other_future_reserve_bytes=2*1024**3,
        GPU_minimum_free_bytes=12*1024**3,CPU_threads=2,CPU_fraction_ceiling=.8,memory_fraction_ceiling=.9,epoch=time.time()))
    print('ROUND33_FRESH_ORDERED_MODEL_PROTOCOL',commit,flush=True)


def check():
    p = e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p[k] for k in ('closure', 'decision', 'inputs', 'targets', 'provenance', 'reference', 'reference_comparison')]:
        valid(b)
    return p


def check_sources():
    for b in e.read(DEST/'FROZEN_PROTOCOL.json')['sources']:
        valid(b)


def admission(stage):
    p = check()
    maximum = int(Path('/sys/fs/cgroup/memory.max').read_text())
    memory = int(Path('/sys/fs/cgroup/memory.current').read_text())
    quota, period = map(int, Path('/sys/fs/cgroup/cpu.max').read_text().split())
    def usage():
        return int(dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])
    a, begin = usage(), time.monotonic()
    time.sleep(2)
    cpu = (usage()-a)/1e6/(time.monotonic()-begin)
    reserve = p['reserve_fit_bytes'] if stage == 'fit' else p['reserve_predict_bytes']
    free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])*1024**2
    admitted = memory+reserve+p['other_future_reserve_bytes'] < .9*maximum and cpu+2 < .8*quota/period
    if stage == 'fit':
        admitted &= free >= p['GPU_minimum_free_bytes']
    e.write(DEST/(stage.upper()+'_ADMISSION.json'), dict(admitted=bool(admitted), memory=memory, maximum=maximum,
        shared_cpu_cores=cpu, free_GPU_bytes=free, reserve=reserve, epoch=time.time()))
    assert admitted, 'Shared resource reservation not available'


def fit():
    p=check();admission('fit')
    cases=e.read(p['targets']['path'])['cases']
    val={q['source_id'] for q in e.read(p['inputs']['path'])['cases'] if q['split']=='VAL'}
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            path=Path(os.fsdecode(args[0]));value=str(path)
            if any(q in value for q in ('/score_coordinate_maps/','/supplement_complete_eval_v130/','/truth/')) or path.name=='TRUTH_INDEX.json':
                raise PermissionError('Training cannot read scoring answers: '+value)
            if any(sid in value for sid in val):raise PermissionError('Training cannot read VAL: '+value)
            if path.name=='model.pkl' and not str(path.resolve()).startswith(str(DEST.resolve())+'/'):
                raise PermissionError('Fresh ordered model cannot read previous fitted models: '+value)
        e.fence(event,args)
    sys.addaudithook(guard)
    network,mean,scale,fit=training.fit(cases,DEST/'fit')
    (DEST/'run').mkdir(exist_ok=True)
    with (DEST/'run/model.pkl').open('wb') as file:pickle.dump(runtime.OrderedSpanSet(network,mean,scale),file,protocol=5)
    e.write(DEST/'run/MODEL_SPEC.json',dict(model=e.bind(DEST/'run/model.pkl'),fit=e.bind(DEST/'fit/FIT.json'),
        parameter_count=fit['parameter_count'],TRAIN_inputs=22,feature_names=list(runtime.NAMES),
        all_saved_parameters_CPU_reload_bit_exact=fit['all_saved_parameters_CPU_reload_bit_exact'],
        no_held_or_VALfit=True,fresh_all_parameters_no_old_weights=True,
        recurrent_ordered_span_decoder_not_R32_point_heads=True,epoch=time.time()))
    print('ROUND33_MODEL_FROZEN',e.bind(DEST/'run/model.pkl'),flush=True)


def predict():
    check_sources()
    assert e.read(DEST/'FIT_ACTUAL_TERMINAL.json')['exit_code'] == 0
    admission('predict')
    valid(e.read(DEST/'run/MODEL_SPEC.json')['model'])
    allowed = (DEST/'run/model.pkl').resolve()
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            path = Path(os.fsdecode(args[0]))
            value = str(path)
            if any(q in value for q in ('/truth/', '/targets/', '/first_failure/', '/training_alternatives_v54/', '/score_coordinate_maps/')) or path.name in ('TRUTH_INDEX.json', 'FIT_TARGETS.json'):
                raise PermissionError('Inference cannot read labels: '+value)
            if path.name == 'model.pkl' and path.resolve() != allowed:
                raise PermissionError('Unexpected inference model: '+value)
        e.fence(event, args)
    sys.addaudithook(guard)
    types.FunctionType(e.predict.__code__, dict(e.__dict__, REC=DEST, OUT=OUTPUT, core=runtime, check_source=check_sources))('run')
    frozen = e.read(DEST/'run/OUTPUTS_FROZEN.json')
    assert frozen['all_outputs_closed'] and frozen['complete_scope'] == 32
    assert not any(q['input_failure'] for q in frozen['cases'])
    for case in frozen['cases']:
        valid(case['prediction'])
        valid(case['csv'])
    print('ROUND33_ALL32_OUTPUTS_CLOSED', flush=True)


def total(rows):
    v = {k: sum(q['metrics'][k] for q in rows) for k in ('G', 'P', 'TP', 'FP', 'FN')}
    v.update(inputs=len(rows), extraction=v['TP']/v['G'] if v['G'] else None, FP_per_G=v['FP']/v['G'] if v['G'] else None)
    return v


def score():
    p = check(); assert e.read(DEST/'PREDICT_ACTUAL_TERMINAL.json')['exit_code']==0
    frozen = e.read(DEST/'run/OUTPUTS_FROZEN.json'); assert frozen['complete_scope']==32 and frozen['all_outputs_closed']
    for case in frozen['cases']:
        valid(case['prediction']); valid(case['csv'])
    added_ids = {q['source_id'] for q in e.read(supplement.DEST/'INPUTS.json')['cases']}
    name = NAME+'/original_scope'; rec, out = e.REC/name, e.OUT/name
    (rec/'run').mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    if not (out/'features').exists(): (out/'features').symlink_to(OUTPUT/'features', target_is_directory=True)
    original = [q for q in frozen['cases'] if q['source_id'] not in added_ids]; assert len(original)==27
    e.write(rec/'run/OUTPUTS_FROZEN.json', dict(frozen, cases=original, complete_scope=27, full32_freeze=e.bind(DEST/'run/OUTPUTS_FROZEN.json')))
    evaluation.evaluate(name)
    old = e.read(evaluation.DEST/'scores'/name/'RESULT.json')
    new_score.check(); index = e.read(new_score.DEST/'TRUTH_INDEX.json'); assert index['complete']
    truthmap = {q['source_id']:q for q in index['cases']}; added = []
    for case in frozen['cases']:
        sid = case['source_id']
        if sid not in added_ids: continue
        packet = e.read(OUTPUT/'features'/sid/'geometry.json'); decoded = e.read(case['prediction']['path'])
        exported = [raw_score.exported(q) for q in decoded['parents']]; chains, groups = grouping.source_groups(packet); parts = []
        for curve in exported:
            ci = curve.get('source_candidate')
            if isinstance(ci, int) and 0<=ci<len(chains):
                gi = chains[ci]; parts.append(evaluation.annotate(curve, groups[gi], gi, packet['candidates'][ci].get('material'), ci))
            else: parts.append(dict(curve, _carrier=None, _material=None))
        normalized = geometry.units(parts); truth = e.read(valid(truthmap[sid]['truth'])['path'])['units']
        metrics = grouping.matching(normalized, truth)
        raw = e.read(truthmap[sid]['raw_truth']['path'])
        row = dict(source_id=sid, name=truthmap[sid]['name'], split='VAL', metrics=metrics, prediction_groups=normalized,
            truth_binding=truthmap[sid]['truth'], prediction_binding=case['prediction'],
            raw_geometry=raw_score.matching(exported, raw['runs']), raw_process=raw_score.matching(exported, raw['runs'], raw),
            raw_curve_children=truthmap[sid]['raw_children'], process_direction_side_attributes_unaccepted=True)
        e.write(DEST/'additional5_scores'/sid/'SCORE.json', row)
        added.append({k:v for k,v in row.items() if k!='prediction_groups'})
    assert len(added)==5
    train = old['totals']['unique_numeric_inputs']['TRAIN']; old4 = old['totals']['unique_numeric_inputs']['VAL']; new5 = total(added)
    assert train['G']==121 and old4['G']==27 and new5['G']==10
    combined = {k:old4[k]+new5[k] for k in ('G','P','TP','FP','FN')}
    combined.update(inputs=9, extraction=combined['TP']/combined['G'], FP_per_G=combined['FP']/combined['G'])
    e.write(DEST/'run/RESULT.json', dict(complete=True, round=33, primary_version='V60',
        totals=dict(TRAIN=train, old4_VAL=old4, additional5_VAL=new5, all9_VAL=combined),
        original_scope_result=e.bind(evaluation.DEST/'scores'/name/'RESULT.json'), original_cases=old['cases'], additional_cases=added,
        frozen_outputs=e.bind(DEST/'run/OUTPUTS_FROZEN.json'), model=e.bind(DEST/'run/model.pkl'),
        all32_frozen_before_scores=True, no_VALfit_mining_calibration=True, review_required_before_adoption_or_next_round=True,
        best_round27_not_replaced=True, no_full_contract_or_goal_acceptance=True, epoch=time.time()))
    print('ROUND33_COMPLETE_SCORES', train, old4, new5, combined, flush=True)


def compare():
    p = check()
    result = e.read(DEST/'run/RESULT.json')
    previous = e.read(p['reference']['path'])
    history = e.read(p['reference_comparison']['path'])['reference_totals']
    reference_cases = {'round32': previous['original_cases']+previous['additional_cases']}
    prior31=e.read(e.REC/'round31/run/RESULT.json')
    reference_cases['round31']=prior31['original_cases']+prior31['additional_cases']
    prior30=e.read(e.REC/'round30/run/RESULT.json')
    reference_cases['round30']=prior30['original_cases']+prior30['additional_cases']
    prior29=e.read(e.REC/'round29/run/RESULT.json')
    reference_cases['round29']=prior29['original_cases']+prior29['additional_cases']
    for name in ('round27', 'round28'):
        primary = e.read(evaluation.DEST/'scores'/name/'RESULT.json')['cases']
        added = e.read(new_score.DEST/'scores'/name/'RESULT.json')['cases']
        reference_cases[name] = primary+added
    current = result['original_cases']+result['additional_cases']
    changes = {}
    for name, rows in reference_cases.items():
        assert {q['source_id'] for q in rows} == {q['source_id'] for q in current}
        before = {(q['source_id'], item['truth']) for q in rows for item in q['metrics']['matches']}
        after = {(q['source_id'], item['truth']) for q in current for item in q['metrics']['matches']}
        totals = previous['totals'] if name == 'round32' else history[name]
        changes[name] = dict(gained=sorted(after-before), lost=sorted(before-after),
            percentage_point_changes={key:100*(result['totals'][key]['extraction']-totals[key]['extraction']) for key in result['totals']},
            FP_changes={key:result['totals'][key]['FP']-totals[key]['FP'] for key in result['totals']})
    e.write(DEST/'run/COMPARISON.json', dict(complete=True, reference_totals=dict(history, round32=previous['totals']),
        current_totals=result['totals'], changes=changes, canonical_V60_units_and_roles_unchanged=True,
        no_unit_change_counted_as_gain=True, epoch=time.time()))
    print('ROUND33_COMPARISONS_COMPLETE', flush=True)


def review_one(job):
    output, active = job['output'], [True]
    sid = output['source_id']
    def guard(event, args):
        if active[0]:
            if event == 'open' and isinstance(args[0], (str, bytes, Path)):
                value = os.fsdecode(args[0])
                if any(q in value for q in ('/truth/', '/targets/', '/first_failure/', '/score_coordinate_maps/')):
                    raise PermissionError('Replay forward cannot read scoring answers: '+value)
            e.fence(event, args)
    sys.addaudithook(guard)
    with (DEST/'run/model.pkl').open('rb') as file:
        model = pickle.load(file)
    folder = OUTPUT/'features'/sid
    features, packet = np.load(folder/'features.npy'), e.read(folder/'geometry.json')
    with threadpool_limits(limits=2):
        proposals = model.predict(features, packet)
        states, fields, full = model.last_states, model.last_fields, model.last_full
        decoded = runtime.decode(features, packet, model)
    frozen = e.read(valid(output['prediction'])['path'])
    assert detail.equal(decoded['parents'], frozen['parents'])
    assert len(proposals) == frozen['finite_source_selection']['input_proposals']
    active[0] = False
    truth = e.read(valid(job['truth'])['path'])['units']
    score = e.read(job['score_path'])
    if job['additional']:
        options = e.read(valid(job['legal'])['path'])['options']
    else:
        traces = e.read(valid(job['legal'])['path'])['traces']
        assert len(traces) == len(truth)
        options = [dict(q['legal_interval'], truth=t['truth']) for t in traces for q in t['legal_candidates']]
    matched = {q['truth'] for q in score['metrics']['matches']}
    counts, traces, legal_sources = Counter(), [], {q['candidate'] for q in options}
    for ti, reference in enumerate(truth):
        legal = [q for q in options if q['truth'] == ti]
        ids = sorted({q['candidate'] for q in legal})
        raw_matches = [q for q in proposals if q['candidate'] in ids and raw_score.geometry_error(q, reference) is not None]
        if ti in matched:
            reason = 'COMPLETE_GEOMETRY_TP'
        elif not ids:
            reason = 'NO_LEGAL_COMPLETE_INPUT_CARRIER'
        elif raw_matches:
            reason = 'SOURCE_SELECTION_EXPORT_OR_GROUPING_REJECTS_COMPLETE_RAW_INTERVAL'
        elif not any(states[ci]['codes'] for ci in ids):
            reason = 'MODEL_REJECTS_ALL_LEGAL_SOURCES'
        else:
            reason = 'RAW_COMPLETE_ENDPOINT_PAIR_RANGE_OR_TYPE_MISMATCH'
        counts[reason] += 1
        traces.append(dict(truth=ti, reason=reason, legal_intervals=legal,
            source_decisions=[dict(states[ci], full_probability=full[ci]) for ci in ids], physical_process_cause='UNKNOWN'))
    matched_predictions = {q['prediction'] for q in score['metrics']['matches']}
    false = []
    for pi, unit in enumerate(score['prediction_groups']):
        if pi in matched_predictions:
            continue
        ids = sorted({q['source_candidate'] for q in unit['members']})
        kind = 'LEGAL_GT_SOURCE_WRONG_INTERVAL_OR_TYPE' if any(ci in legal_sources for ci in ids) else 'NO_FULL_LEGAL_GT_ON_SOURCE'
        false.append(dict(prediction=pi, candidates=ids, kind=kind, physical_process_cause='UNKNOWN'))
    assert len(false) == score['metrics']['FP'] and counts['COMPLETE_GEOMETRY_TP'] == score['metrics']['TP']
    exported = [raw_score.exported(q) for q in frozen['parents']]
    raw_truth = e.read(valid(job['raw_truth'])['path'])
    diagnostics = dict(geometry=raw_score.matching(exported, raw_truth['runs']),
        process=raw_score.matching(exported, raw_truth['runs'], raw_truth))
    turns = [float(v)*180/np.pi for q in exported for v in q['_segment_turns'] if v is not None]
    target = DEST/'review/cases'/sid
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target/'FIELDS.npz', fields=np.concatenate(fields, axis=1),
        offsets=np.r_[0, np.cumsum([q.shape[1] for q in fields])], full=full)
    e.write(target/'TRACE.json', dict(source_id=sid, split=output['split'], counts=dict(counts), traces=traces,
        false_positives=false, source_decisions=states, fields=e.bind(target/'FIELDS.npz'),
        raw_record_diagnostics=diagnostics, CSV=detail.exports(output),
        primitive_representation_normalization=model.last_representation_normalization,
        raw_forward_and_published_output_bit_exact=True, published_parents=len(exported),
        valid_export_parents=sum(q['_valid_export'] for q in exported),
        export_problems=[q['_problems'] for q in exported if q['_problems']],
        maximum_ARC_child_degrees=max(turns) if turns else None))
    return dict(source_id=sid, split=output['split'], counts=dict(counts),
        FP_classes=dict(Counter(q['kind'] for q in false)), raw_record_diagnostics=diagnostics,
        trace=e.bind(target/'TRACE.json'))


def review():
    p = check()
    assert e.read(DEST/'SCORE_ACTUAL_TERMINAL.json')['exit_code'] == 0
    compare()
    outputs = e.read(DEST/'run/OUTPUTS_FROZEN.json')['cases']
    oldtruth = {q['source_id']:q for q in e.read(evaluation.DEST/'TRUTH_INDEX.json')['cases']}
    rawold = {q['source_id']:q for q in e.read(e.REC/'TRUTH_INDEX.json')['cases']}
    newtruth = {q['source_id']:q for q in e.read(new_score.DEST/'TRUTH_INDEX.json')['cases']}
    precise = e.read(e.REC/'family_graph_v67/FIRST_FAILURE_REVIEW.json')['cases']
    legal = {q['source_id']:q['precise_trace'] for q in precise if q['name'].endswith('/graph')}
    for output in outputs:
        sid=output['source_id']
        if sid not in newtruth and sid not in legal:
            legal[sid]=e.bind(e.REC/'round27/run/first_failure'/sid/'TRACE_V63.json')
    jobs = []
    for output in outputs:
        sid = output['source_id']
        additional = sid in newtruth
        if additional:
            q = newtruth[sid]
            job = dict(truth=q['truth'], legal=q['legal_complete'], raw_truth=q['raw_truth'],
                score_path=str(DEST/'additional5_scores'/sid/'SCORE.json'))
        else:
            job = dict(truth=oldtruth[sid]['truth'], legal=legal[sid], raw_truth=rawold[sid]['truth'],
                score_path=str(evaluation.DEST/'scores'/NAME/'original_scope/cases'/sid/'SCORE.json'))
        jobs.append(dict(job, output=output, additional=additional))
    assert len(jobs) == 32
    admission('review')
    memory = int(Path('/sys/fs/cgroup/memory.current').read_text())
    maximum = int(Path('/sys/fs/cgroup/memory.max').read_text())
    quota, period = map(int, Path('/sys/fs/cgroup/cpu.max').read_text().split())
    def usage():
        return int(dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])
    a, begin = usage(), time.monotonic()
    time.sleep(2)
    cpu = (usage()-a)/1e6/(time.monotonic()-begin)
    workers = min(3, int((.9*maximum-memory-p['other_future_reserve_bytes'])/(2*1024**3)),
                  int((.8*quota/period-cpu)/2))
    assert workers >= 1
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(review_one, q) for q in jobs]):
            rows.append(future.result())
            e.write(DEST/'review/PROGRESS.json', dict(complete=len(rows), total=32, workers=workers, epoch=time.time()))
    canonical = {q['canonical_source_id'] for q in e.read(evaluation.DEST/'FROZEN_PROTOCOL.json')['numeric_input_choices']}|set(newtruth)
    first, fp, raw = {}, {}, {}
    for split in ('TRAIN', 'VAL'):
        selected = [q for q in rows if q['split'] == split and q['source_id'] in canonical]
        a, b = Counter(), Counter()
        for q in selected:
            a.update(q['counts'])
            b.update(q['FP_classes'])
        first[split], fp[split] = dict(a), dict(b)
        raw[split] = {kind:{k:sum(q['raw_record_diagnostics'][kind][k] for q in selected)
            for k in ('G', 'P', 'TP', 'FP', 'FN')} for kind in ('geometry', 'process')}
    assert sum(first['TRAIN'].values()) == 121 and sum(first['VAL'].values()) == 37
    e.write(DEST/'review/FIRST_FAILURE_REVIEW.json', dict(complete=True, cases=rows,
        unique_numeric_first_failures=first, unique_numeric_FP_classes=fp,
        raw_record_diagnostics_numeric=raw, all32_forward_raw_count_CSV_replayed=True,
        all_original_fileversions_and_truth_units_retained=True, physical_process_causes_remain_UNKNOWN=True,
        epoch=time.time()))
    e.write(DEST/'review/DIAGNOSTIC_SCHEMA.json',dict(fields=['emitted_range_support_confidence','maximum_executed_start_attention','maximum_executed_end_attention'],not_R32_independent_foreground_or_event_probabilities=True,recurrent_steps_in_source_decisions=True))
    print('ROUND33_ALL32_REVIEW_COMPLETE', first, fp, flush=True)


def visual():
    assert e.read(DEST/'review/FIRST_FAILURE_REVIEW.json')['complete']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = e.read(DEST/'run/RESULT.json')
    cases = {q['source_id']:q for q in result['original_cases']+result['additional_cases']}
    selected = ['curve_0ccbe50eaf3cb217', 'curve_08dcd4a0601b8674', 'curve_e51507fb1de03bf5', 'curve_fe85142e7fb36e8e']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    panels = []
    for ax, sid in zip(axes.flat, selected):
        case = cases[sid]
        path = DEST/'additional5_scores'/sid/'SCORE.json' if sid in {q['source_id'] for q in result['additional_cases']} else evaluation.DEST/'scores'/NAME/'original_scope/cases'/sid/'SCORE.json'
        score = e.read(path)
        truth = e.read(case['truth_binding']['path'])['units']
        predictions = score['prediction_groups']
        points = np.concatenate([np.asarray(q['points']) for q in truth+predictions])
        center = points.mean(0)
        _, _, basis = np.linalg.svd(points-center, full_matrices=False)
        matched = {q['prediction'] for q in score['metrics']['matches']}
        for q in truth:
            xy = (np.asarray(q['points'])-center)@basis[:2].T
            ax.plot(xy[:, 0], xy[:, 1], color='black', linewidth=2.7, alpha=.55)
        for pi, q in enumerate(predictions):
            xy = (np.asarray(q['points'])-center)@basis[:2].T
            ax.plot(xy[:, 0], xy[:, 1], color='royalblue' if pi in matched else 'firebrick', linewidth=1.2)
        metrics = case['metrics']
        ax.set_title(f"{sid}\nR33 {metrics['TP']}/{metrics['G']} FP{metrics['FP']} FN{metrics['FN']}")
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('Display plane / mm')
        ax.set_ylabel('Display plane / mm')
        panels.append(dict(source_id=sid, metrics={k:metrics[k] for k in ('G', 'TP', 'FP', 'FN')}))
    fig.suptitle('R33 published complete curves: black truth, blue TP, red FP\nProjection for display only; all metrics use unchanged strict3D coordinates')
    fig.tight_layout()
    target = OUTPUT/'MY_G9000_AND_FAILURES.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    e.write(DEST/'review/VISUAL.json', dict(figure=e.bind(target), panels=panels,
        actual_main_inspection_required=True, no_scoring_or_pose_or_prediction_change=True))


if __name__ == '__main__':
    dict(prepare=prepare, fit=fit, predict=predict, score=score, review=review, visual=visual)[sys.argv[1]]()
