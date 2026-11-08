"""R29: full22 fit and all32 outputs after the closed V144 family control."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import fcntl
import pickle
import subprocess
import sys
import time
import types
from pathlib import Path
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_section_factorial_v144 as control
from tools import curveonly_01a1ac7c_local_train_v98 as engine
from tools import curveonly_01a1ac7c_structured_family_v116 as family
from tools import curveonly_01a1ac7c_supplement_predictions_v129 as supplement
from tools import curveonly_01a1ac7c_supplement_score_v130 as new_score
from tools import curveonly_01a1ac7c_geometry_eval_v27 as grouping
from tools import curveonly_01a1ac7c_complete_eval_v60 as evaluation
from tools import curveonly_01a1ac7c_score_v2 as raw_score
from weld_pipeline import curveonly_01a1ac7c_section_local_v135 as section
from weld_pipeline import curveonly_01a1ac7c_section_policy_runtime_v146 as runtime
from weld_pipeline import curveonly_01a1ac7c_feature_policy_v144 as policy
from weld_pipeline import curveonly_01a1ac7c_complete_units_v60 as geometry
import numpy as np

NAME = 'round29'
DEST, OUTPUT = e.REC/NAME, e.OUT/NAME
MODULE = 'tools.curveonly_01a1ac7c_section_round_v146'


def prepare():
    control.check(); assert e.read(control.DEST/'CLOSURE.json')['complete']
    decision = e.read(control.DEST/'review_v145/CONTROL_DECISION.json')
    assert decision['analysis_complete'] and decision['visual_actually_inspected']
    assert decision['proceed_full22_research_fit'] and decision['selected_policy']=='new_local'
    assert e.read(control.DEST/'review_v145/RUN_ACTUAL_TERMINAL.json')['exit_code']==0
    original = e.read(e.REC/'round28/INPUTS.json')['cases']
    added = e.read(supplement.DEST/'INPUTS.json')['cases']
    cases = original+added
    assert len(original)==27 and len(added)==5 and len({q['source_id'] for q in cases})==32
    assert sum(q['split']=='TRAIN' for q in cases)==22
    profiles = {}
    for path in [control.recovery.train.DEST/'OTHER19_INPUTS_FROZEN.json', control.recovery.DEST/'pilot/ALL_INPUTS_FROZEN.json', control.recovery.DEST/'val/ALL_INPUTS_FROZEN.json']:
        frozen = e.read(path); assert frozen['complete']
        for profile in frozen['cases']:
            assert profile['source_id'] not in profiles
            profiles[profile['source_id']] = profile
    assert set(profiles)=={q['source_id'] for q in cases}
    DEST.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True); provenance = []
    for case in cases:
        sid = case['source_id']; profile = profiles[sid]; assert profile['split']==case['split']
        for key in ('features', 'original34', 'original_geometry'): control.valid(profile[key])
        folder = OUTPUT/'features'/sid; folder.parent.mkdir(exist_ok=True)
        if case['split']=='TRAIN':
            source = control.OUTPUT/'features'/sid
            if not folder.exists(): folder.symlink_to(source, target_is_directory=True)
            assert folder.resolve()==source.resolve() and (folder/'RECEIPT.json').exists()
        else:
            folder.mkdir(exist_ok=True)
            raw = np.load(profile['original34']['path']); packet = e.read(profile['original_geometry']['path'])
            values = np.load(profile['features']['path'])
            x, packet = section.augment(raw, packet, values, profile['receipt'])
            np.save(folder/'features.npy', x); e.write(folder/'geometry.json', packet)
            e.write(folder/'RECEIPT.json', dict(source_id=sid, split='VAL', features=e.bind(folder/'features.npy'),
                geometry=e.bind(folder/'geometry.json'), raw=case['raw'], STEP=case.get('STEP'), pose=packet['pose'],
                original34_and_all_geometry_preserved=True, labels_models_scores_read=False, cached_input_assembly=True))
        provenance.append(dict(source_id=sid, split=case['split'], original34=profile['original34'],
            section74=profile['features'], geometry=e.bind(folder/'geometry.json'), features=e.bind(folder/'features.npy'),
            material_available=profile['receipt']['available']))
    training = {}
    for fold in e.read(control.DEST/'PREPARED.json')['folds']:
        for row in e.read(control.valid(fold['new_complement'])['path'])['cases']:
            sid = row['source_id']
            if sid in training: assert training[sid]==row
            training[sid] = row
    assert set(training)=={q['source_id'] for q in cases if q['split']=='TRAIN'}
    e.write(DEST/'FIT_TARGETS.json', dict(cases=[training[s] for s in sorted(training)], TRAIN_only=True,
        original_canonical_targets_weights_and_choices_unchanged=True))
    e.write(DEST/'INPUTS.json', dict(cases=cases, TRAIN_inputs=22, VAL_fileversions=10, cached_precision_not_cold=True))
    e.write(DEST/'INPUT_PROVENANCE.json', dict(cases=provenance, no_val_targets_in_features=True,
        unknown_material_TRAIN_inputs_retained=[q['source_id'] for q in provenance if not q['material_available']]))
    commit = subprocess.check_output(['git', 'rev-parse', 'refs/heads/codex/curve-only-01a1ac7c'], text=True).strip()
    for path in [Path(__file__).resolve(), Path(runtime.__file__).resolve()]:
        assert path.read_bytes()==subprocess.check_output(['git', 'show', commit+':'+str(path.relative_to(e.ROOT))])
    sources = [e.bind(q.__file__) for q in (runtime, section, policy, control, engine, family, evaluation, new_score, grouping, geometry, raw_score)]
    sources.append(e.bind(__file__))
    e.write(DEST/'FROZEN_PROTOCOL.json', dict(source_commit=commit, sources=sources,
        decision=e.bind(control.DEST/'review_v145/CONTROL_DECISION.json'), control_closure=e.bind(control.DEST/'CLOSURE.json'),
        inputs=e.bind(DEST/'INPUTS.json'), provenance=e.bind(DEST/'INPUT_PROVENANCE.json'), targets=e.bind(DEST/'FIT_TARGETS.json'),
        baseline_model=e.bind(e.REC/'round27/run/model.pkl'), settings=e.read(control.DEST/'FROZEN_PLAN.json')['settings'],
        selected_policy='new_local', raw_columns=108, local_columns=756,
        scope=dict(TRAIN_inputs=22, TRAIN_G=121, VAL_fileversions=10, VAL_numeric_inputs=9, old4_G=27, new5_G=10),
        single_research_version='Full22 fit of the closed-control new756 local head, original canonical supervision/weights/reversal/HGB and fullTRAIN CTMC prior; direct local proposals, frozen source56/export. Global R27 baseline retained inside for replay only; not selected at inference.',
        no_VALfit_mining_or_threshold_calibration=True, no_mask122=True, no_truth_posefit=True,
        all32_outputs_before_any_new_scores=True, no_best_adoption_or_goal_claim_from_this_tool=True,
        scoring='Original27 with frozenV60; new5 with frozen130 truth units and identical V60 normalization. Report TRAIN/old4/new5/all9 and fileversions separately.',
        reserve_fit_bytes=7*1024**3, reserve_predict_bytes=2*1024**3, other_active_future_reserve_bytes=2*1024**3,
        CPU_threads=2, CPU_fraction_ceiling=.8, memory_fraction_ceiling=.9, epoch=time.time()))
    print('ROUND29_ALL32_PROTOCOL_FROZEN', commit, flush=True)


def check():
    p = e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p[k] for k in ('decision', 'control_closure', 'inputs', 'provenance', 'targets', 'baseline_model')]: control.valid(b)
    return p


def check_sources():
    for b in e.read(DEST/'FROZEN_PROTOCOL.json')['sources']: control.valid(b)


def fit():
    p = check(); cases = e.read(p['targets']['path'])['cases']; assert len(cases)==22
    allowed = Path(p['baseline_model']['path']).resolve()
    val = {q['source_id'] for q in e.read(p['inputs']['path'])['cases'] if q['split']=='VAL'}
    def guard(event, args):
        if event=='open' and isinstance(args[0], (str, bytes, Path)):
            path = Path(os.fsdecode(args[0])); value = str(path)
            if any(k in value for k in ('/score_coordinate_maps/', '/supplement_complete_eval_v130/', '/truth/')) or path.name=='TRUTH_INDEX.json':
                raise PermissionError('Fit cannot read truth/VAL answers: '+value)
            if any(sid in value for sid in val): raise PermissionError('Fit cannot read VAL inputs or targets: '+value)
            if path.name=='model.pkl' and path.resolve()!=allowed and not str(path.resolve()).startswith(str(DEST.resolve())+'/'):
                raise PermissionError('Unrelated model forbidden: '+value)
        e.fence(event, args)
    sys.addaudithook(guard)
    assemble = types.FunctionType(engine.assemble.__code__, dict(engine.__dict__, local=section))
    train = types.FunctionType(engine.fit.__code__, dict(engine.__dict__, local=section, assemble=assemble))
    fitted, record = train(cases, p['settings'], p['baseline_model'], DEST/'local_fit')
    prior, evidence = family.learn_prior(cases)
    base = section.LocalRefinement(fitted.baseline, section.StructuredCurveSet(fitted.local.classifier, prior))
    model = policy.FeaturePolicy(base, 'new756', 'local')
    (DEST/'run').mkdir(exist_ok=True)
    with (DEST/'run/model.pkl').open('wb') as file: pickle.dump(model, file, protocol=5)
    with (DEST/'run/model.pkl').open('rb') as file: restored = pickle.load(file)
    assert restored.model.local.prior==prior and (DEST/'run/model.pkl').stat().st_size<16*1024**2
    e.write(DEST/'run/MODEL_SPEC.json', dict(model=e.bind(DEST/'run/model.pkl'), local_fit=e.bind(DEST/'local_fit/FIT.json'),
        TRAIN_inputs=22, no_VALread=True, raw_columns=108, local_columns=756, prior=prior, prior_evidence=evidence,
        policy='DIRECT_LOCAL_SOURCE56_EXPORT', baseline_not_used_for_prediction=True,
        raw_input_runtime=e.bind(runtime.__file__), original_canonical_labels_weights_reversal_and_settings_unchanged=True))
    print('ROUND29_FULL22_MODEL_FROZEN', e.bind(DEST/'run/model.pkl'), prior, flush=True)


def predict():
    check_sources(); assert e.read(DEST/'FIT_ACTUAL_TERMINAL.json')['exit_code']==0
    control.valid(e.read(DEST/'run/MODEL_SPEC.json')['model'])
    allowed = (DEST/'run/model.pkl').resolve()
    def guard(event, args):
        if event=='open' and isinstance(args[0], (str, bytes, Path)):
            path = Path(os.fsdecode(args[0])); value = str(path)
            if any(k in value for k in ('/truth/', '/targets/', '/first_failure/', '/training_alternatives_v54/', '/score_coordinate_maps/')) or path.name in ('TRUTH_INDEX.json', 'FIT_TARGETS.json'):
                raise PermissionError('Prediction cannot read answers: '+value)
            if path.name=='model.pkl' and path.resolve()!=allowed: raise PermissionError('Unexpected prediction model: '+value)
        e.fence(event, args)
    sys.addaudithook(guard)
    types.FunctionType(e.predict.__code__, dict(e.__dict__, REC=DEST, OUT=OUTPUT, core=runtime, check_source=check_sources))('run')
    frozen = e.read(DEST/'run/OUTPUTS_FROZEN.json'); assert frozen['all_outputs_closed'] and frozen['complete_scope']==32
    assert not any(q['input_failure'] for q in frozen['cases'])
    print('ROUND29_ALL32_PREDICTIONS_CLOSED', flush=True)


def total(rows):
    v = {k:sum(q['metrics'][k] for q in rows) for k in ('G', 'P', 'TP', 'FP', 'FN')}
    v.update(inputs=len(rows), extraction=v['TP']/v['G'] if v['G'] else None, FP_per_G=v['FP']/v['G'] if v['G'] else None)
    return v


def score():
    p = check(); assert e.read(DEST/'PREDICT_ACTUAL_TERMINAL.json')['exit_code']==0
    frozen = e.read(DEST/'run/OUTPUTS_FROZEN.json'); assert frozen['complete_scope']==32 and frozen['all_outputs_closed']
    for case in frozen['cases']:
        control.valid(case['prediction']); control.valid(case['csv'])
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
        normalized = geometry.units(parts); truth = e.read(control.valid(truthmap[sid]['truth'])['path'])['units']
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
    e.write(DEST/'run/RESULT.json', dict(complete=True, round=29, primary_version='V60',
        totals=dict(TRAIN=train, old4_VAL=old4, additional5_VAL=new5, all9_VAL=combined),
        original_scope_result=e.bind(evaluation.DEST/'scores'/name/'RESULT.json'), original_cases=old['cases'], additional_cases=added,
        frozen_outputs=e.bind(DEST/'run/OUTPUTS_FROZEN.json'), model=e.bind(DEST/'run/model.pkl'),
        all32_frozen_before_scores=True, no_VALfit_mining_calibration=True, review_required_before_adoption_or_next_round=True,
        best_round27_not_replaced=True, no_full_contract_or_goal_acceptance=True, epoch=time.time()))
    print('ROUND29_COMPLETE_SCORES', train, old4, new5, combined, flush=True)


def admission(kind):
    p = check(); reserve = p['reserve_fit_bytes'] if kind=='fit' else p['reserve_predict_bytes']
    while True:
        maximum = int(Path('/sys/fs/cgroup/memory.max').read_text()); current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        quota, period = map(int, Path('/sys/fs/cgroup/cpu.max').read_text().split())
        def usage(): return int(dict(q.split() for q in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])
        a = usage(); start = time.monotonic(); time.sleep(2); cpu = (usage()-a)/1e6/(time.monotonic()-start)
        okay = current+reserve+p['other_active_future_reserve_bytes'] < p['memory_fraction_ceiling']*maximum and cpu+2 < p['CPU_fraction_ceiling']*quota/period
        e.write(DEST/'ADMISSION.json', dict(stage=kind, memory=current, maximum=maximum, cpu_cores=cpu, reserve=reserve,
            other_future=p['other_active_future_reserve_bytes'], admitted=okay, epoch=time.time()))
        if okay: return
        time.sleep(10)


def stage(action):
    path = DEST/(action.upper()+'_ACTUAL_TERMINAL.json')
    if path.exists(): assert e.read(path)['exit_code']==0; return
    command = [sys.executable, '-B', '-u', '-m', MODULE, action]; started = time.time()
    with (DEST/(action.upper()+'.log')).open('w') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        e.write(DEST/'FLOW_PROGRESS.json', dict(stage=action, pid=child.pid, epoch=time.time())); code = child.wait()
    e.write(path, dict(exit_code=code, pid=child.pid, actual_subprocess_wait=True, start=started, end=time.time(), command=command))
    assert code==0, (action, code)


def flow():
    with (DEST/'FLOW.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        admission('fit'); stage('fit'); admission('predict'); stage('predict'); admission('score'); stage('score')
        print('ROUND29_NUMERICAL_COMPLETE_FIRST_FAILURE_REVIEW_REQUIRED', flush=True)


if __name__=='__main__':
    dict(prepare=prepare, fit=fit, predict=predict, score=score, flow=flow)[sys.argv[1]]()
