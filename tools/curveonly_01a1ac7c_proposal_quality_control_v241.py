"""TRAIN-only family-held quality control on the frozen R33 proposal generator."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import csv
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_ordered_span_full_v238 as full
from weld_pipeline import curveonly_01a1ac7c_proposal_quality_v240 as quality

DEST = e.REC/'proposal_quality_control_v241'
OUTPUT = e.OUT/'proposal_quality_control_v241'
SETTINGS = dict(max_iter=128, learning_rate=.06, max_leaf_nodes=7,
    l2_regularization=.3, min_samples_leaf=8, max_bins=63,
    early_stopping=False, random_state=20261106)


def valid(binding):
    assert e.sha(binding['path']) == binding['sha256'], binding['path']
    return binding


def source_check():
    p = e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p['baseline_model'], p['baseline_closure'], p['targets'], p['inputs']]:
        valid(b)
    return p


def prepare():
    assert e.read(full.DEST/'run/CLOSURE.json')['complete']
    DEST.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'refs/heads/codex/curve-only-01a1ac7c'], text=True).strip()
    modules = (quality, sys.modules[__name__])
    for module in modules:
        p = Path(module.__file__).resolve()
        assert p.read_bytes() == subprocess.check_output(['git', 'show', commit+':'+str(p.relative_to(e.ROOT))])
    inputs = e.read(full.DEST/'INPUTS.json')['cases']
    families = sorted({c['family'] for c in inputs if c['split'] == 'TRAIN'})
    assert len(families) == 18
    np.random.default_rng(20261106).shuffle(families)
    fold = {f:i % 4 for i, f in enumerate(families)}
    p = dict(source_commit=commit, sources=e.read(full.DEST/'FROZEN_PROTOCOL.json')['sources']+[e.bind(m.__file__) for m in modules],
        baseline_model=e.bind(full.DEST/'run/model.pkl'), baseline_closure=e.bind(full.DEST/'run/CLOSURE.json'),
        targets=e.bind(full.DEST/'FIT_TARGETS.json'), inputs=e.bind(full.DEST/'INPUTS.json'),
        families=fold, settings=SETTINGS, features=quality.NAMES, modes=list(quality.MODES),
        design='Freeze R33 generator, geometry, primitive normalization, overlap policy and export. Learn strict complete generated-proposal validity from input profiles, finite provenance and actual pointer/continue decisions; compare rank-only, gate-only and rank+gate using4family-held quality heads.',
        closest_history=[dict(source='v178/v179 complete interval anchor quality',
            distinction='Old dense centerness quality was trained at native anchors jointly with interval generation. New head learns complete3D validity of actual autoregressively generated spans; generator stays frozen.'),
            dict(source='R25 representation-first-failure/R26 complete units',
            distinction='Earlier full-circle source competition was investigated with a unit correction; no equivalent complete-proposal validity head in this curve-only branch. No unit correction here.')],
        labels='Any TRAIN completeV60 truth with matching type/known side, closure, both outer endpoints and whole3D curve<=3mm. All geometrically equivalent raw alternatives may be positive; final one-to-one matching and grouping unchanged.',
        weighting='EachTRAINcase equal mass; observed positive and negative groups equal mass percase. Positive mass equally across represented truth and its valid proposal alternatives, negative mass equally across generated invalid proposals.',
        decision='Choose largest actual final-output family-held F1, tie TP then lowerFP; ties favor baseline. Proceed to full22 quality fit/all32 frozen outputs only if a mode strictly improves F1 and improves TP or FP. Report all losses/gains and categories. This is a research criterion, not final acceptance.',
        folds_conditioned_on_generator_seen_all22TRAIN=True, not_end_to_end_unseen_family_validation=True,
        no_VALread_fit_mining_or_calibration=True, no_added_minimum_length_or_gap=True,
        no_new_geometry_or_overlap_tolerance=True, epoch=time.time())
    e.write(DEST/'FROZEN_PROTOCOL.json', p)
    print('QUALITY_CONTROL_PROTOCOL', commit, len(quality.NAMES), flush=True)


def training_fence():
    inputs = e.read(full.DEST/'INPUTS.json')['cases']
    val = {c['source_id'] for c in inputs if c['split'] == 'VAL'}
    def guard(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            s = os.fsdecode(args[0])
            if any(sid in s for sid in val) or '/score_coordinate_maps/' in s or '/supplement_complete_eval_v130/' in s:
                raise PermissionError('QualityTRAIN cannot readVAL: '+s)
        e.fence(event, args)
    sys.addaudithook(guard)


def load_case(case):
    sid = case['source_id']
    tracepath = full.DEST/'review/cases'/sid/'TRACE.json'
    trace = e.read(tracepath)
    observed = np.load(valid(case['features'])['path'])
    packet = e.read(valid(case['geometry'])['path'])
    rows = quality.generator.candidate_rows(observed, packet)
    raw = quality.generator.normalize(quality.generator.codec.proposals(packet, rows,
        [q['codes'] for q in trace['source_decisions']]), packet)[0]
    original_path = full.OUTPUT/'run/outputs'/sid/'curves.json'
    original = e.read(original_path)
    decoded = quality.decode(observed, packet, quality.StaticProposals(raw))
    assert full.detail.equal(decoded['parents'], original['parents'])
    assert decoded['finite_source_selection']['kept'] == original['finite_source_selection']['kept']
    assert len(raw) == original['finite_source_selection']['input_proposals']
    x = quality.features(rows, trace['source_decisions'], raw, packet)
    return observed, packet, raw, x


def dataset():
    p = source_check(); training_fence()
    cases = e.read(p['targets']['path'])['cases']
    assert len(cases) == 22 and all(c['split'] == 'TRAIN' for c in cases)
    families = {c['source_id']:c['family'] for c in e.read(p['inputs']['path'])['cases'] if c['split'] == 'TRAIN'}
    records, matrices, labels, weights, folds = [], [], [], [], []
    for case in sorted(cases, key=lambda c:c['source_id']):
        sid = case['source_id']
        _, packet, raw, x = load_case(case)
        truths = e.read(valid(case['truth'])['path'])['units']
        matches = [[ti for ti, t in enumerate(truths)
            if (t.get('weld_side') is None or q.get('weld_side') == t['weld_side'])
            and full.raw_score.geometry_error(q, t) is not None] for q in raw]
        assert all(len(m) <= 1 for m in matches), 'Ambiguous complete-proposal labels need explicit audit'
        y = np.array([bool(m) for m in matches], np.int64)
        mass = np.zeros(len(y))
        counts = Counter(m[0] for m in matches if m)
        groups = len(set(y.tolist()))
        for i, options in enumerate(matches):
            mass[i] = 1/(groups*len(counts)*counts[options[0]]) if options else 1/(groups*int((y == 0).sum()))
        if len(y):
            assert abs(mass.sum()-1) < 1e-10
        data_path = OUTPUT/'dataset'/sid/'DATA.npz'; data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(data_path, x=x, y=y, weight=mass)
        fold = p['families'][families[sid]]
        records.append(dict(source_id=sid, family=families[sid], fold=fold,
            G=len(truths), proposals=len(raw), positive=int(y.sum()), negative=int((1-y).sum()),
            represented_truth=sorted(counts), labels=matches, data=e.bind(data_path),
            target=case, baseline_all_parents_and_selection_replay_exact=True))
        matrices.append(x); labels.append(y); weights.append(mass); folds.append(np.full(len(y), fold, np.int64))
        print('QUALITY_TRAIN_DATASET', sid, len(raw), int(y.sum()), len(truths), flush=True)
    path = OUTPUT/'DATASET.npz'
    np.savez_compressed(path, x=np.vstack(matrices), y=np.concatenate(labels),
        weight=np.concatenate(weights), fold=np.concatenate(folds),
        offsets=np.r_[0, np.cumsum([len(x) for x in matrices])])
    e.write(DEST/'DATASET.json', dict(complete=True, cases=records, arrays=e.bind(path),
        features=len(quality.NAMES), TRAIN_only=True, all22_baseline_final_outputs_reproduced=True,
        missing_true_units_preserved_in_final_score=True, no_VALread=True, epoch=time.time()))


def publish(target, sid, decoded):
    e.write(target/'curves.json', decoded)
    fields = ['workpiece_id','weld_id','parent_id','segment_index','weld_type','geometry_type','is_closed',
        'start_x','start_y','start_z','end_x','end_y','end_z','length_mm','confidence','curve_points_file','status']
    with (target/'curves.csv').open('w', newline='', encoding='utf8') as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader()
        for parent in decoded['parents']:
            for segment in parent['segments']:
                a, b = segment['points'][0], segment['points'][-1]
                writer.writerow(dict(workpiece_id=sid, weld_id=parent['parent_id']+'_'+str(segment['segment_index']),
                    parent_id=parent['parent_id'], segment_index=segment['segment_index'], weld_type=parent['weld_type'],
                    geometry_type=segment['geometry_type'], is_closed=parent['is_closed'], start_x=a[0], start_y=a[1], start_z=a[2],
                    end_x=b[0], end_y=b[1], end_z=b[2], length_mm=segment['length_mm'], confidence=parent['confidence'],
                    curve_points_file='curves.json', status=segment['status']))
    return dict(source_id=sid, prediction=e.bind(target/'curves.json'), csv=e.bind(target/'curves.csv'))


def cv():
    p = source_check(); training_fence(); start = time.perf_counter()
    data = e.read(DEST/'DATASET.json'); arrays = np.load(valid(data['arrays'])['path'])
    x, y, weight, fold = [arrays[k] for k in ('x', 'y', 'weight', 'fold')]
    probability = np.zeros(len(y)); fitted = []
    for held in range(4):
        selected = fold != held
        assert set(y[selected]) == {0, 1}
        classifier = HistGradientBoostingClassifier(**SETTINGS)
        with threadpool_limits(limits=2):
            classifier.fit(x[selected], y[selected], sample_weight=weight[selected])
            probability[~selected] = classifier.predict_proba(x[~selected])[:, 1]
        path = DEST/'folds'/str(held)/'quality.pkl'; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as file:
            pickle.dump(classifier, file, protocol=5)
        with path.open('rb') as file:
            restored = pickle.load(file)
        with threadpool_limits(limits=2):
            assert np.array_equal(probability[~selected], restored.predict_proba(x[~selected])[:, 1])
        fitted.append(dict(held_fold=held, model=e.bind(path), train_rows=int(selected.sum()), held_rows=int((~selected).sum()),
            held_families=sorted(f for f, fi in p['families'].items() if fi == held)))
        print('QUALITY_FAMILY_HELD_FIT', held, int(selected.sum()), int((~selected).sum()), flush=True)
    path = OUTPUT/'HELD_PROBABILITY.npy'; np.save(path, probability)
    outputs = {mode:[] for mode in quality.MODES}
    for i, case in enumerate(data['cases']):
        sid = case['source_id']; observed, packet, raw, current = load_case(case['target'])
        lo, hi = arrays['offsets'][i:i+2]
        assert np.array_equal(current, x[lo:hi])
        for mode in quality.MODES:
            proposals, decisions = quality.apply(raw, probability[lo:hi], mode)
            decoded = quality.decode(observed, packet, quality.StaticProposals(proposals))
            decoded['complete_proposal_quality'] = decisions
            row = publish(OUTPUT/'cv'/mode/sid, sid, decoded)
            outputs[mode].append(dict(row, truth=case['target']['truth']))
        print('QUALITY_HELD_FINAL_OUTPUTS', sid, flush=True)
    e.write(DEST/'OUTPUTS_FROZEN.json', dict(complete=True, outputs=outputs,
        full_scope_each_mode=22, folds=fitted, probabilities=e.bind(path),
        all_modes_frozen_before_final_scoring=True, generator_seen_all22TRAIN=True,
        no_end_to_end_unseen_family_claim=True, elapsed_seconds=time.perf_counter()-start))


def score_one(case):
    sid = case['source_id']
    packet = e.read(full.OUTPUT/'features'/sid/'geometry.json')
    decoded = e.read(valid(case['prediction'])['path']); valid(case['csv'])
    exported = [full.raw_score.exported(q) for q in decoded['parents']]
    assert all(q['_valid_export'] for q in exported)
    chains, groups = full.grouping.source_groups(packet)
    parts = []
    for curve in exported:
        ci = curve['source_candidate']; gi = chains[ci]
        parts.append(full.evaluation.annotate(curve, groups[gi], gi, packet['candidates'][ci].get('material'), ci))
    units = full.geometry.units(parts)
    truth = e.read(valid(case['truth'])['path'])['units']
    metrics = full.grouping.matching(units, truth)
    closed = lambda q:np.linalg.norm(np.asarray(q['points'])[0]-np.asarray(q['points'])[-1]) < .1
    categories = {}
    for name, status in [('closed', True), ('open', False)]:
        categories[name] = full.grouping.matching([q for q in units if closed(q) == status],
            [q for q in truth if closed(q) == status])
    return dict(source_id=sid, metrics=metrics, categories=categories)


def score():
    source_check(); training_fence()
    frozen = e.read(DEST/'OUTPUTS_FROZEN.json'); assert frozen['all_modes_frozen_before_final_scoring']
    baseline = [c for c in e.read(full.DEST/'run/RESULT.json')['original_cases'] if c['split'] == 'TRAIN']
    before = {(c['source_id'], m['truth']) for c in baseline for m in c['metrics']['matches']}
    totals, rows = {}, {}
    for mode, cases in frozen['outputs'].items():
        assert len(cases) == 22
        result = []
        for c in cases:
            result.append(score_one(c))
            print('QUALITY_HELD_SCORE', mode, c['source_id'], result[-1]['metrics']['TP'], result[-1]['metrics']['FP'], flush=True)
        rows[mode] = result; totals[mode] = full.total(result)
    totals['baseline'] = full.total(baseline)
    for value in totals.values():
        value['F1'] = 2*value['TP']/(value['G']+value['P'])
    order = ['baseline']+list(quality.MODES)
    best = max(order, key=lambda mode:(totals[mode]['F1'], totals[mode]['TP'], -totals[mode]['FP']))
    changes = {}
    for mode, result in rows.items():
        after = {(c['source_id'], m['truth']) for c in result for m in c['metrics']['matches']}
        changes[mode] = dict(gained=sorted(after-before), lost=sorted(before-after),
            categories={name:full.total([dict(metrics=c['categories'][name]) for c in result]) for name in ('closed','open')})
    proceed = best != 'baseline' and totals[best]['F1'] > totals['baseline']['F1'] and (
        totals[best]['TP'] > totals['baseline']['TP'] or totals[best]['FP'] < totals['baseline']['FP'])
    e.write(DEST/'RESULT.json', dict(complete=True, totals=totals, cases=rows, changes=changes,
        selected_mode=best, proceed_full22_fit_and_all32_outputs=proceed,
        generator_seen_allTRAIN=True, only_quality_head_family_held=True,
        no_VALread_fit_mining_or_threshold_calibration=True, no_full_model_or_contract_pass=True,
        source=e.bind(__file__), frozen_outputs=e.bind(DEST/'OUTPUTS_FROZEN.json'), epoch=time.time()))
    print('QUALITY_HELD_DECISION', best, proceed, totals, flush=True)


if __name__ == '__main__':
    action = sys.argv[1]
    with threadpool_limits(limits=2):
        dict(prepare=prepare, dataset=dataset, cv=cv, score=score)[action]()
