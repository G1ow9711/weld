"""Direct final-output control of the transverse-plate structural exception."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import json
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_native_coverage_control_v296 as shared
from weld_pipeline import curveonly_01a1ac7c_transverse_rules_v338 as rules
from weld_pipeline import curveonly_01a1ac7c_runtime_v51 as exporter
from weld_pipeline import curveonly_01a1ac7c_corner_publication_v256 as corner
from weld_pipeline import curveonly_01a1ac7c_arc_publication_v6 as publication

DEST, OUTPUT = e.REC/'transverse_control_v339', e.OUT/'transverse_control_v339'
QUICK = ('curve_0e430b19e6c300a4', 'curve_dfea4a82bc9967c4', 'curve_e51507fb1de03bf5',
         'curve_e2e843b1da887cf5', 'curve_0ccbe50eaf3cb217', 'curve_74f38ace29077eb1',
         'curve_08dcd4a0601b8674', 'curve_4961ced282d6b548')
valid, full = shared.valid, shared.full


def check():
    p = e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p['inputs'], p['baseline_outputs'], p['baseline_scores']]:
        valid(b)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    commit = subprocess.check_output(['git', 'rev-parse', 'refs/heads/codex/curve-only-01a1ac7c'], text=True).strip()
    sources = [Path(__file__).resolve(), Path(rules.__file__).resolve()]
    for s in sources:
        assert s.read_bytes() == subprocess.check_output(['git', 'show', commit+':'+str(s.relative_to(e.ROOT))])
    jobs = e.read(e.REC/'native_coverage_control_v296/INPUTS.json')['cases']
    jobs = [dict(c, prepared=e.bind(e.OUT/'event_cut_full_v309/node_inputs'/c['source_id']/'RECEIPT.json')) for c in jobs]
    DEST.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json', dict(cases=jobs, no_labels_or_score_answers=True))
    e.write(DEST/'FROZEN_PROTOCOL.json', dict(source_commit=commit,
        sources=[e.bind(s) for s in sources]+e.read(e.REC/'native_coverage_control_v296/FROZEN_PROTOCOL.json')['sources'],
        inputs=e.bind(DEST/'INPUTS.json'), baseline_outputs=e.bind(e.REC/'round35/run/OUTPUTS_FROZEN.json'),
        baseline_scores=e.bind(e.REC/'round35/run/RESULT.json'),
        frozen_R35_model=e.bind(e.REC/'round35/run/model.pkl'),
        preceding_failed_model_closed=e.bind(e.REC/'refined_range_full_v334/CLOSURE.json'),
        user_request=e.bind(e.REC/'USER_RULE_INTEGRATION_REQUEST_SIDE_01a1b45d_20261107.md'),
        design=e.bind(e.REC/'TARGETED_STRUCTURE_RULE_CHANGE_V338.md'),
        no_new_model_fit_or_negative_mining_or_threshold_calibration=True,
        no_validation_answers_in_rule_forward=True, same_program_for_all_inputs=True,
        cached_R35_executed_states_replay_exact_required=True, cached_precision_not_cold=True,
        current_geometry_3mm_real_gaps_and_ARC30_unchanged=True,
        quick_ids=list(QUICK), quick_is_exposed_TRAIN_and_iterative_VAL=True,
        full_scope=dict(TRAIN_inputs=22, TRAIN_G=121, VAL_numeric=9, VAL_G=37, VAL_fileversions=10, VAL_fileversion_G=48),
        time_optimization_canceled=True, epoch=time.time()))
    print('TRANSVERSE_RULE_PROTOCOL_FROZEN', commit, flush=True)


class Fixed:
    def __init__(self, proposals):
        self.proposals = proposals
        self.last_timing = dict(cached_R35_plus_geometry_rule=True)

    def predict(self, features, packet):
        return self.proposals


def publish(raw, x, packet):
    selected, selection = rules.provenance.select(raw, packet)
    old = publication._fallback
    publication._fallback = corner.fallback
    try:
        decoded = exporter.decode(x, packet, Fixed(selected))
    finally:
        publication._fallback = old
    decoded['finite_source_selection'] = selection
    return decoded


def one(job, phase):
    p = check(); sid = job['source_id']
    def fence(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            path = Path(os.fsdecode(args[0])); s = str(path)
            if any(t in s for t in ('/truth/', '/targets/', '/scores/', '/review/', '/score_coordinate_maps/')):
                raise PermissionError('Structural rule forward cannot read answers '+s)
            if path.name in ('FIT_TARGETS.json', 'RESULT.json', 'SCORE.json', 'model.pkl'):
                raise PermissionError('Structural rule forward cannot read labels, scores or another model '+s)
        e.fence(event, args)
    sys.addaudithook(fence)
    x = np.load(valid(job['features'])['path']); packet = e.read(valid(job['geometry'])['path'])
    prepared = e.read(valid(job['prepared'])['path']); forward = e.read(valid(prepared['original_forward'])['path'])
    assert forward['same_R35_actual_neural_states_and_raw_normalization_exact']
    assert forward['model']['sha256'] == p['frozen_R35_model']['sha256']
    rows = rules.prior.candidate_rows(x, packet)
    raw, receipt = rules.prior.normalize(rules.prior.codec.proposals(packet, rows,
        [s['codes'] for s in forward['source_decisions']]), packet)
    assert receipt == forward['primitive_representation_normalization']
    replay = publish(raw, x, packet)
    baseline = e.read(valid(job['baseline_prediction'])['path'])
    assert full.detail.equal(replay['parents'], baseline['parents'])
    if job['STEP'] and packet.get('pose', {}).get('passed'):
        solids = e.core.load_material(valid(job['STEP'])['path'])
        inventory = e.read(valid(prepared['inventory'])['path'])
        repaired, audit = rules.apply(raw, x, packet, inventory, solids)
        decoded = publish(repaired, x, packet)
    else:
        decoded = replay
        audit = dict(not_applicable='NO_QUALIFIED_FINITE_STEP', no_input_removed=True)
    decoded['structural_rule_v338'] = audit
    target = OUTPUT/phase/'outputs'/sid; target.mkdir(parents=True, exist_ok=True)
    e.write(target/'curves.json', decoded); shared.write_csv(target/'curves.csv', sid, decoded)
    e.write(target/'RULE_TRACE.json', dict(source_id=sid, actual_R35_parent_replay_exact=True,
        pure_geometry_rule=audit, input=job, rule_sources=p['sources'][:2], epoch=time.time()))
    result = dict(source_id=sid, split=job['split'], prediction=e.bind(target/'curves.json'),
        csv=e.bind(target/'curves.csv'), trace=e.bind(target/'RULE_TRACE.json'), input_failure=False)
    print('TRANSVERSE_OUTPUT_FROZEN', phase, sid, 'P', decoded['geometry_units'],
        'rule', audit.get('chosen_candidates', []), 'new', audit.get('new_rule_proposals', 0), flush=True)
    return result


def predict(phase):
    p = check(); assert phase in ('quick', 'full')
    target = DEST/phase; target.mkdir(exist_ok=True)
    assert not (target/'OUTPUTS_FROZEN.json').exists()
    if phase == 'full':
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full']
    jobs = e.read(p['inputs']['path'])['cases']
    if phase == 'quick':
        jobs = [c for c in jobs if c['source_id'] in QUICK]
    resource = shared.resources.resources()
    workers = min(3, int((.9*resource['maximum']-resource['memory']-1024**3)/(2*1024**3)),
        int((.8*resource['quota']-resource['cpu'])/2))
    assert workers >= 1, resource
    outputs = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(one, c, phase) for c in jobs]):
            outputs.append(future.result())
    outputs.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json', dict(cases=outputs, all_outputs_closed_before_scores=True,
        cached_precision_not_cold=True, workers=workers, resource_start=resource,
        protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'), epoch=time.time()))
    print('ALL_STRUCTURAL_OUTPUTS_FROZEN', phase, len(outputs), flush=True)


def score(phase):
    p = check(); target = DEST/phase
    frozen = e.read(target/'OUTPUTS_FROZEN.json')
    assert frozen['all_outputs_closed_before_scores']
    baseline = e.read(p['baseline_scores']['path'])
    old = {c['source_id']:c for c in baseline['original_cases']+baseline['additional_cases']}
    inputs = {c['source_id']:c for c in e.read(p['inputs']['path'])['cases']}
    outputs = []
    for output in frozen['cases']:
        sid = output['source_id']; reference = old[sid]
        decoded = e.read(valid(output['prediction'])['path'])
        packet = e.read(valid(inputs[sid]['geometry'])['path'])
        truths = e.read(valid(reference['truth_binding'])['path'])['units']
        curves = [full.raw_score.exported(c) for c in decoded['parents']]
        assert all(c['_valid_export'] for c in curves)
        chains, groups = full.grouping.source_groups(packet)
        parts = [full.evaluation.annotate(c, groups[chains[c['source_candidate']]],
            chains[c['source_candidate']], packet['candidates'][c['source_candidate']].get('material'),
            c['source_candidate']) for c in curves]
        units = full.geometry.units(parts); metrics = full.grouping.matching(units, truths)
        before = {m['truth'] for m in reference['metrics']['matches']}
        after = {m['truth'] for m in metrics['matches']}
        outputs.append(dict(source_id=sid, split=inputs[sid]['split'], metrics=metrics,
            gained=sorted(after-before), lost=sorted(before-after), baseline=reference['metrics'],
            prediction=output['prediction'], truth_binding=reference['truth_binding']))
        print('STRUCTURAL_FINAL_SCORE', sid, {k:metrics[k] for k in ('G', 'TP', 'FP', 'FN')},
            'gain', sorted(after-before), 'lost', sorted(before-after), flush=True)
    numeric = [q for q in outputs if q['source_id'] != 'curve_b3d2a4b469e93565']
    totals = {split:shared.total([q for q in numeric if q['split'] == split]) for split in ('TRAIN', 'VAL')}
    e.write(target/'RESULT.json', dict(complete=True, same_primary_version='V60', cases=outputs,
        totals=totals, VAL_fileversions=shared.total([q for q in outputs if q['split'] == 'VAL']),
        baseline='R35 retained', same3mm_denominators_real_gaps=True,
        no_new_model_fit_or_VALcalibration=True, no_contract_acceptance=True, epoch=time.time()))
    print('STRUCTURAL_SCOPE_TOTALS', totals, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        {'predict':predict, 'score':score}[sys.argv[1]](sys.argv[2])
