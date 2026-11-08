"""Postfreeze additional validation, reusing the frozen V60 curve unit policy."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import multiprocessing
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_supplement_predictions_v129 as prediction
from tools import curveonly_01a1ac7c_complete_eval_v60 as frozen_eval
from tools import curveonly_01a1ac7c_complete_eval_v36 as base_eval
from tools import curveonly_01a1ac7c_geometry_eval_v27 as grouping
from tools import curveonly_01a1ac7c_targets_v14 as target
from tools import curveonly_01a1ac7c_score_v2 as raw_score
from weld_pipeline import curveonly_01a1ac7c_geometry_units_v26 as legacy_geometry
from weld_pipeline import curveonly_01a1ac7c_complete_units_v60 as geometry

R, O = e.REC, e.OUT
DEST = R / 'supplement_complete_eval_v130'
DATASET = Path('/root/autodl-tmp/weld_cloud/data/validation_only/supplement_20261106')


def prepare():
    closed = e.read(prediction.DEST / 'ALL_OUTPUTS_FROZEN.json')
    release = e.read(prediction.DEST / 'SCORING_ANSWER_RELEASE.json')
    assert closed['complete'] and closed['all10_predictions_and_CSV_closed'] and release['allowed_after_all10_frozen']
    assert e.read(prediction.DEST / 'OUTPUTS_ACTUAL_TERMINAL.json')['exit_code'] == 0
    old_inventory = e.read(R / 'supplement_curve_inventory_v108/RESULT.json')
    assert e.sha(DATASET / 'dataset_manifest.json') == old_inventory['manifest']['sha256']
    manifest = e.read(DATASET / 'dataset_manifest.json')
    inputs = {q['key']: q for q in e.read(prediction.DEST / 'INPUTS.json')['cases']}
    cases = []
    for row in manifest['cases']:
        case = inputs[row['key']]
        assert row['training_allowed'] is False and row['negative_mining_or_threshold_calibration_allowed'] is False
        folder = Path(row['truth']['path']).parent
        excel = list(folder.glob('*_weld.xlsx')); assert len(excel) == 1
        cases.append(dict(source_id=case['source_id'], key=row['key'], name=row['name'], split='VAL',
            label=row['truth'], excel=[e.bind(excel[0])], coordinate_map=e.bind(row['coordinate_map']),
            raw_geometry=e.bind(prediction.OUTPUT / 'features' / case['source_id'] / 'geometry.json'),
            expected_raw_curve_children=row['curve_segments_raw_export'],
            input_label_CAD_byte_identical=row['STEP_byte_identical'], CAD_revision_note=row.get('CAD_revision_note')))
    DEST.mkdir(exist_ok=True)
    e.write(DEST / 'CASES.json', dict(cases=cases, post_prediction_freeze_only=True))
    commit = subprocess.check_output(['git', 'rev-parse', 'refs/heads/codex/curve-only-01a1ac7c'], text=True).strip()
    assert Path(__file__).read_bytes() == subprocess.check_output(['git', 'show', commit + ':' + str(Path(__file__).resolve().relative_to(e.ROOT))])
    e.write(DEST / 'FROZEN_PROTOCOL.json', dict(
        name='FROZEN_V60_ADDITIONAL5_POSTFREEZE_SCORE_V130', source_commit=commit,
        sources=[e.bind(__file__), e.bind(e.__file__), e.bind(frozen_eval.__file__), e.bind(base_eval.__file__),
                 e.bind(grouping.__file__), e.bind(target.__file__), e.bind(raw_score.__file__),
                 e.bind(legacy_geometry.__file__), e.bind(geometry.__file__)],
        cases=e.bind(DEST / 'CASES.json'), predictions=e.bind(prediction.DEST / 'ALL_OUTPUTS_FROZEN.json'),
        answer_release=e.bind(prediction.DEST / 'SCORING_ANSWER_RELEASE.json'), dataset_manifest=e.bind(DATASET / 'dataset_manifest.json'),
        previous_old4_scores={name:e.bind(frozen_eval.DEST / 'scores' / name / 'RESULT.json') for name in ('round27','round28')},
        primary_policy=geometry.POLICY, version=geometry.VERSION, endpoint_and_whole_mm=3., sampling_mm=1., ARCchild_max_deg=30.,
        normalizer='Existing truth_record curve runs/children and ExcelUUID types -> frozen geometry26 units using input-only chains -> V60 complete units using independently legal native paths. Preserve all original curves, points, gaps, cuts and owners.',
        witness='Same common frozen129 input geometry for both models; candidate choice for GT witness uses only finite geometry, never model probabilities or predictions. Predictions normalized separately without reading GT.',
        coordinate_rule='Only invert the supplied unlabelled-material-derived source_point_to_truth_frame for GT points; never apply exportToWorld again, fit to predictions, or alter input pose/outputs. Provided map is not customer-authenticated pose metadata.',
        no_train_mining_threshold_or_model_selection=True, no_old_GTunits_or_scores_changed=True,
        old4_and_new5_separate_reports=True, family_independence_unproved=True, no_blind_or_contract_acceptance_claim=True,
        maximum_workers=2, reserve_per_worker_bytes=1024**3, CPU_fraction_ceiling=.8, memory_fraction_ceiling=.9,
        epoch=time.time()))
    print('SUPPLEMENT_SCORE_PROTOCOL_FROZEN', commit, flush=True)


def check():
    p = e.read(DEST / 'FROZEN_PROTOCOL.json')
    for b in p['sources'] + [p['cases'], p['predictions'], p['answer_release'], p['dataset_manifest']]:
        assert e.sha(b['path']) == b['sha256']
    return p


def controls():
    check()
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
    translation = np.array([123., -456., 78.])
    points = np.array([[0., 0., 0.], [100., 0., 0.], [30., 40., -20.]])
    provided = points @ rotation.T + translation
    restored = (provided - translation) @ rotation
    assert np.allclose(restored, points, atol=1e-10, rtol=0)
    assert e.read(frozen_eval.DEST / 'CONTROLS.json')['passed']
    e.write(DEST / 'CONTROLS.json', dict(passed=True, rigid_inverse_scoring_transform=True,
        frozen_V60_geometry_controls_reused=e.bind(frozen_eval.DEST / 'CONTROLS.json'),
        no_prediction_transform_fitted=True, no_new_geometry_tolerance=True))
    print('SUPPLEMENT_SCORE_CONTROLS', flush=True)


def legal_options(curves, packet):
    result = []
    for ci, candidate in enumerate(packet['candidates']):
        result.extend(dict(candidate=ci, **q) for q in target.legal_intervals(candidate, dict(runs=curves)))
    return result


def truth_one(case):
    check()

    def no_predictions(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, Path)):
            path = Path(os.fsdecode(args[0])); value = str(path)
            if path.name == 'model.pkl' or ('/outputs/' in value and path.name in ('curves.json','curves.csv')):
                raise PermissionError('Truth unit normalization cannot read model or predictions: ' + value)

    sys.addaudithook(no_predictions)
    for key in ('label', 'coordinate_map', 'raw_geometry'):
        b = case[key]; assert e.sha(b['path']) == b['sha256']
    mapping = e.read(case['coordinate_map']['path'])
    assert mapping['derived_from_unlabelled_material_only'] and not mapping['weld_labels_used_for_pose_fit']
    assert mapping['inference_must_not_load_truth_or_this_fit_as_material_pose']
    transform = np.asarray(mapping['source_point_to_truth_frame'], float)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    assert transform.shape == (4,4) and np.isfinite(transform).all()
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0) and abs(np.linalg.det(rotation)-1) <= 1e-8
    truth = e.truth_record(case)
    assert len(truth['children']) == case['expected_raw_curve_children']
    maximum_roundtrip = 0.
    for curve in truth['runs'] + truth['children']:
        points = np.asarray(curve['points']); raw = (points - translation) @ rotation
        maximum_roundtrip = max(maximum_roundtrip, float(np.linalg.norm(raw @ rotation.T + translation - points,axis=1).max()))
        curve['points'] = raw.tolist(); curve['circle'] = e.core.circle_fit(raw)
        curve['closed'] = bool(np.linalg.norm(raw[0]-raw[-1]) < .1)
    assert maximum_roundtrip <= 1e-7
    sid = case['source_id']; folder = DEST / 'truth' / sid
    e.write(folder / 'RAW_CURVE_TRUTH.json', dict(truth, original_coordinate_map=case['coordinate_map'],
            offline_scoring_only=True, inverse_map_applied_once=True, original_exportToWorld_not_applied=True))
    packet = e.read(case['raw_geometry']['path'])
    raw_legal = legal_options(truth['runs'], packet)
    assigned = {}
    for ti in range(len(truth['runs'])):
        options = [q for q in raw_legal if q['truth'] == ti]
        if options:
            assigned[ti] = min(options, key=lambda q:(sum(q['error'].values()), q['candidate']))['candidate']
    annotated, _ = grouping.annotate(truth['runs'], packet, assigned)
    previous = legacy_geometry.units(annotated)
    assert sum(len(q['members']) for q in previous) == len(truth['runs'])
    whole_legal = legal_options(previous, packet)
    parts = []
    for ti, curve in enumerate(previous):
        options = [q for q in whole_legal if q['truth'] == ti]
        if options:
            q = min(options, key=lambda r:(sum(r['error'].values()),r['candidate']))
            ci = q['candidate']; native = packet['candidates'][ci]
            parts.append(frozen_eval.annotate(curve, native, ci, native.get('material'), ci, q))
        else:
            parts.append(dict(curve, _carrier=None, _material=None))
    units = geometry.units(parts)
    assert sum(len(m['original_members']) for q in units for m in q['members']) == len(truth['runs'])
    e.write(folder / 'COMPLETE_UNITS.json', dict(source_id=sid, split='VAL', units=units,
        raw_curve_children=len(truth['children']), raw_record_runs=len(truth['runs']),
        previous_geometry_units=len(previous), complete_units=len(units),
        source_label=case['label'], Excel=case['excel'], coordinate_map=case['coordinate_map'],
        original_curve_points_and_all_internal_boundaries_retained=True,
        original_records_accounted=True, witness=case['raw_geometry'],
        inverse_coordinate_roundtrip_error_mm=maximum_roundtrip, process_side_direction_attributes_unaccepted=True))
    e.write(folder / 'LEGAL_RAW_INTERVALS.json', dict(options=raw_legal, no_model_decisions_used=True))
    complete_legal = legal_options(units, packet)
    e.write(folder / 'LEGAL_COMPLETE_INTERVALS.json', dict(options=complete_legal, no_model_decisions_used=True))
    row = dict(source_id=sid,name=case['name'],split='VAL',truth=e.bind(folder/'COMPLETE_UNITS.json'),
        raw_truth=e.bind(folder/'RAW_CURVE_TRUTH.json'),legal_complete=e.bind(folder/'LEGAL_COMPLETE_INTERVALS.json'),
        G=len(units),raw_records=len(truth['runs']),raw_children=len(truth['children']),
        represented_complete_units=len({q['truth'] for q in complete_legal}),
        all_units_kept_when_source_missing=True)
    print('SUPPLEMENT_COMPLETE_TRUTH', case['name'], row['G'], row['raw_records'], row['raw_children'],flush=True)
    return row


def truths():
    p=check(); assert e.read(DEST/'CONTROLS.json')['passed']
    memory=int(Path('/sys/fs/cgroup/memory.current').read_text());limit=int(Path('/sys/fs/cgroup/memory.max').read_text())
    reserve=p['reserve_per_worker_bytes'];assert memory+reserve<.9*limit
    workers=min(p['maximum_workers'],max(1,int((.9*limit-memory)/reserve)))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1)as pool:
        for f in as_completed([pool.submit(truth_one,q)for q in e.read(p['cases']['path'])['cases']]):
            rows.append(f.result());e.write(DEST/'TRUTH_PROGRESS.json',dict(complete=len(rows),total=5,workers=workers))
    assert len(rows)==5 and sum(q['raw_children']for q in rows)==34
    e.write(DEST/'TRUTH_INDEX.json',dict(complete=True,cases=rows,total_complete_G=sum(q['G']for q in rows),
        all34_raw_curve_children_and_all_original_runs_retained=True,no_denominator_selected_by_prediction=True,
        old121TRAIN_27VAL_unchanged=True,protocol=e.bind(DEST/'FROZEN_PROTOCOL.json')))
    print('SUPPLEMENT_ALL_TRUTHS_FROZEN',sum(q['G']for q in rows),flush=True)


def score():
    p=check();index=e.read(DEST/'TRUTH_INDEX.json');assert index['complete']
    truths={q['source_id']:q for q in index['cases']};all_scores=[]
    for model in e.read(p['predictions']['path'])['variants']:
        rows=[]
        for case in e.read(model['frozen_outputs']['path'])['cases']:
            sid=case['source_id'];binding=case['prediction'];assert e.sha(binding['path'])==binding['sha256']
            packet=e.read(prediction.OUTPUT/'features'/sid/'geometry.json')
            decoded=e.read(binding['path']);original=[raw_score.exported(q)for q in decoded['parents']]
            chains,groups=grouping.source_groups(packet);parts=[]
            for curve in original:
                ci=curve.get('source_candidate')
                if isinstance(ci,int)and 0<=ci<len(chains):
                    gi=chains[ci];parts.append(frozen_eval.annotate(curve,groups[gi],gi,packet['candidates'][ci].get('material'),ci))
                else:parts.append(dict(curve,_carrier=None,_material=None))
            normalized=geometry.units(parts);truth=e.read(truths[sid]['truth']['path'])['units']
            metrics=grouping.matching(normalized,truth);raw=e.read(truths[sid]['raw_truth']['path'])
            raw_geometry=raw_score.matching(original,raw['runs']);raw_process=raw_score.matching(original,raw['runs'],raw)
            row=dict(source_id=sid,name=truths[sid]['name'],split='VAL',metrics=metrics,prediction_groups=normalized,
                truth_binding=truths[sid]['truth'],prediction_binding=binding,raw_geometry=raw_geometry,raw_process=raw_process,
                raw_curve_children=truths[sid]['raw_children'],original_prediction_units=len(original),
                process_direction_side_attributes_unaccepted=True,input_failure=case.get('input_failure',False))
            e.write(DEST/'scores'/model['variant']/'cases'/sid/'SCORE.json',row)
            rows.append({k:v for k,v in row.items()if k!='prediction_groups'})
        total={k:sum(q['metrics'][k]for q in rows)for k in ('G','P','TP','FP','FN')}
        total.update(inputs=5,extraction=total['TP']/total['G'],FP_per_G=total['FP']/total['G'])
        old=e.read(p['previous_old4_scores'][model['variant']]['path'])['totals']['unique_numeric_inputs']['VAL']
        combined={k:total[k]+old[k]for k in ('G','P','TP','FP','FN')}
        combined.update(inputs=9,extraction=combined['TP']/combined['G'],FP_per_G=combined['FP']/combined['G'])
        result=dict(complete=True,variant=model['variant'],totals=dict(additional5=total,old4_unchanged=old,all9_numeric_inputs=combined),cases=rows,
            frozen_outputs=model['frozen_outputs'],truth_index=e.bind(DEST/'TRUTH_INDEX.json'),protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),
            family_independence_unproved=True,no_model_selection_or_contract_pass=True)
        e.write(DEST/'scores'/model['variant']/'RESULT.json',result);all_scores.append(result)
        print('SUPPLEMENT_SCORE',model['variant'],result['totals'],flush=True)
    e.write(DEST/'RESULT.json',dict(complete=True,models=all_scores,all_new_inputs_scored=True,
        old4_new5_reported_separately=True,no_model_input_transform_or_threshold_changed=True,epoch=time.time()))


if __name__=='__main__':
    dict(prepare=prepare,controls=controls,truths=truths,score=score)[sys.argv[1]]()
