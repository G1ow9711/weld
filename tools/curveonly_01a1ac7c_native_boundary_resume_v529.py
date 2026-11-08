"""Resume V524 with corrected input-edge provenance, retaining all receipts."""
import copy,subprocess,sys,time,types
from pathlib import Path
from tools import curveonly_01a1ac7c_native_boundary_control_v524 as previous
from weld_pipeline import curveonly_01a1ac7c_native_boundary_runtime_v528 as runtime

e,common,full,prior,metrics=previous.e,previous.common,previous.full,previous.prior,previous.metrics
DEST,OUTPUT=previous.DEST,previous.OUTPUT


def check(for_scoring=False):
    original=previous.check(for_scoring)
    repair=e.read(DEST/'REPAIR_PROTOCOL_V527.json')
    for b in repair['sources']+[repair['original_protocol'],repair['failure_analysis'],repair['domain_controls']]:common.valid(b)
    return dict(original,sources=original['sources']+repair['sources'],source_commit=repair['source_commit'],
        original_source_commit=original['source_commit'],repair_protocol=e.bind(DEST/'REPAIR_PROTOCOL_V527.json'))


def prepare():
    assert not (DEST/'REPAIR_PROTOCOL_V527.json').exists()
    original=previous.check()
    assert e.read(DEST/'GENERATE_ACTUAL_TERMINAL.json')['exit_code']==1
    controls=e.REC/'native_boundary_domain_controls_v530'
    assert e.read(controls/'ACTUAL_TERMINAL.json')['exit_code']==0
    assert e.read(controls/'CLOSURE.json')['all_passed']
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    paths=[Path(__file__).resolve(),Path(runtime.__file__).resolve(),Path(runtime.rules.__file__).resolve()]
    for path in paths:assert path.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(path.relative_to(e.ROOT))])
    e.write(DEST/'REPAIR_PROTOCOL_V527.json',dict(source_commit=commit,sources=[e.bind(p) for p in paths],
        original_protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),
        failure_analysis=e.bind(DEST/'GENERATE_FIRST_FAILURE_ANALYSIS_V527.json'),
        domain_controls=e.bind(controls/'CLOSURE.json'),
        change='Only observer metadata: generated-candidate indices are not original material-edge indices. Use exact original point replay before accepting old members; retain explicit path boundaries and UNKNOWN. No feature/point/model/score changes.',
        reuse='Completed old banks already passed every original member/point replay; corrected guard is an exact identity there. Bind every input and output before reuse.',
        no_original_failure_overwritten=True,no_VALfit_or_threshold_calibration=True,epoch=time.time()))


def generate():
    p=check();assert not (DEST/'NATIVE_INPUTS_FROZEN.json').exists()
    jobs=e.read(p['inputs']['path'])['cases'];done=[];pending=[];proof=[]
    for job in jobs:
        receipt=OUTPUT/'native_inputs'/job['source_id']/'INPUT_RECEIPT.json'
        if not receipt.exists():pending.append(job);continue
        row=e.read(receipt)
        assert all(row[k]==job[k] for k in ['model_geometry','current_prediction','source_geometry','STEP'])
        for key in ['bank','model_geometry','current_prediction','source_geometry']:
            common.valid(row[key])
        if row['STEP']:common.valid(row['STEP'])
        evidence=e.read(row['bank']['path'])
        if evidence['available']:
            packet=e.read(row['model_geometry']['path']);before=e.read(row['current_prediction']['path'])
            view=runtime.rules.provenance.published_source_view(before,packet)
            expected={ci for ci,c in enumerate(view['candidates']) if c.get('original_native_members') and not c.get('source_material_path')}
            replayed={q['candidate'] for q in evidence['original_point_replays']}
            assert replayed==expected
            assert all(q['maximum_point_error_mm']<=1e-9 for q in evidence['original_point_replays'])
            assert evidence['native_replay']['original_primitive_count_exact']
        done.append(row);proof.append(dict(source_id=row['source_id'],receipt=e.bind(receipt),
            old_native_primitive_and_every_member_point_replay_already_qualified=True,
            new_guard_does_not_modify_verified_members=True,all_bound_geometry_and_inputs_exact=True))
    e.write(DEST/'COMPLETED_NATIVE_BANK_REUSE_V527.json',dict(cases=proof,count=len(done),
        original_failure=e.bind(DEST/'GENERATE_ACTUAL_TERMINAL.json'),cached_precision_not_cold=True,
        no_score_or_label_reads=True,epoch=time.time()))
    rows,resources=previous.batches(pending,generate_one,'NATIVE_INPUT_RESUME527')
    combined=sorted(done+rows,key=lambda r:r['source_id']);assert len(combined)==32
    e.write(DEST/'NATIVE_INPUTS_FROZEN.json',dict(cases=combined,resources=resources,
        reused=e.bind(DEST/'COMPLETED_NATIVE_BANK_REUSE_V527.json'),
        original_failed_receipt=e.bind(DEST/'GENERATE_ACTUAL_TERMINAL.json'),repair_protocol=e.bind(DEST/'REPAIR_PROTOCOL_V527.json'),
        all_native_inputs_closed_before_new_predictions_and_scores=True,no_labels_or_weights_loaded=True,epoch=time.time()))


_namespace=dict(previous.__dict__,runtime=runtime,check=check,__name__=__name__)
generate_one=types.FunctionType(previous.generate_one.__code__,_namespace)
predict=types.FunctionType(previous.predict.__code__,_namespace)
score=types.FunctionType(previous.score.__code__,_namespace)
integration=types.FunctionType(previous.integration.__code__,_namespace)


if __name__=='__main__':
    if sys.argv[1] in ['prepare','generate','integration']:
        {'prepare':prepare,'generate':generate,'integration':integration}[sys.argv[1]]()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
