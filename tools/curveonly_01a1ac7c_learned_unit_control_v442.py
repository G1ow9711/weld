"""TRAIN-first full final-output control for learned native-unit arbitration."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,pickle,types,gc,copy,subprocess
from pathlib import Path
import numpy as np
from tools import curveonly_01a1ac7c_paired_wall_control_v436 as prior
from tools import curveonly_01a1ac7c_publication_trace_v439 as diagnosis
from weld_pipeline import curveonly_01a1ac7c_learned_unit_rules_v440 as rules
from weld_pipeline import curveonly_01a1ac7c_learned_unit_runtime_v441 as runtime

e,common=prior.e,prior.common
DEST,OUTPUT=e.REC/'learned_unit_control_v442',e.OUT/'learned_unit_control_v442'
QUICK=diagnosis.IDS


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p['inputs'],p['baseline_outputs'],p['model_bundle']]:common.valid(b)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(prior.DEST/'full/CLOSURE.json')['analysis_closed']
    assert e.read(diagnosis.DEST/'CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'learned_unit_controls_v443/CLOSURE.json')['all_passed']
    old=prior.check();before={c['source_id']:c for c in e.read(prior.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    trace={c['source_id']:c for c in e.read(diagnosis.DEST/'OUTPUTS_FROZEN.json')['cases']}
    jobs=[dict(j,current_prediction=before[j['source_id']]['prediction'],
        actual_raw_replay_proof=trace[j['source_id']]['raw'] if j['source_id'] in trace else None)
        for j in e.read(old['inputs']['path'])['cases']]
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve(),Path(runtime.__file__).resolve()]
    for source in sources:assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(source.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_labels_or_scores=True))
    baseline=e.read(prior.DEST/'full/RESULT.json')
    e.write(DEST/'BASELINE_V436_SCORES.json',dict(original_cases=baseline['cases'],additional_cases=[]))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,sources=[e.bind(s) for s in sources]+old['sources'],
        inputs=e.bind(DEST/'INPUTS.json'),baseline_outputs=e.bind(prior.DEST/'full/OUTPUTS_FROZEN.json'),
        baseline_scores=e.bind(DEST/'BASELINE_V436_SCORES.json'),model_bundle=old['model_bundle'],
        component_origin=old['component_origin'],
        design=e.bind(e.REC/'LEARNED_UNIT_ARBITRATION_DESIGN_V440.md'),
        controls=e.bind(e.REC/'learned_unit_controls_v443/CLOSURE.json'),
        diagnosis=e.bind(diagnosis.DEST/'CLOSURE.json'),quick_ids=list(QUICK),quick_is_3_failed_TRAIN_only=True,
        single_change='Use agreement of existing post-structural R35 raw unit and own frozen full expert to arbitrate native source competition, preserving different evidenced units and unmatched longer current outputs.',
        no_new_fit_or_VALcalibration=True,no_VALanswers_in_forward=True,same_program_for_all_inputs=True,
        cached_R35_executed_states_with_actual_frozen_source_expert_forward=True,cached_precision_not_cold=True,
        full_scope=dict(TRAIN_inputs=22,TRAIN_G=121,VAL_numeric=9,VAL_G=37,VAL_fileversions=10,VAL_fileversion_G=48),
        current_geometry_3mm_real_gaps_and_ARC30_unchanged=True,time_optimization_canceled=True,epoch=time.time()))
    print('LEARNED_UNIT_PROTOCOL_FROZEN',len(jobs),flush=True)


def one(job,phase,model):
    sid=job['source_id'];packet=e.read(common.valid(job['geometry'])['path'])
    x=np.load(common.valid(job['features'])['path']);before=e.read(common.valid(job['current_prediction'])['path'])
    qualified=before['curve_input_modality_rule']['selected']=='QUALIFIED_FINITE_STEP'
    if qualified:
        prepared=e.read(common.valid(job['prepared'])['path']);forward=e.read(common.valid(prepared['original_forward'])['path'])
        model.paired.primary.last_states=forward['source_decisions']
        model.paired.primary.last_representation_normalization=forward['primitive_representation_normalization']
        raw=runtime.actual_raw(x,packet,model)
        if job['actual_raw_replay_proof']:
            proof=e.read(common.valid(job['actual_raw_replay_proof'])['path'])
            assert all(json.dumps(raw,sort_keys=True)==json.dumps(q,sort_keys=True) for q in proof['calls'])
        result=rules.correct(before,x,packet,raw,model.point)
        audit=result['learned_unit_source_rule']
    else:
        result=copy.deepcopy(before);audit=dict(not_applicable='CURRENT_FULL_POINT_EXPERT_ROUTE_ALREADY_USED',new_parent_count=0,replaced_upstream_parent_indices=[])
    removed=set(audit['replaced_upstream_parent_indices']);retained=[v for i,v in enumerate(before['parents']) if i not in removed]
    for a,b in zip(retained,result['parents']):assert common.full.detail.equal(
        {k:v for k,v in a.items() if k!='parent_id'},{k:v for k,v in b.items() if k!='parent_id'})
    target=OUTPUT/phase/'outputs'/sid;target.mkdir(parents=True,exist_ok=True)
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',sid,result)
    e.write(target/'RULE_TRACE.json',dict(source_id=sid,input=job,audit=audit,
        actual_frozen_expert_forward=qualified,cached_R35_states_reconstituted_exact=True,
        all_untargeted_parent_fields_exact_except_parent_id=True,no_labels_or_scores_in_forward=True,epoch=time.time()))
    print('LEARNED_UNIT_FINAL_OUTPUT',phase,sid,len(before['parents']),len(result['parents']),
        'add',audit['new_parent_count'],'remove',len(removed),flush=True)
    return dict(source_id=sid,split=job['split'],prediction=e.bind(target/'curves.json'),csv=e.bind(target/'curves.csv'),
        trace=e.bind(target/'RULE_TRACE.json'),input_failure=False)


def predict(phase):
    p=check();prior.guards.install_fence();target=DEST/phase;target.mkdir(exist_ok=True)
    assert not (target/'OUTPUTS_FROZEN.json').exists();jobs=e.read(p['inputs']['path'])['cases'];outputs=[]
    if phase=='quick':jobs=[j for j in jobs if j['source_id'] in QUICK]
    elif phase=='full':
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full']
        outputs=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases'];reuse={c['source_id'] for c in outputs};jobs=[j for j in jobs if j['source_id'] not in reuse]
    else:raise ValueError(phase)
    resource=common.shared.resources.resources()
    assert .9*resource['maximum']-resource['memory']>1.2*1024**3 and resource['cpu']+2<.8*resource['quota'],resource
    e.write(target/'RUNNING.json',dict(workers=1,resource_start=resource,pid=os.getpid(),epoch=time.time()))
    with Path(common.valid(p['model_bundle'])['path']).open('rb') as stream:model=pickle.load(stream)
    for job in jobs:outputs.append(one(job,phase,model));gc.collect()
    outputs.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=outputs,all_outputs_closed_before_scores=True,cached_precision_not_cold=True,
        protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))


def score(phase):
    types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=check))(phase)
    result=e.read(DEST/phase/'RESULT.json');result['baseline']='V436 retained complete-curve model';e.write(DEST/phase/'RESULT.json',result)
    if phase=='quick':
        gained=sum(len(c['gained']) for c in result['cases']);lost=sum(len(c['lost']) for c in result['cases'])
        fp_delta=sum(c['metrics']['FP']-c['baseline']['FP'] for c in result['cases'])
        passed=gained>0 and lost==0 and fp_delta<=0
        e.write(DEST/'quick/CLOSURE.json',dict(analysis_closed=True,proceed_full=passed,gained_TP=gained,lost_TP=lost,FP_delta=fp_delta,
            failed_TRAIN_representatives_only=True,not_full_score=True,reason='Actual final-output gain with no correct loss or FP increase' if passed else 'Stop expansion and diagnose',epoch=time.time()))


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
