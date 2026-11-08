"""Freeze and score the single independently-certified containment change."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import sys,time,types,gc,subprocess
from pathlib import Path
from tools import curveonly_01a1ac7c_coplanar_source_control_v458 as prior
from tools import curveonly_01a1ac7c_paired_face_control_v409 as guards
from weld_pipeline import curveonly_01a1ac7c_representation_containment_v461 as rules
from weld_pipeline import curveonly_01a1ac7c_representation_runtime_v462 as runtime

e,common=prior.e,prior.common
DEST=e.REC/'representation_control_v464';OUTPUT=e.OUT/'representation_control_v464'
QUICK_VAL={'curve_74f38ace29077eb1','curve_71166e99b1f46203','curve_08dcd4a0601b8674',
           'curve_6812484e745c54b3','curve_fb4f3a910837f6d9'}


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for binding in p['sources']+[p['inputs'],p['baseline_outputs'],p['model_bundle']]:common.valid(binding)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(prior.DEST/'full/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'representation_controls_v463/CLOSURE.json')['all_passed']
    old=prior.check();before={c['source_id']:c for c in e.read(prior.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    jobs=[dict(j,current_prediction=before[j['source_id']]['prediction']) for j in e.read(old['inputs']['path'])['cases']]
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve(),Path(runtime.__file__).resolve()]
    for s in sources:assert s.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(s.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truth_or_scores=True))
    baseline=e.read(prior.DEST/'full/RESULT.json')
    e.write(DEST/'BASELINE_V458_SCORES.json',dict(original_cases=baseline['cases'],additional_cases=[]))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,sources=[e.bind(s) for s in sources]+old['sources'],
        inputs=e.bind(DEST/'INPUTS.json'),model_bundle=old['model_bundle'],component_origin=old['component_origin'],
        baseline_outputs=e.bind(prior.DEST/'full/OUTPUTS_FROZEN.json'),baseline_scores=e.bind(DEST/'BASELINE_V458_SCORES.json'),
        design=e.bind(e.REC/'REPRESENTATION_CONTAINMENT_DESIGN_V461.md'),controls=e.bind(e.REC/'representation_controls_v463/CLOSURE.json'),
        quick_scope='all22TRAIN+5exposedVAL',quick_VAL_ids=sorted(QUICK_VAL),full_scope=old['full_scope'],
        same_primary_version='V60',single_change='Exact original source containment after qualified recovery, both actual output conversions independently continuously certified under original0.1mm limit.',
        no_new_fit_mining_or_VALthreshold_calibration=True,no_answers_in_forward=True,
        cached_base_model_outputs_not_cold=True,time_optimization_canceled=True,epoch=time.time()))
    print('REPRESENTATION_PROTOCOL_FROZEN',len(jobs),flush=True)


def one(job,phase):
    before=e.read(common.valid(job['current_prediction'])['path']);packet=e.read(common.valid(job['geometry'])['path'])
    result=rules.correct(before,packet);audit=result['representation_containment_rule'];removed=set(audit['removed_parent_indices'])
    kept=[p for i,p in enumerate(before['parents']) if i not in removed]
    for a,b in zip(kept,result['parents']):assert common.full.detail.equal({k:v for k,v in a.items() if k!='parent_id'},{k:v for k,v in b.items() if k!='parent_id'})
    if not removed:assert common.full.detail.equal(before['parents'],result['parents'])
    target=OUTPUT/phase/'outputs'/job['source_id'];target.mkdir(parents=True,exist_ok=True)
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',job['source_id'],result)
    e.write(target/'RULE_TRACE.json',dict(source_id=job['source_id'],input=job,audit=audit,
        base_model_output_cached_not_labels=True,retained_fields_exact_except_parent_id=True,epoch=time.time()))
    print('REPRESENTATION_FINAL_OUTPUT',phase,job['source_id'],len(before['parents']),len(result['parents']),sorted(removed),flush=True)
    return dict(source_id=job['source_id'],split=job['split'],prediction=e.bind(target/'curves.json'),csv=e.bind(target/'curves.csv'),trace=e.bind(target/'RULE_TRACE.json'),input_failure=False)


def predict(phase):
    p=check();guards.install_fence();target=DEST/phase;target.mkdir(exist_ok=True)
    assert not (target/'OUTPUTS_FROZEN.json').exists();jobs=e.read(p['inputs']['path'])['cases'];out=[]
    if phase=='quick':jobs=[j for j in jobs if j['split']=='TRAIN' or j['source_id'] in QUICK_VAL]
    elif phase=='full':
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full'];out=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases'];seen={j['source_id'] for j in out};jobs=[j for j in jobs if j['source_id'] not in seen]
    else:raise ValueError(phase)
    resource=common.shared.resources.resources()
    assert .9*resource['maximum']-resource['memory']>.8*1024**3 and resource['cpu']+1<.8*resource['quota'],resource
    e.write(target/'RUNNING.json',dict(workers=1,resource_start=resource,pid=os.getpid(),epoch=time.time()))
    for job in jobs:out.append(one(job,phase));gc.collect()
    out.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=out,all_outputs_closed_before_scores=True,
        cached_precision_not_cold=True,protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))


def score(phase):
    types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=check))(phase)
    result=e.read(DEST/phase/'RESULT.json');result['baseline']='V458 adopted full output';e.write(DEST/phase/'RESULT.json',result)
    if phase=='quick':
        gained=sum(len(c['gained']) for c in result['cases']);lost=sum(len(c['lost']) for c in result['cases']);delta=sum(c['metrics']['FP']-c['baseline']['FP'] for c in result['cases'])
        e.write(DEST/'quick/CLOSURE.json',dict(analysis_closed=True,proceed_full=lost==0 and (gained>0 or delta<0),
            gained_TP=gained,lost_TP=lost,FP_delta=delta,quick_scope='all22TRAIN+5exposedVAL',not_full_score=True,epoch=time.time()))


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
