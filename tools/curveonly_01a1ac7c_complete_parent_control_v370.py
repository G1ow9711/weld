"""Final-output control for source-faithful adaptive rules and parent coverage."""
import sys
import types
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from tools import curveonly_01a1ac7c_adaptive_rule_control_v361 as preceding_control
from weld_pipeline import curveonly_01a1ac7c_complete_parent_guard_v367 as rules
from weld_pipeline import curveonly_01a1ac7c_complete_parent_runtime_v368 as runtime

original=preceding_control.original
e=original.e
DEST,OUTPUT=e.REC/'complete_parent_control_v370',e.OUT/'complete_parent_control_v370'
QUICK=('curve_06599ca427fb5556','curve_4738a0b266814107','curve_e2e843b1da887cf5')


def namespace():
    return dict(original.__dict__,__file__=__file__,DEST=DEST,OUTPUT=OUTPUT,
        rules=rules,runtime=runtime,check=check,one=one,QUICK=QUICK,namespace=namespace)


def check():return types.FunctionType(original.check.__code__,namespace())()


def prepare():
    assert e.read(e.REC/'adaptive_rule_control_v361/quick/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'source_uncertainty_audit_v366/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'resolution_control_v354/full/CLOSURE.json')['passed_research_adoption']
    types.FunctionType(original.prepare.__code__,namespace())()
    baseline=e.read(e.REC/'resolution_control_v354/full/RESULT.json')
    e.write(DEST/'BASELINE_V354_SCORES.json',dict(original_cases=baseline['cases'],additional_cases=[],
        source=e.bind(e.REC/'resolution_control_v354/full/RESULT.json'),same_V60=True))
    previous={c['source_id']:c for c in e.read(e.REC/'resolution_control_v354/INPUTS.json')['cases']}
    jobs=e.read(DEST/'INPUTS.json')
    jobs['cases']=[dict(c,fallback_prediction=previous[c['source_id']]['fallback_prediction']) for c in jobs['cases']]
    e.write(DEST/'INPUTS.json',jobs)
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(p,
        sources=p['sources']+e.read(e.REC/'resolution_control_v354/FROZEN_PROTOCOL.json')['sources']+[
            e.bind(original.__file__),e.bind(rules.adaptive.__file__),e.bind(runtime.__file__)],
        inputs=e.bind(DEST/'INPUTS.json'),baseline_scores=e.bind(DEST/'BASELINE_V354_SCORES.json'),
        single_repair='Preserve longer continuous source parent on partial native recovery. Original carrier stations and source discretization uncertainty; adaptive finite sections. No radius/name/count switch.',
        design=e.bind(e.REC/'COMPLETE_PARENT_ADAPTIVE_DESIGN_V367.md'),
        previous_failed_TRAIN=e.bind(e.REC/'adaptive_rule_control_v361/quick/CLOSURE.json'),
        actual_first_failure=e.bind(e.REC/'adaptive_rule_control_v361/ACTUAL_PARTIAL_PARENT_REPLACEMENT_CAUSE.json'),
        passed_TRAIN_kernel_control=e.bind(e.REC/'source_uncertainty_audit_v366/INVARIANCE.json'),
        quick_is_all22TRAIN_only=False,quick_is_exposed_TRAIN_and_iterative_VAL=False,
        quick_is_3_representative_TRAIN=True,
        workers_admission='Measured prior child RSS around1.1GiB; budget1.5GiB each plus0.5GiB reserve, shared90% memory/80% CPU',
        no_unmatched_fragment_negative_labels=True,no_new_fit=True,no_VALcalibration=True))


def one(job,phase):
    output=types.FunctionType(original.one.__code__,namespace())(job,phase)
    packet=e.read(job['geometry']['path']);decoded=e.read(output['prediction']['path'])
    result=runtime.resolution.resolve(decoded,packet,lambda:e.read(original.valid(job['fallback_prediction'])['path']))
    result['adaptive_local_section_runtime']='V368_ADAPTIVE_WITH_COMPLETE_PARENT_GUARD'
    folder=OUTPUT/phase/'outputs'/job['source_id']
    e.write(folder/'curves.json',result);original.shared.write_csv(folder/'curves.csv',job['source_id'],result)
    trace=e.read(folder/'RULE_TRACE.json');trace['resolution_guard']=result['automatic_recovery_resolution_guard']
    e.write(folder/'RULE_TRACE.json',trace)
    print('PARENT_SAFE_ADAPTIVE_OUTPUT_FROZEN',phase,job['source_id'],result['geometry_units'],flush=True)
    return dict(output,prediction=e.bind(folder/'curves.json'),csv=e.bind(folder/'curves.csv'),trace=e.bind(folder/'RULE_TRACE.json'))


def predict(phase):
    p=check();assert phase in ('quick','full')
    target=DEST/phase;target.mkdir(exist_ok=True);assert not (target/'OUTPUTS_FROZEN.json').exists()
    jobs=e.read(p['inputs']['path'])['cases'];outputs=[];reused=[]
    if phase=='quick':jobs=[c for c in jobs if c['source_id'] in QUICK]
    else:
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full']
        outputs=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases']
        reused=[c['source_id'] for c in outputs];jobs=[c for c in jobs if c['source_id'] not in reused]
    resource=original.shared.resources.resources()
    workers=min(3,int((.9*resource['maximum']-resource['memory']-.5*1024**3)/(1.5*1024**3)),
        int((.8*resource['quota']-resource['cpu'])/2))
    assert workers>=1,resource
    e.write(target/'RUNNING.json',dict(workers=workers,new_inputs=len(jobs),reused=reused,resource_start=resource,epoch=time.time()))
    print('ADAPTIVE_PARENT_CONTROL_START',phase,len(jobs),'workers',workers,flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
        pending={pool.submit(one,c,phase) for c in jobs}
        for future in __import__('concurrent.futures',fromlist=['as_completed']).as_completed(pending):outputs.append(future.result())
    outputs.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=outputs,all_outputs_closed_before_scores=True,
        workers=workers,resource_start=resource,reused_same_frozen_precision_outputs=reused,
        cached_precision_not_cold=True,protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))
    print('PARENT_SAFE_ALL_OUTPUTS_FROZEN',phase,len(outputs),flush=True)


def score(phase):return types.FunctionType(original.score.__code__,namespace())(phase)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
