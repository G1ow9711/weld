"""Same entire learned-unit stage with the V445 periodic overlap guard."""
import sys,types,time,subprocess
from pathlib import Path
from tools import curveonly_01a1ac7c_learned_unit_control_v442 as prior
from weld_pipeline import curveonly_01a1ac7c_periodic_unit_rules_v445 as rules
from weld_pipeline import curveonly_01a1ac7c_periodic_unit_runtime_v447 as runtime

e,common=prior.e,prior.common
DEST,OUTPUT=e.REC/'periodic_unit_control_v448',e.OUT/'periodic_unit_control_v448'
QUICK=('curve_baf6eacd4a0bf4d2',)
check=types.FunctionType(prior.check.__code__,dict(prior.__dict__,DEST=DEST))
one=types.FunctionType(prior.one.__code__,dict(prior.__dict__,DEST=DEST,OUTPUT=OUTPUT,rules=rules,runtime=runtime))
predict=types.FunctionType(prior.predict.__code__,dict(prior.__dict__,DEST=DEST,OUTPUT=OUTPUT,QUICK=QUICK,check=check,one=one))


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(prior.DEST/'full/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'periodic_unit_controls_v450/CLOSURE.json')['all_passed']
    p=prior.check();jobs=e.read(p['inputs']['path'])
    # Replace the last arbitration stage; its input remains the exact V436
    # stage output. All measured deltas compare against full adopted V442.
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve(),Path(runtime.__file__).resolve()]
    for s in sources:assert s.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(s.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True);e.write(DEST/'INPUTS.json',jobs)
    baseline=e.read(prior.DEST/'full/RESULT.json');e.write(DEST/'BASELINE_V442_SCORES.json',dict(original_cases=baseline['cases'],additional_cases=[]))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,sources=[e.bind(s) for s in sources]+p['sources'],
        inputs=e.bind(DEST/'INPUTS.json'),baseline_outputs=e.bind(prior.DEST/'full/OUTPUTS_FROZEN.json'),
        baseline_scores=e.bind(DEST/'BASELINE_V442_SCORES.json'),model_bundle=p['model_bundle'],component_origin=p['component_origin'],
        previous_closed=e.bind(prior.DEST/'full/CLOSURE.json'),
        first_failure=e.bind(prior.DEST/'full/review/NEW_TRAIN_FP_FIRST_FAILURE.json'),
        controls=e.bind(e.REC/'periodic_unit_controls_v450/CLOSURE.json'),quick_ids=list(QUICK),quick_scope='one failed periodic TRAIN case',
        replaced_last_stage_input='Exact V436 output, same asV442; comparisons toV442',
        single_change='Reuse original proven periodic native source map for overlap. An unknown interval cannot certify absence of an existing published unit.',
        no_new_fit_or_VALcalibration=True,cached_precision_not_cold=True,same_program_for_all_inputs=True,
        full_scope=p['full_scope'],current_geometry_3mm_real_gaps_and_ARC30_unchanged=True,epoch=time.time()))
    print('PERIODIC_UNIT_PROTOCOL_FROZEN',len(jobs['cases']),flush=True)


def score(phase):
    types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=check))(phase)
    result=e.read(DEST/phase/'RESULT.json');result['baseline']='V442 adopted full output';e.write(DEST/phase/'RESULT.json',result)
    if phase=='quick':
        gain=sum(len(c['gained']) for c in result['cases']);lost=sum(len(c['lost']) for c in result['cases']);delta=sum(c['metrics']['FP']-c['baseline']['FP'] for c in result['cases'])
        e.write(DEST/'quick/CLOSURE.json',dict(analysis_closed=True,proceed_full=lost==0 and (gain>0 or delta<0),gained_TP=gain,lost_TP=lost,FP_delta=delta,not_full_score=True,epoch=time.time()))


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
