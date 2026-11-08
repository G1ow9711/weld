"""Full32 cached final-output control for the native recovery resolution guard."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key] = '1'
import subprocess
import sys
import time
import types
from pathlib import Path
from tools import curveonly_01a1ac7c_transverse_control_v339 as common
from tools import curveonly_01a1ac7c_edge_wall_control_v349 as preceding
from tools import curveonly_01a1ac7c_preserved_gap_control_v343 as fallback
from weld_pipeline import curveonly_01a1ac7c_recovery_resolution_v353 as runtime

e = common.e
DEST, OUTPUT = e.REC/'resolution_control_v354', e.OUT/'resolution_control_v354'


def check():
    p = e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p['inputs'],p['baseline_scores'],p['current_outputs'],p['fallback_outputs']]:
        common.valid(b)
    return p


def prepare():
    assert e.read(preceding.DEST/'full/CLOSURE.json')['analysis_closed']
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    commit = subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources = [Path(__file__).resolve(),Path(runtime.__file__).resolve()]
    for source in sources:
        assert source.read_bytes() == subprocess.check_output(['git','show',commit+':'+str(source.relative_to(e.ROOT))])
    current = {c['source_id']:c for c in e.read(preceding.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    before = {c['source_id']:c for c in e.read(fallback.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    inputs = e.read(preceding.DEST/'INPUTS.json')['cases']
    DEST.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)
    jobs = [dict(c,current_prediction=current[c['source_id']]['prediction'],
        fallback_prediction=before[c['source_id']]['prediction']) for c in inputs]
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truths_or_score_answers=True))
    scores = e.read(preceding.DEST/'full/RESULT.json')
    e.write(DEST/'BASELINE_V349_SCORES.json',dict(original_cases=scores['cases'],additional_cases=[],
        source=e.bind(preceding.DEST/'full/RESULT.json')))
    p = e.read(preceding.DEST/'FROZEN_PROTOCOL.json')
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(p,source_commit=commit,
        sources=[e.bind(s) for s in sources]+p['sources']+[e.bind(runtime.recovery.__file__),
            e.bind(runtime.baseline.__file__),e.bind(runtime.numeric.__file__)],
        inputs=e.bind(DEST/'INPUTS.json'),baseline_scores=e.bind(DEST/'BASELINE_V349_SCORES.json'),
        current_outputs=e.bind(preceding.DEST/'full/OUTPUTS_FROZEN.json'),
        fallback_outputs=e.bind(fallback.DEST/'full/OUTPUTS_FROZEN.json'),
        preceding_closed=e.bind(preceding.DEST/'full/CLOSURE.json'),
        single_change='Automatic edge-wall recovery requires curvature observable beyond the existing native projection/fit uncertainty; otherwise keep the preceding complete model output. No minimum weld length, new fitting or truth change.',
        cached_complete_model_outputs_not_cold=True,
        one_frozen_input_based_route_for_all32=True,all_outputs_closed_before_scores=True,
        numerical_projection_mm=runtime.numeric.NUMERIC_MM,epoch=time.time()))


def predict():
    p = check(); target = DEST/'full'; target.mkdir(exist_ok=True)
    assert not (target/'OUTPUTS_FROZEN.json').exists()
    def guard(event,args):
        if event == 'open' and isinstance(args[0],(str,bytes,Path)):
            path=Path(os.fsdecode(args[0]));value=str(path)
            if path.name in ('RESULT.json','SCORE.json','FIT_TARGETS.json') or any(
                    t in value for t in ('/truth/','/scores/','/targets/','/review/','/score_coordinate_maps/')):
                raise PermissionError('Resolution rule cannot read answers '+value)
        e.fence(event,args)
    sys.addaudithook(guard)
    outputs=[]
    for c in e.read(p['inputs']['path'])['cases']:
        sid=c['source_id'];packet=e.read(common.valid(c['geometry'])['path'])
        current=e.read(common.valid(c['current_prediction'])['path'])
        decoded=runtime.resolve(current,packet,lambda:e.read(common.valid(c['fallback_prediction'])['path']))
        directory=OUTPUT/'full/outputs'/sid;directory.mkdir(parents=True,exist_ok=True)
        e.write(directory/'curves.json',decoded);common.shared.write_csv(directory/'curves.csv',sid,decoded)
        e.write(directory/'RULE_TRACE.json',dict(source_id=sid,input=c,
            actual_R35_parent_replay_previously_verified=True,
            resolution_guard=decoded['automatic_recovery_resolution_guard'],
            cached_model_outputs_not_reference_answers=True))
        outputs.append(dict(source_id=sid,split=c['split'],prediction=e.bind(directory/'curves.json'),
            csv=e.bind(directory/'curves.csv'),trace=e.bind(directory/'RULE_TRACE.json'),input_failure=False))
        print('RESOLUTION_GUARD_OUTPUT_FROZEN',sid,decoded['automatic_recovery_resolution_guard']['selected_route'],
            decoded['geometry_units'],flush=True)
    assert len(outputs)==32
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=outputs,all_outputs_closed_before_scores=True,
        cached_precision_not_cold=True,protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))


def score():
    fn=types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=check))
    return fn('full')


if __name__=='__main__':
    {'prepare':prepare,'predict':predict,'score':score}[sys.argv[1]]()
