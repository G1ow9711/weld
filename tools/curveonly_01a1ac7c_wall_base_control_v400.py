"""TRAIN-first final output control for the model-anchored finite step rule."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys
import hashlib
import json
import time
import types
import subprocess
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from tools import curveonly_01a1ac7c_transverse_control_v339 as common
from weld_pipeline import curveonly_01a1ac7c_wall_base_rules_v398 as rules
from weld_pipeline import curveonly_01a1ac7c_wall_base_runtime_v399 as runtime

e=common.e
DEST,OUTPUT=e.REC/'wall_base_control_v400',e.OUT/'wall_base_control_v400'


def observation_key(job,inventory):
    bindings=dict(STEP=job['STEP']['sha256'] if job['STEP'] else None,
        geometry=job['geometry']['sha256'],inventory=inventory['sha256'],
        current_prediction=job['current_prediction']['sha256'],
        observation_source=e.bind(rules.observation.__file__)['sha256'])
    return hashlib.sha256(json.dumps(bindings,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    # Scoring bindings are validated/read only by the scorer, not the forward.
    for b in p['sources']+[p['inputs'],p['baseline_outputs'],p['observations_cache']]:common.valid(b)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(e.REC/'partial_contact_control_v386/full/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'wall_base_pilot_v393/ACTUAL_TERMINAL.json')['exit_code']==0
    assert e.read(e.REC/'wall_base_pilot_v393/CLOSURE.json')['analysis_closed']
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve(),Path(runtime.__file__).resolve()]
    for s in sources:assert s.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(s.relative_to(e.ROOT))])
    inputs=e.read(e.REC/'partial_contact_control_v386/INPUTS.json')['cases']
    outputs=e.read(e.REC/'partial_contact_control_v386/full/OUTPUTS_FROZEN.json');before={c['source_id']:c for c in outputs['cases']}
    jobs=[dict(c,current_prediction=before[c['source_id']]['prediction']) for c in inputs]
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truths_or_scores=True))
    baseline=e.read(e.REC/'partial_contact_control_v386/full/RESULT.json')
    e.write(DEST/'BASELINE_V386_SCORES.json',dict(original_cases=baseline['cases'],additional_cases=[],
        source=e.bind(e.REC/'partial_contact_control_v386/full/RESULT.json')))
    p=e.read(e.REC/'partial_contact_control_v386/FROZEN_PROTOCOL.json')
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(p,source_commit=commit,
        sources=[e.bind(s) for s in sources]+p['sources']+[e.bind(rules.observation.__file__),
            e.bind(rules.baseline.__file__),e.bind(rules.guard.__file__),e.bind(rules.observation.step.__file__),e.bind(common.__file__)],
        inputs=e.bind(DEST/'INPUTS.json'),baseline_outputs=e.bind(e.REC/'partial_contact_control_v386/full/OUTPUTS_FROZEN.json'),
        baseline_scores=e.bind(DEST/'BASELINE_V386_SCORES.json'),
        observations_cache=e.bind(e.REC/'wall_base_pilot_v393/OBSERVATION_CACHE.json'),
        frozen_R35_model=e.bind(e.REC/'round35/run/model.pkl'),
        design=e.bind(e.REC/'WALL_BASE_FINAL_DESIGN_V398.md'),
        single_change='Recover model-anchored native compound curved walls on a finite base using original material path, source uncertainty, fully verified local wall/base sections and weld-side-aware transverse passage. No new carrier bridging or VAL endpoint lookup.',
        quick_is_all22TRAIN_only=True,quick_is_exposed_TRAIN_and_iterative_VAL=False,
        no_new_fit_or_VALcalibration=True,cached_model_outputs_not_cold=True,
        all32_outputs_frozen_before_full_scores=True,epoch=time.time()))


def one(job,phase):
    p=check();sid=job['source_id']
    def fence(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            f=Path(os.fsdecode(args[0]));s=str(f)
            if f.name in ('RESULT.json','SCORE.json','FIT_TARGETS.json','COMPLETE_UNITS.json') or any(
                x in s for x in ('/truth/','/scores/','/targets/','/review/','/score_coordinate_maps/')):
                raise PermissionError('Step forward cannot read answers '+s)
        e.fence(event,args)
    sys.addaudithook(fence)
    before=e.read(common.valid(job['current_prediction'])['path']);packet=e.read(common.valid(job['geometry'])['path'])
    x=np.load(common.valid(job['features'])['path']);prepared=e.read(common.valid(job['prepared'])['path'])
    inventory=e.read(common.valid(prepared['inventory'])['path'])
    solids=e.core.load_material(common.valid(job['STEP'])['path']) if job['STEP'] and packet.get('pose',{}).get('passed') else None
    cache=e.read(common.valid(p['observations_cache'])['path'])
    key=observation_key(job,prepared['inventory'])
    cached=next((v for v in cache['entries'] if v['context_sha256']==key),None)
    observations={int(ci):value for ci,value in cached['observations'].items()} if cached else None
    result=rules.correct(before,x,packet,inventory,solids,observations);audit=result['wall_base_chain_rule']
    if not audit.get('certificates'):
        assert common.full.detail.equal(before['parents'],result['parents'])
    else:
        remove=set(audit['replaced_upstream_parent_indices']);retained=[v for i,v in enumerate(before['parents']) if i not in remove]
        for old,new in zip(retained,result['parents']):
            a={k:v for k,v in old.items() if k!='parent_id'};b={k:v for k,v in new.items() if k!='parent_id'}
            assert common.full.detail.equal(a,b)
    target=OUTPUT/phase/'outputs'/sid;target.mkdir(parents=True,exist_ok=True)
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',sid,result)
    e.write(target/'RULE_TRACE.json',dict(source_id=sid,input=job,upstream_R35_V386_preserved=True,
        model_output_cached_not_labels=True,audit=audit,epoch=time.time()))
    print('WALL_BASE_FINAL_OUTPUT_FROZEN',phase,sid,len(before['parents']),len(result['parents']),
        [c['candidate'] for c in audit.get('certificates',[])],flush=True)
    return dict(source_id=sid,split=job['split'],prediction=e.bind(target/'curves.json'),
        csv=e.bind(target/'curves.csv'),trace=e.bind(target/'RULE_TRACE.json'),input_failure=False)


def predict(phase):
    p=check();assert phase in ('quick','full')
    target=DEST/phase;target.mkdir(exist_ok=True);assert not (target/'OUTPUTS_FROZEN.json').exists()
    jobs=e.read(p['inputs']['path'])['cases'];outputs=[];reuse=[]
    if phase=='quick':jobs=[c for c in jobs if c['split']=='TRAIN']
    else:
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full']
        outputs=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases'];reuse=[c['source_id'] for c in outputs]
        jobs=[c for c in jobs if c['source_id'] not in reuse]
    resource=common.shared.resources.resources()
    workers=min(3,int((.9*resource['maximum']-resource['memory']-.5*1024**3)/(1.5*1024**3)),int((.8*resource['quota']-resource['cpu'])/2))
    assert workers>=1,resource
    e.write(target/'RUNNING.json',dict(workers=workers,new_inputs=len(jobs),reused=reuse,resource_start=resource,epoch=time.time()))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
        for future in as_completed([pool.submit(one,c,phase) for c in jobs]):outputs.append(future.result())
    outputs.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=outputs,all_outputs_closed_before_scores=True,
        cached_precision_not_cold=True,workers=workers,resource_start=resource,reused=reuse,
        protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))
    print('WALL_BASE_ALL_OUTPUTS_FROZEN',phase,len(outputs),flush=True)


def score(phase):
    p=check();common.valid(p['baseline_scores'])
    return types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=check))(phase)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
