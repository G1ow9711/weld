"""TRAIN-first final curve output control for paired finite wall recovery."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys
import json
import time
import types
import hashlib
import subprocess
import gc
from pathlib import Path
import numpy as np
from tools import curveonly_01a1ac7c_native_join_control_v420 as prior
from tools import curveonly_01a1ac7c_paired_face_control_v409 as guards
from weld_pipeline import curveonly_01a1ac7c_paired_wall_rules_v434 as rules
from weld_pipeline import curveonly_01a1ac7c_paired_wall_runtime_v435 as runtime

e,common=prior.e,prior.common
DEST,OUTPUT=e.REC/'paired_wall_control_v436',e.OUT/'paired_wall_control_v436'


def observation_sources():
    return [Path(m.__file__).resolve() for m in (rules.observation,rules.observation.previous,
        rules.kernel,rules.observation.sampler,rules.observation.safe,rules.old.observation,rules.old)]


def observation_key(job,inventory):
    values=dict(STEP=job['STEP']['sha256'] if job['STEP'] else None,geometry=job['geometry']['sha256'],
        features=job['features']['sha256'],inventory=inventory['sha256'],
        current_prediction=job['current_prediction']['sha256'],
        sources={p.name:e.sha(p) for p in observation_sources()})
    return hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p['inputs'],p['baseline_outputs'],p['model_bundle'],p['paired_geometry_cache']]:common.valid(b)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(prior.DEST/'full/CLOSURE.json')['analysis_closed']
    dense=e.REC/'paired_wall_native_pilot_v433/dense'
    assert e.read(dense/'CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'paired_wall_geometry_controls_v438/CLOSURE.json')['all_passed']
    p=prior.check();commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve(),Path(runtime.__file__).resolve()]+observation_sources()
    for s in sources:assert s.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(s.relative_to(e.ROOT))])
    previous_outputs=e.bind(prior.DEST/'full/OUTPUTS_FROZEN.json');before={c['source_id']:c for c in e.read(previous_outputs['path'])['cases']}
    jobs=[dict(c,current_prediction=before[c['source_id']]['prediction']) for c in e.read(p['inputs']['path'])['cases']]
    observed_protocol=e.read(dense/'FROZEN_PROTOCOL.json');common.valid(observed_protocol['model_outputs'])
    assert observed_protocol['model_outputs']['sha256']==previous_outputs['sha256']
    assert observed_protocol['input_jobs']['sha256']==p['inputs']['sha256']
    observed={c['source_id']:c for c in e.read(dense/'OBSERVATIONS.json')['cases']};entries=[]
    for job in jobs:
        prepared=e.read(common.valid(job['prepared'])['path'])
        entries.append(dict(context_sha256=observation_key(job,prepared['inventory']),
            observation=observed[job['source_id']]['observation']))
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truth_or_scores=True))
    result=e.read(prior.DEST/'full/RESULT.json')
    e.write(DEST/'BASELINE_V420_SCORES.json',dict(original_cases=result['cases'],additional_cases=[]))
    e.write(DEST/'PAIRED_GEOMETRY_CACHE.json',dict(entries=entries,
        original_dense_observations=e.bind(dense/'OBSERVATIONS.json'),
        original_protocol=e.bind(dense/'FROZEN_PROTOCOL.json'),
        model_parent_input_geometry_and_observation_sources_exact=True))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(p,source_commit=commit,
        sources=[e.bind(s) for s in sources]+p['sources'],inputs=e.bind(DEST/'INPUTS.json'),
        baseline_outputs=previous_outputs,baseline_scores=e.bind(DEST/'BASELINE_V420_SCORES.json'),
        paired_geometry_cache=e.bind(DEST/'PAIRED_GEOMETRY_CACHE.json'),
        design=e.bind(e.REC/'PAIRED_WALL_FINAL_DESIGN_V434.md'),
        paired_controls=e.bind(e.REC/'paired_wall_geometry_controls_v438/CLOSURE.json'),
        single_change='Recover a complete opposite finite wall/base component only with actual two-boundary correspondence to a verified complete model reference, independent contiguous own-model coverage within original source uncertainty, actual end/gap evidence and no parent outside the component.',
        no_new_fit_or_VALcalibration=True,quick_is_all22TRAIN_only=True,cached_precision_not_cold=True,epoch=time.time()))
    print('PAIRED_WALL_PROTOCOL_FROZEN',len(jobs),flush=True)


def one(job,phase):
    p=check();guards.install_fence();sid=job['source_id']
    before=e.read(common.valid(job['current_prediction'])['path']);packet=e.read(common.valid(job['geometry'])['path'])
    features=np.load(common.valid(job['features'])['path']);prepared=e.read(common.valid(job['prepared'])['path'])
    key=observation_key(job,prepared['inventory']);cache=e.read(common.valid(p['paired_geometry_cache'])['path'])
    entry=next(c for c in cache['entries'] if c['context_sha256']==key)
    result=rules.correct(before,features,packet,None,None,entry['observation']);audit=result['paired_wall_recovery_rule']
    removed=set(audit.get('replaced_upstream_parent_indices',[]));retained=[v for i,v in enumerate(before['parents']) if i not in removed]
    for a,b in zip(retained,result['parents']):assert common.full.detail.equal(
        {k:v for k,v in a.items() if k!='parent_id'},{k:v for k,v in b.items() if k!='parent_id'})
    if not audit.get('certificates'):assert common.full.detail.equal(before['parents'],result['parents'])
    target=OUTPUT/phase/'outputs'/sid;target.mkdir(parents=True,exist_ok=True)
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',sid,result)
    e.write(target/'RULE_TRACE.json',dict(source_id=sid,input=job,audit=audit,
        paired_geometry_context_sha256=key,exact_bound_dense_geometry_cache=True,
        all_untargeted_parent_fields_exact_except_parent_id=True,model_output_cached_not_labels=True))
    print('PAIRED_WALL_FINAL_OUTPUT',phase,sid,len(before['parents']),len(result['parents']),
        [(c['candidate'],c['start_mm'],c['end_mm']) for c in audit['certificates']],flush=True)
    return dict(source_id=sid,split=job['split'],prediction=e.bind(target/'curves.json'),csv=e.bind(target/'curves.csv'),
        trace=e.bind(target/'RULE_TRACE.json'),input_failure=False)


def predict(phase):
    p=check();assert phase in ('quick','full');target=DEST/phase;target.mkdir(exist_ok=True)
    assert not (target/'OUTPUTS_FROZEN.json').exists()
    jobs=e.read(p['inputs']['path'])['cases'];outputs=[]
    if phase=='quick':jobs=[j for j in jobs if j['split']=='TRAIN']
    else:
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full']
        outputs=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases'];reuse={c['source_id'] for c in outputs}
        jobs=[j for j in jobs if j['source_id'] not in reuse]
    resource=common.shared.resources.resources()
    assert .9*resource['maximum']-resource['memory']>.8*1024**3 and resource['cpu']+1<.8*resource['quota'],resource
    e.write(target/'RUNNING.json',dict(workers=1,resource_start=resource,epoch=time.time()))
    for job in jobs:outputs.append(one(job,phase));gc.collect()
    outputs.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=outputs,all_outputs_closed_before_scores=True,
        cached_precision_not_cold=True,protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))


def score(phase):
    return types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=check))(phase)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
