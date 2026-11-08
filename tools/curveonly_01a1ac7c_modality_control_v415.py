"""V413 preparation repair: preserve separate frozen objects; no inference change."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys
import time
import pickle
import hashlib
import types
import subprocess
import gc
import copy
from pathlib import Path
import numpy as np
from tools import curveonly_01a1ac7c_paired_face_control_v409 as prior
from weld_pipeline import curveonly_01a1ac7c_modality_runtime_v412 as runtime

e,common=prior.e,prior.common
DEST,OUTPUT=e.REC/'modality_control_v415',e.OUT/'modality_control_v415'


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p['inputs'],p['model_bundle'],p['baseline_outputs']]:common.valid(b)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(prior.DEST/'full/CLOSURE.json')['adopted_research']
    previous=prior.check();commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(runtime.__file__).resolve()]
    for s in sources:assert s.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(s.relative_to(e.ROOT))])
    outputs=e.read(prior.DEST/'full/OUTPUTS_FROZEN.json');before={c['source_id']:c for c in outputs['cases']}
    jobs=[dict(c,current_prediction=before[c['source_id']]['prediction']) for c in e.read(previous['inputs']['path'])['cases']]
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truths_or_scores=True))
    res=e.read(prior.DEST/'full/RESULT.json')
    e.write(DEST/'BASELINE_V409_SCORES.json',dict(original_cases=res['cases'],additional_cases=[]))
    legacy=e.read(e.REC/'round27/run/MODEL_SPEC.json')['weights'];common.valid(legacy)
    with Path(common.valid(previous['model_bundle'])['path']).open('rb') as stream:paired=pickle.load(stream)
    with Path(legacy['path']).open('rb') as stream:point=pickle.load(stream)
    failure=e.read(e.REC/'modality_control_v413/CLOSURE.json')
    assert failure['analysis_closed'] and failure['all_tree_states_exact'] and failure['parameters_exact']
    origin=dict(previous=previous['model_bundle'],point_expert=legacy,
        point_fit_protocol=e.bind(e.REC/'round27/FROZEN_PROTOCOL.json'),same_count_object_shared=False,
        no_new_model_fit=True,no_VALcalibration=True)
    model=runtime.ModalityCurveModel(paired,point,origin)
    with (DEST/'model.pkl').open('wb') as stream:pickle.dump(model,stream,protocol=5)
    spec=dict(weights=e.bind(DEST/'model.pkl'),runtime=e.bind(runtime.__file__),components=origin,
        point_input_columns=34,new_feature_construction=False,full_contract_pass=False)
    e.write(DEST/'MODEL_SPEC.json',spec)
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(previous,source_commit=commit,
        sources=[e.bind(s) for s in sources]+previous['sources']+[e.bind(runtime.point_runtime.__file__),e.bind(runtime.point_runtime.previous.__file__)],
        inputs=e.bind(DEST/'INPUTS.json'),baseline_outputs=e.bind(prior.DEST/'full/OUTPUTS_FROZEN.json'),
        baseline_scores=e.bind(DEST/'BASELINE_V409_SCORES.json'),model_bundle=spec['weights'],component_origin=origin,
        design=e.bind(e.REC/'MODALITY_ROUTE_DESIGN_V412.md'),serialization_fix=e.bind(e.REC/'MODALITY_SERIALIZATION_FIX_V415.md'),prepare_failure=e.bind(e.REC/'modality_control_v413/CLOSURE.json'),single_change='Select the own frozen full R27 source/interval/style expert if original finite STEP geometry is unavailable or not qualified; otherwise retain V408. Route is based only on input modality qualification, not workpiece identity or output count.',
        quick_is_all22TRAIN_only=True,no_new_fit_or_VALcalibration=True,epoch=time.time()))
    print('MODALITY_MODEL_FROZEN',spec['weights'],flush=True)


def one(job,phase,model):
    prior.install_fence();sid=job['source_id'];packet=e.read(common.valid(job['geometry'])['path'])
    before=e.read(common.valid(job['current_prediction'])['path']);prepared=e.read(common.valid(job['prepared'])['path'])
    x=np.load(common.valid(job['features'])['path'])
    if job['STEP']:common.valid(job['STEP'])
    available=bool(job['STEP'] and packet.get('pose',{}).get('passed') and
        any(c.get('STEP') and c.get('pose') for c in packet['candidates']))
    if available:
        result=copy.deepcopy(before)
        result['curve_input_modality_rule']=dict(branch='INPUT_GEOMETRY_AVAILABILITY_V412',selected='QUALIFIED_FINITE_STEP',
            pose_qualified=True,no_qualified_geometry_not_a_negative_weld_target=True,
            no_name_coordinate_size_or_count_lookup=True,no_new_fit_or_VALcalibration=True)
        assert common.full.detail.equal(before['parents'],result['parents'])
        cached=True
    else:
        result=runtime.decode(x,packet,model)
        cached=False
    target=OUTPUT/phase/'outputs'/sid;target.mkdir(parents=True,exist_ok=True)
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',sid,result)
    e.write(target/'RULE_TRACE.json',dict(source_id=sid,input=job,audit=result['curve_input_modality_rule'],
        qualified_upstream_published_geometry_cached_exact=cached,point_expert_actual_forward=not cached,
        point_expert_timing=None if cached else model.point.last_timing,epoch=time.time()))
    print('MODALITY_OUTPUT_FROZEN',phase,sid,result['curve_input_modality_rule']['selected'],len(result['parents']),flush=True)
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
    assert .9*resource['maximum']-resource['memory']>.65*1024**3 and resource['cpu']+1<.8*resource['quota'],resource
    e.write(target/'RUNNING.json',dict(workers=1,resource_start=resource,epoch=time.time()))
    with Path(common.valid(p['model_bundle'])['path']).open('rb') as stream:model=pickle.load(stream)
    for job in jobs:outputs.append(one(job,phase,model));gc.collect()
    outputs.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=outputs,all_outputs_closed_before_scores=True,
        cached_precision_not_cold=True,workers=1,protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))


def score(phase):
    return types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=check))(phase)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
