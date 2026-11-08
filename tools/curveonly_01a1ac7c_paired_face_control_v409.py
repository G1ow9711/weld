"""TRAIN-first final-output control for geometry-qualified source-side voting."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys
import time
import types
import pickle
import hashlib
import subprocess
import multiprocessing
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from tools import curveonly_01a1ac7c_transverse_control_v339 as common
from weld_pipeline import curveonly_01a1ac7c_paired_face_rules_v407 as rules
from weld_pipeline import curveonly_01a1ac7c_paired_face_runtime_v408 as runtime

e=common.e
DEST,OUTPUT=e.REC/'paired_face_control_v409',e.OUT/'paired_face_control_v409'
FENCE_INSTALLED=False


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for binding in p['sources']+[p['inputs'],p['baseline_outputs'],p['model_bundle']]:common.valid(binding)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(e.REC/'partial_contact_control_v386/full/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'lower_face_pilot_v404/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'source_expert_pilot_v406/CLOSURE.json')['analysis_closed']
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve(),Path(runtime.__file__).resolve(),
        Path(rules.expert.__file__).resolve(),Path(rules.geometry.__file__).resolve()]
    for source in sources:assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(source.relative_to(e.ROOT))])
    before={c['source_id']:c for c in e.read(e.REC/'partial_contact_control_v386/full/OUTPUTS_FROZEN.json')['cases']}
    jobs=[dict(c,current_prediction=before[c['source_id']]['prediction']) for c in e.read(e.REC/'partial_contact_control_v386/INPUTS.json')['cases']]
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truths_or_scores=True))
    baseline=e.read(e.REC/'partial_contact_control_v386/full/RESULT.json')
    e.write(DEST/'BASELINE_V386_SCORES.json',dict(original_cases=baseline['cases'],additional_cases=[],
        source=e.bind(e.REC/'partial_contact_control_v386/full/RESULT.json')))
    primary=e.bind(e.REC/'round35/run/model.pkl');source_expert=e.read(e.REC/'round27/run/MODEL_SPEC.json')['weights']
    for binding in (primary,source_expert):common.valid(binding)
    with Path(primary['path']).open('rb') as stream:base=pickle.load(stream)
    with Path(source_expert['path']).open('rb') as stream:old=pickle.load(stream)
    count_bytes=pickle.dumps(old.count,protocol=5)
    origin=dict(primary=primary,source_expert=source_expert,source_count_component_sha256=hashlib.sha256(count_bytes).hexdigest(),
        source_count_component_bytes=len(count_bytes),source_expert_TRAIN_fit=e.bind(e.REC/'round27/FROZEN_PROTOCOL.json'),
        no_new_model_fit=True,no_output_count_class_limit=True)
    model=runtime.PairedCurveModel(base,old.count,origin)
    with (DEST/'model.pkl').open('wb') as stream:pickle.dump(model,stream,protocol=5)
    model_binding=e.bind(DEST/'model.pkl')
    e.write(DEST/'MODEL_SPEC.json',dict(weights=model_binding,components=origin,runtime=e.bind(runtime.__file__),
        primary_variable_count_decoder_unchanged=True,extra_source_gate_only=True,base_columns=34,pooled_source_columns=189,
        no_new_geometric_feature_construction=True,full_contract_pass=False))
    old_protocol=e.read(e.REC/'partial_contact_control_v386/FROZEN_PROTOCOL.json')
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(old_protocol,source_commit=commit,
        sources=[e.bind(s) for s in sources]+old_protocol['sources']+[e.bind(rules.expert.graph.__file__),
            e.bind(rules.expert.graph.base.__file__)],
        inputs=e.bind(DEST/'INPUTS.json'),baseline_outputs=e.bind(e.REC/'partial_contact_control_v386/full/OUTPUTS_FROZEN.json'),
        baseline_scores=e.bind(DEST/'BASELINE_V386_SCORES.json'),model_bundle=model_binding,component_origin=origin,
        design=e.bind(e.REC/'PAIRED_FACE_SOURCE_RULE_DESIGN_V407.md'),
        single_change='For a fully ordered opposing-face projection of the same finite plate, use the frozen source expert only when it rejects this face and endorses an existing opposite-face model anchor, with the same finite upright passing both plate faces. Distinct faces are selected, never geometrically merged.',
        quick_is_all22TRAIN_only=True,no_new_fit_or_VALcalibration=True,cached_model_outputs_not_cold=True,
        full_outputs_frozen_before_scores=True,resource_policy='SharedCPU<=80percent/memory<=90percent. Use an in-process serial worker when parallel import memory is unavailable; numeric operations and per-input sources remain unchanged.',epoch=time.time()))
    print('PAIRED_FACE_MODEL_FROZEN',model_binding,flush=True)


def install_fence():
    global FENCE_INSTALLED
    if FENCE_INSTALLED:return
    def fence(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            path=Path(os.fsdecode(args[0]));s=str(path)
            if path.name in ('RESULT.json','SCORE.json','FIT_TARGETS.json','COMPLETE_UNITS.json') or any(
                    x in s for x in ('/truth/','/scores/','/targets/','/review/','/score_coordinate_maps/')):
                raise PermissionError('Paired-face forward cannot read answers '+s)
        e.fence(event,args)
    sys.addaudithook(fence);FENCE_INSTALLED=True


def one(job,phase):
    p=check();install_fence();sid=job['source_id']
    before=e.read(common.valid(job['current_prediction'])['path']);packet=e.read(common.valid(job['geometry'])['path'])
    x=np.load(common.valid(job['features'])['path']);prepared=e.read(common.valid(job['prepared'])['path'])
    inventory=e.read(common.valid(prepared['inventory'])['path'])
    solids=e.core.load_material(common.valid(job['STEP'])['path']) if job['STEP'] and packet.get('pose',{}).get('passed') else None
    with Path(common.valid(p['model_bundle'])['path']).open('rb') as stream:model=pickle.load(stream)
    result=rules.correct(before,x,packet,inventory,solids,model.source_expert);audit=result['paired_face_source_rule']
    removed=set(audit.get('removed_parent_indices',[]))
    retained=[v for i,v in enumerate(before['parents']) if i not in removed]
    assert len(retained)==len(result['parents'])
    for old,new in zip(retained,result['parents']):
        assert common.full.detail.equal({k:v for k,v in old.items() if k!='parent_id'},
            {k:v for k,v in new.items() if k!='parent_id'})
    target=OUTPUT/phase/'outputs'/sid;target.mkdir(parents=True,exist_ok=True)
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',sid,result)
    e.write(target/'RULE_TRACE.json',dict(source_id=sid,input=job,audit=audit,model=p['model_bundle'],
        primary_V386_output_cached_not_labels=True,all_retained_geometry_and_fields_exact=True,epoch=time.time()))
    print('PAIRED_FACE_FINAL_OUTPUT_FROZEN',phase,sid,len(before['parents']),len(result['parents']),sorted(removed),flush=True)
    return dict(source_id=sid,split=job['split'],prediction=e.bind(target/'curves.json'),csv=e.bind(target/'curves.csv'),
        trace=e.bind(target/'RULE_TRACE.json'),input_failure=False)


def predict(phase):
    p=check();assert phase in ('quick','full');target=DEST/phase;target.mkdir(exist_ok=True)
    assert not (target/'OUTPUTS_FROZEN.json').exists()
    jobs=e.read(p['inputs']['path'])['cases'];outputs=[];reuse=[]
    if phase=='quick':jobs=[j for j in jobs if j['split']=='TRAIN']
    else:
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full']
        outputs=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases'];reuse=[c['source_id'] for c in outputs]
        jobs=[j for j in jobs if j['source_id'] not in reuse]
    resource=common.shared.resources.resources();available=.9*resource['maximum']-resource['memory']
    cpu_workers=int((.8*resource['quota']-resource['cpu'])/1.3)
    parallel_workers=min(3,int((available-.25*1024**3)/(1.5*1024**3)),cpu_workers)
    if parallel_workers>=2:
        mode='PROCESS_POOL';workers=parallel_workers
    else:
        assert cpu_workers>=1 and available>=.65*1024**3,resource
        mode='SERIAL_IN_EXISTING_PROCESS';workers=1
    e.write(target/'RUNNING.json',dict(workers=workers,mode=mode,new_inputs=len(jobs),reused=reuse,resource_start=resource,epoch=time.time()))
    if mode=='PROCESS_POOL':
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            for future in as_completed([pool.submit(one,j,phase) for j in jobs]):outputs.append(future.result())
    else:
        for job in jobs:
            outputs.append(one(job,phase));gc.collect()
    outputs.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=outputs,all_outputs_closed_before_scores=True,cached_precision_not_cold=True,
        workers=workers,execution_mode=mode,resource_start=resource,reused=reuse,protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))
    print('PAIRED_FACE_ALL_OUTPUTS_FROZEN',phase,len(outputs),flush=True)


def score(phase):
    p=check();common.valid(p['baseline_scores'])
    return types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=check))(phase)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
