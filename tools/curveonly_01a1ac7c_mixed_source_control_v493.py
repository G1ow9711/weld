"""All-scope source generation, TRAIN-first output control and unchanged V60."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import copy,gc,hashlib,json,pickle,subprocess,sys,time,types
from pathlib import Path
import numpy as np
from tools import curveonly_01a1ac7c_independent_endcap_control_v481 as prior
from tools import curveonly_01a1ac7c_mixed_geometry_forward_v491 as pilot
from tools import curveonly_01a1ac7c_paired_face_control_v409 as guards
from weld_pipeline import curveonly_01a1ac7c_mixed_source_runtime_v492 as runtime

e,common=prior.e,prior.common
DEST=e.REC/'mixed_source_control_v493';OUTPUT=e.OUT/'mixed_source_control_v493'


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p['inputs'],p['model_bundle']]:common.valid(b)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists();assert e.read(pilot.DEST/'CLOSURE.json')['analysis_closed']
    assert e.read(pilot.DEST/'SCORE.json')['metrics']['TP']==1
    old=prior.check();outputs={c['source_id']:c for c in e.read(prior.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    jobs=[dict(j,current_prediction=outputs[j['source_id']]['prediction'],
        baseline_source_geometry=outputs[j['source_id']]['source_geometry']) for j in e.read(old['inputs']['path'])['cases']]
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    paths=[Path(__file__).resolve(),Path(runtime.__file__).resolve(),Path(runtime.sources.__file__).resolve(),
        Path(runtime.observations.__file__).resolve(),Path(runtime.observations.previous.__file__).resolve()]
    for path in paths:assert path.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(path.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truth_or_scores=True))
    before=e.read(prior.DEST/'full/RESULT.json')
    e.write(DEST/'BASELINE_V481_SCORES.json',dict(original_cases=before['cases'],additional_cases=[]))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,sources=[e.bind(s) for s in paths]+old['sources'],
        inputs=e.bind(DEST/'INPUTS.json'),model_bundle=old['model_bundle'],baseline_scores=e.bind(DEST/'BASELINE_V481_SCORES.json'),
        baseline_outputs=e.bind(prior.DEST/'full/OUTPUTS_FROZEN.json'),pilot=e.bind(pilot.DEST/'CLOSURE.json'),
        same_primary_version='V60',no_new_fit_or_VALcalibration=True,scope=old['full_scope'],
        single_change='Actual finite wall section at observed base planes, unique existing-bound curve/line connections, real134observations and actual frozen models, complete event inventory. Straight leaves are provenance only.',
        TRAIN_proceed='Final TRAIN F1 improves, retain original120TP; report all gains/losses/FP. No VAL tuning.',
        same3mm_true_gaps_ARC30=True,cached_unchanged_inputs_not_cold=True,epoch=time.time()))


def generate():
    p=check();guards.install_fence();rows=[]
    for job in e.read(p['inputs']['path'])['cases']:
        resource=common.shared.resources.resources()
        assert .9*resource['maximum']-resource['memory']>1.*1024**3 and resource['cpu']+1<.8*resource['quota'],resource
        sid=job['source_id'];target=OUTPUT/'sources'/sid;target.mkdir(parents=True,exist_ok=True)
        assert not (target/'GENERATED.json').exists()
        packet=e.read(common.valid(job['geometry'])['path']);prep=e.read(common.valid(job['prepared'])['path'])
        inventory=e.read(common.valid(prep['inventory'])['path'])
        solids=e.core.load_material(common.valid(job['STEP'])['path']) if job['STEP'] and packet.get('pose',{}).get('passed') else None
        print('FINITE_SOURCE_BEGIN',sid,len(solids) if solids else 0,flush=True)
        tick=time.perf_counter();value=runtime.sources.build(packet,inventory,solids)
        e.write(target/'GENERATED.json',dict(observation=value,input_geometry=job['geometry'],STEP=job['STEP'],
            inventory=prep['inventory'],no_label_or_model_reads=True))
        rows.append(dict(source_id=sid,split=job['split'],generated=e.bind(target/'GENERATED.json'),
            new_mixed=len(value['candidates']),primitives=len(value['primitives']),seconds=time.perf_counter()-tick))
        e.write(DEST/'SOURCE_PROGRESS.json',dict(cases=rows,total=32))
        print('FINITE_SOURCE_FROZEN',sid,len(value['candidates']),len(value['primitives']),round(rows[-1]['seconds'],3),flush=True)
        del solids;gc.collect()
    e.write(DEST/'SOURCES_FROZEN.json',dict(cases=rows,all_sources_closed_before_model_outputs_and_scores=True,
        no_truth_or_score_or_model_reads=True,epoch=time.time()))


def one(job,phase,source,model):
    target=OUTPUT/phase/'outputs'/job['source_id'];target.mkdir(parents=True,exist_ok=True)
    before=e.read(common.valid(job['current_prediction'])['path'])
    if not source['new_mixed']:
        result=copy.deepcopy(before);source_geometry=job['baseline_source_geometry'];f=job['features'];geometry=job['geometry'];actual=False
        receipt=dict(new_candidates=0,original_parent_fields_exact=True,unqualified_source_cached_model_precision_not_cold=True)
    else:
        generated=e.read(common.valid(source['generated'])['path'])['observation']
        packet=e.read(common.valid(job['geometry'])['path']);x=np.load(common.valid(job['features'])['path'])
        prep=e.read(common.valid(job['prepared'])['path']);inventory=e.read(common.valid(prep['inventory'])['path'])
        xyz=e.core.read_xyz(common.valid(job['raw'])['path']);solids=e.core.load_material(common.valid(job['STEP'])['path'])
        result,packet,x,inventory=runtime.forward(x,packet,model,solids,inventory,xyz,generated)
        old_packet=e.read(job['geometry']['path']);chains,groups=common.full.grouping.source_groups(old_packet)
        newchains,newgroups=common.full.grouping.source_groups(packet)
        assert chains==newchains[:len(chains)] and groups==newgroups[:len(groups)]
        np.save(target/'features.npy',x);e.write(target/'MODEL_GEOMETRY.json',packet)
        geometry=e.bind(target/'MODEL_GEOMETRY.json');f=e.bind(target/'features.npy')
        published=runtime.baseline.rules.published_source_view(result,packet)
        if published is packet:source_geometry=geometry
        else:e.write(target/'PUBLISHED_SOURCE_GEOMETRY.json',published);source_geometry=e.bind(target/'PUBLISHED_SOURCE_GEOMETRY.json')
        e.write(target/'MATERIAL_INVENTORY.json',inventory)
        receipt=result['mixed_finite_source_rule'];actual=True
        e.write(target/'ACTUAL_FORWARD.json',dict(states=model.last_states,model=p_model_binding(),no_labels=True))
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',job['source_id'],result)
    e.write(target/'RULE_TRACE.json',dict(source_id=job['source_id'],input=job,source=source,
        receipt=receipt,actual_model_forward=actual,no_truth_or_score_reads=True,epoch=time.time()))
    print('MIXED_FINAL_OUTPUT',phase,job['source_id'],len(before['parents']),len(result['parents']),actual,flush=True)
    return dict(source_id=job['source_id'],split=job['split'],prediction=e.bind(target/'curves.json'),csv=e.bind(target/'curves.csv'),
        trace=e.bind(target/'RULE_TRACE.json'),source_geometry=source_geometry,model_input_geometry=geometry,features=f,input_failure=False)


def p_model_binding():return e.read(DEST/'FROZEN_PROTOCOL.json')['model_bundle']


def predict(phase):
    p=check();assert phase in ['quick','full'];guards.install_fence()
    alljobs=e.read(p['inputs']['path'])['cases'];jobs=[j for j in alljobs if j['split']=='TRAIN'];out=[]
    if phase=='full':
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full']
        out=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases'];jobs=[j for j in alljobs if j['split']=='VAL']
    sources=e.read(DEST/'SOURCES_FROZEN.json');assert sources['all_sources_closed_before_model_outputs_and_scores']
    source={j['source_id']:j for j in sources['cases']}
    target=DEST/phase;target.mkdir(exist_ok=True);assert not (target/'OUTPUTS_FROZEN.json').exists()
    with Path(common.valid(p['model_bundle'])['path']).open('rb') as stream:model=pickle.load(stream)
    for job in jobs:
        if source[job['source_id']]['new_mixed']:
            r=common.shared.resources.resources();assert .9*r['maximum']-r['memory']>1.*1024**3 and r['cpu']+1<.8*r['quota'],r
        out.append(one(job,phase,source[job['source_id']],model));gc.collect()
    out.sort(key=lambda c:c['source_id']);lookup={j['source_id']:j for j in out}
    score_jobs=[dict(j,geometry=lookup[j['source_id']]['source_geometry'],
        original_model_geometry=j['geometry']) for j in alljobs if j['source_id'] in lookup]
    e.write(target/'SCORING_SOURCE_INPUTS.json',dict(cases=score_jobs,no_truth_or_scores=True))
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=out,all_outputs_closed_before_scores=True,
        source_geometry_inputs=e.bind(target/'SCORING_SOURCE_INPUTS.json'),epoch=time.time()))


def score(phase):
    p=check();f=e.read(DEST/phase/'OUTPUTS_FROZEN.json')
    def localcheck():return dict(p,inputs=f['source_geometry_inputs'])
    types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,check=localcheck))(phase)
    value=e.read(DEST/phase/'RESULT.json')
    if phase=='quick':
        now=value['totals']['TRAIN'];old=e.read(prior.DEST/'full/SCOPE_TOTALS.json')['TRAIN']
        f1=lambda m:2*m['TP']/(m['P']+m['G'])
        passed=now['TP']>=old['TP'] and f1(now)>f1(old)
        e.write(DEST/phase/'CLOSURE.json',dict(analysis_closed=True,proceed_full=passed,metrics=now,
            baseline=old,gained=sum(len(c['gained']) for c in value['cases']),lost=sum(len(c['lost']) for c in value['cases']),
            no_VALcalibration=True,epoch=time.time()))
        print('MIXED_TRAIN_PROCEED',passed,now,flush=True)


if __name__=='__main__':
    if sys.argv[1] in ['prepare','generate']:{'prepare':prepare,'generate':generate}[sys.argv[1]]()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
