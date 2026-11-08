"""Full-scope finite identity/scope control of the current mixed-source model."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import copy,gc,hashlib,json,multiprocessing,pickle,subprocess,sys,time,types
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from tools import curveonly_01a1ac7c_mixed_source_control_v493 as research
from tools import curveonly_01a1ac7c_independent_endcap_control_v481 as prior
from tools import curveonly_01a1ac7c_finite_curve_alias_pilot_v503 as pilot
from tools import curveonly_01a1ac7c_paired_face_control_v409 as guards
from weld_pipeline import curveonly_01a1ac7c_finite_curve_runtime_v505 as runtime

e,common=research.e,research.common
DEST=e.REC/'finite_curve_control_v506';OUTPUT=e.OUT/'finite_curve_control_v506'


def key(job):
    value={k:job[k]['sha256'] if job.get(k) else None for k in ['STEP','geometry','prediction','features','inventory']}
    value['rules']=[e.sha(runtime.rules.__file__),e.sha(runtime.rules.previous.__file__)]
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p['inputs'],p['model_bundle'],p['alias_cache']]:common.valid(b)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(pilot.DEST/'CLOSURE.json')['analysis_closed']
    for name,count in [('finite_curve_alias_controls_v507',18),('point_geometry_controls_v510',5)]:
        assert e.read(e.REC/name/'ACTUAL_TERMINAL.json')['exit_code']==0
        c=e.read(e.REC/name/'CLOSURE.json');assert c['all_passed'] and c['count']==count
    rp=research.check();old=prior.check();po=e.read(pilot.DEST/'FROZEN_PROTOCOL.json')
    base={c['source_id']:c for c in e.read(prior.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    train={c['source_id']:c for c in e.read(research.DEST/'quick/OUTPUTS_FROZEN.json')['cases']}
    sources={c['source_id']:c for c in e.read(research.DEST/'SOURCES_FROZEN.json')['cases']}
    jobs=[]
    for j in e.read(rp['inputs']['path'])['cases']:
        sid=j['source_id'];out=train[sid] if j['split']=='TRAIN' else base[sid]
        new=sources[sid]['new_mixed'];assert j['split']=='TRAIN' or not new
        jobs.append(dict(j,upstream_prediction=out['prediction'],model_geometry=out.get('model_input_geometry',j['geometry']),
            source_geometry=out['source_geometry'],model_features=out.get('features',j['features']),new_mixed_sources=new,
            actual_inventory=e.bind(Path(out['prediction']['path']).parent/'MATERIAL_INVENTORY.json') if new else None))
    observed=e.read(pilot.DEST/'OBSERVATIONS.json')['cases'];entries=[]
    for row in observed:
        job=next(j for j in jobs if j['source_id']==row['source_id'])
        context=dict(STEP=job['STEP'],geometry=job['model_geometry'],prediction=job['upstream_prediction'],features=job['model_features'],inventory=job['actual_inventory'])
        assert all(context[k]==row['input'][k] for k in context)
        entries.append(dict(context_sha256=key(context),result=row['prediction'],input=row['input'],source_id=row['source_id']))
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    paths=[Path(__file__).resolve(),Path(runtime.__file__).resolve(),Path(runtime.schema.__file__).resolve(),
        Path(runtime.rules.__file__).resolve(),Path(runtime.rules.previous.__file__).resolve()]
    for path in paths:assert path.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(path.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truth_or_scores=True))
    e.write(DEST/'ALIAS_CACHE.json',dict(entries=entries,original=e.bind(pilot.DEST/'OBSERVATIONS.json'),exact_input_and_model_output_binding=True))
    previous=e.read(prior.DEST/'full/RESULT.json');e.write(DEST/'BASELINE_V481_SCORES.json',dict(original_cases=previous['cases'],additional_cases=[]))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,sources=[e.bind(p) for p in paths]+po['sources']+po['dependencies']+rp['sources'],
        inputs=e.bind(DEST/'INPUTS.json'),model_bundle=rp['model_bundle'],alias_cache=e.bind(DEST/'ALIAS_CACHE.json'),
        baseline_scores=e.bind(DEST/'BASELINE_V481_SCORES.json'),baseline_outputs=e.bind(prior.DEST/'full/OUTPUTS_FROZEN.json'),
        rejected_research=bind_research(),controls=e.bind(e.REC/'finite_curve_alias_controls_v507/CLOSURE.json'),
        schema_controls=e.bind(e.REC/'point_geometry_controls_v510/CLOSURE.json'),
        same_primary_version='V60',scope=rp['scope'],no_new_fit_or_VALcalibration=True,
        change='Unique finite joint identity and independent continuous source/conversion bounds; preserve whole hybrids and remove only represented endpoint spill into straight scope. Explicit point fallback schema preserves geometry.',
        cached_precision_not_cold=True,all_outputs_frozen_before_scores=True,epoch=time.time()))


def bind_research():return e.bind(research.DEST/'CLOSURE.json')


def one(job,phase,cache):
    before=e.read(common.valid(job['upstream_prediction'])['path']);packet=e.read(common.valid(job['model_geometry'])['path'])
    if job['new_mixed_sources']:
        context=dict(STEP=job['STEP'],geometry=job['model_geometry'],prediction=job['upstream_prediction'],features=job['model_features'],inventory=job['actual_inventory'])
        entry=next(c for c in cache['entries'] if c['context_sha256']==key(context))
        alias=e.read(common.valid(entry['result'])['path'])
    else:alias=runtime.rules.correct(before,packet,None,None)
    removed=set(alias['finite_curve_alias_scope_rule']['removed_parent_indices'])
    kept=[p for i,p in enumerate(before['parents']) if i not in removed]
    assert len(kept)==len(alias['parents'])
    for a,b in zip(kept,alias['parents']):assert {k:v for k,v in a.items() if k!='parent_id'}=={k:v for k,v in b.items() if k!='parent_id'}
    result=runtime.schema.correct(alias)
    assert not result['point_geometry_schema_rule']['unknown'],'Unresolved published curve representation'
    for parent in result['parents']:
        for segment in parent['segments']:
            if segment['geometry_type']=='arc':
                assert isinstance(segment.get('arc_parameters'),dict)
                assert abs(segment['arc_parameters']['sweep_rad'])<=np.pi/6+1e-12
    for record in result['point_geometry_schema_rule']['changed']:
        i=record['parent_index'];a=common.full.raw_score.exported(alias['parents'][i]);b=common.full.raw_score.exported(result['parents'][i])
        assert a['_valid_export'] and b['_valid_export'] and a['points']==b['points'] and a['_segment_turns']==b['_segment_turns']
    target=OUTPUT/phase/'outputs'/job['source_id'];target.mkdir(parents=True,exist_ok=True)
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',job['source_id'],result)
    e.write(target/'RULE_TRACE.json',dict(input=job,identity=result['finite_curve_alias_scope_rule'],schema=result['point_geometry_schema_rule'],
        retained_fields_exact_before_representation_metadata=True,canonical_geometry_and_segment_turns_exact=True,
        cached_actual_models_and_bound_finite_certificates_not_cold=True,no_labels_or_scores=True,epoch=time.time()))
    print('FINITE_CURVE_FINAL_OUTPUT',phase,job['source_id'],len(before['parents']),len(result['parents']),sorted(removed),flush=True)
    return dict(source_id=job['source_id'],split=job['split'],prediction=e.bind(target/'curves.json'),csv=e.bind(target/'curves.csv'),
        trace=e.bind(target/'RULE_TRACE.json'),source_geometry=job['source_geometry'],model_input_geometry=job['model_geometry'],
        features=job['model_features'],input_failure=False)


def predict(phase):
    p=check();guards.install_fence();alljobs=e.read(p['inputs']['path'])['cases'];out=[]
    if phase=='quick':jobs=[j for j in alljobs if j['split']=='TRAIN']
    elif phase=='full':
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full'];out=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases'];jobs=[j for j in alljobs if j['split']=='VAL']
    else:raise ValueError(phase)
    target=DEST/phase;target.mkdir(exist_ok=True);assert not (target/'OUTPUTS_FROZEN.json').exists()
    cache=e.read(p['alias_cache']['path'])
    for job in jobs:out.append(one(job,phase,cache));gc.collect()
    out.sort(key=lambda c:c['source_id']);lookup={c['source_id']:c for c in out}
    score_jobs=[dict(j,geometry=lookup[j['source_id']]['source_geometry']) for j in alljobs if j['source_id'] in lookup]
    e.write(target/'SCORING_SOURCE_INPUTS.json',dict(cases=score_jobs,no_truth_or_scores=True,actual_model_input_geometry_separately_bound=True))
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=out,all_outputs_closed_before_scores=True,
        source_geometry_inputs=e.bind(target/'SCORING_SOURCE_INPUTS.json'),cached_precision_not_cold=True,epoch=time.time()))


def score(phase):
    p=check();f=e.read(DEST/phase/'OUTPUTS_FROZEN.json')
    def localcheck():return dict(p,inputs=f['source_geometry_inputs'])
    types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,check=localcheck))(phase)
    value=e.read(DEST/phase/'RESULT.json')
    value['baseline']='V481_previous_full_and_frozen_historical_best'
    value['comparison_frozen_baseline']=p['baseline_scores']
    e.write(DEST/phase/'RESULT.json',value)
    if phase=='quick':
        now=value['totals']['TRAIN'];old=e.read(prior.DEST/'full/SCOPE_TOTALS.json')['TRAIN']
        passed=now['TP']>=old['TP'] and now['F1']>old['F1']
        e.write(DEST/phase/'CLOSURE.json',dict(analysis_closed=True,proceed_full=passed,metrics=now,baseline=old,
            gained=sum(len(c['gained']) for c in value['cases']),lost=sum(len(c['lost']) for c in value['cases']),epoch=time.time()))
        print('FINITE_CURVE_TRAIN_PROCEED',passed,now,flush=True)


def actual_one(value):
    job,model_binding,expected_binding=value;guards.install_fence()
    packet=e.read(common.valid(job['model_geometry'])['path']);features=np.load(common.valid(job['model_features'])['path'])
    inventory=e.read(common.valid(job['actual_inventory'])['path']);solids=e.core.load_material(common.valid(job['STEP'])['path'])
    with Path(common.valid(model_binding)['path']).open('rb') as stream:model=pickle.load(stream)
    print('FINITE_CURVE_ACTUAL_BEGIN',job['source_id'],flush=True)
    result,before=runtime.decode_existing(features,packet,model,solids,inventory,return_before=True)
    previous=e.read(common.valid(job['upstream_prediction'])['path']);expected=e.read(common.valid(expected_binding)['path'])
    assert before['parents']==previous['parents']
    assert result['parents']==expected['parents']
    states=e.read(Path(job['upstream_prediction']['path']).parent/'ACTUAL_FORWARD.json')['states']
    assert model.last_states==states
    a=copy.deepcopy(result['finite_curve_alias_scope_rule']);b=copy.deepcopy(expected['finite_curve_alias_scope_rule']);a.pop('seconds');b.pop('seconds')
    from tools import curveonly_01a1ac7c_coplanar_source_report_v459 as compare
    diffs=compare.compare_geometry(a,b)
    assert result['point_geometry_schema_rule']==expected['point_geometry_schema_rule']
    print('FINITE_CURVE_ACTUAL_EQUAL',job['source_id'],'query_diffs',len(diffs),flush=True)
    return dict(source_id=job['source_id'],actual_R35_and_frozen_experts_forward=True,upstream_parents_exact=True,
        final_parents_and_schema_exact=True,R35_states_exact=True,new_finite_faces_and_continuous_bounds_recomputed=True,
        original_pilot_cache_not_used=True,numeric_diffs=diffs,original_NUMERIC_mm=runtime.rules.NUMERIC,
        cached_actual_input_observations_not_cold=True)


def integration():
    p=check();assert e.read(DEST/'FULL_SCORE_ACTUAL_TERMINAL.json')['exit_code']==0
    jobs=[j for j in e.read(p['inputs']['path'])['cases'] if j['new_mixed_sources']]
    outs={c['source_id']:c for c in e.read(DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    r=common.shared.resources.resources();remaining=.9*r['maximum']-r['memory']
    workers=2 if remaining>3.*1024**3 and r['cpu']+2<.8*r['quota'] else 1
    assert remaining>1.*1024**3 and r['cpu']+workers<.8*r['quota'],r
    cases=[];values=[(j,p['model_bundle'],outs[j['source_id']]['prediction']) for j in jobs]
    if workers==1:
        for value in values:cases.append(actual_one(value));gc.collect()
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for future in as_completed([pool.submit(actual_one,v) for v in values]):
                cases.append(future.result());e.write(DEST/'full/INTEGRATION_PROGRESS.json',dict(cases=cases,total=len(jobs),workers=workers))
    e.write(DEST/'full/ACTUAL_MODEL_RULE_INTEGRATION.json',dict(passed=True,cases=sorted(cases,key=lambda c:c['source_id']),
        model=p['model_bundle'],workers=workers,resource=r,no_label_or_score_reads_in_actual_forward=True,
        all_new_source_contexts_tested=True,source=e.bind(__file__),epoch=time.time()))


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='integration':integration()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
