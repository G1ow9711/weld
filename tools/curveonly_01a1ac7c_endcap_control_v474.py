"""Freeze post-model endcap recovery and score original V60 on source geometry."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import gc,hashlib,json,subprocess,sys,time,types
from pathlib import Path
from tools import curveonly_01a1ac7c_representation_control_v464 as prior
from tools import curveonly_01a1ac7c_endcap_coverage_pilot_v470 as pilot
from tools import curveonly_01a1ac7c_paired_face_control_v409 as guards
from weld_pipeline import curveonly_01a1ac7c_endcap_recovery_v471 as rules
from weld_pipeline import curveonly_01a1ac7c_endcap_runtime_v472 as runtime

e,common=prior.e,prior.common
DEST=e.REC/'endcap_control_v474';OUTPUT=e.OUT/'endcap_control_v474'
QUICK_VAL={'curve_74f38ace29077eb1','curve_71166e99b1f46203','curve_08dcd4a0601b8674',
           'curve_6812484e745c54b3','curve_fb4f3a910837f6d9'}


def connection_key(job):
    sources=[rules, rules.observation,rules.observation.original,rules.observation.original.caps,
        rules.observation.original.caps.previous,rules.observation.original.kernel]
    data=dict(STEP=job['STEP'],features=job['features'],geometry=job['geometry'],prepared=job['prepared'],
        prediction=job['current_prediction'],sources={m.__name__:e.sha(m.__file__) for m in sources})
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for binding in p['sources']+[p['inputs'],p['baseline_outputs'],p['model_bundle'],p['connection_cache']]:common.valid(binding)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(prior.DEST/'full/CLOSURE.json')['analysis_closed']
    assert e.read(pilot.DEST/'CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'endcap_controls_v473/CLOSURE.json')['all_passed']
    old=prior.check();outs={c['source_id']:c for c in e.read(prior.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    jobs=[dict(j,current_prediction=outs[j['source_id']]['prediction']) for j in e.read(old['inputs']['path'])['cases']]
    pp=e.read(pilot.DEST/'FROZEN_PROTOCOL.json');pj={j['source_id']:j for j in e.read(pp['inputs']['path'])['cases']}
    for binding in pp['sources']:common.valid(binding)
    observed={c['source_id']:c for c in e.read(pilot.DEST/'OBSERVATIONS.json')['cases']}
    entries=[]
    for job in jobs:
        same=pj[job['source_id']]
        assert all(job[k]==same[k] for k in ['STEP','geometry','features','prepared','current_prediction'])
        value=observed[job['source_id']];assert value['prediction']==job['current_prediction']
        entries.append(dict(context_sha256=connection_key(job),source_id=job['source_id'],observation=value['observation']))
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve(),Path(runtime.__file__).resolve()]
    for source in sources:assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(source.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truth_or_scores=True))
    e.write(DEST/'CONNECTION_CACHE.json',dict(entries=entries,source=e.bind(pilot.DEST/'OBSERVATIONS.json'),
        source_protocol=e.bind(pilot.DEST/'FROZEN_PROTOCOL.json'),exact_model_input_and_output_context=True))
    before=e.read(prior.DEST/'full/RESULT.json');e.write(DEST/'BASELINE_V464_SCORES.json',dict(original_cases=before['cases'],additional_cases=[]))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,sources=[e.bind(s) for s in sources]+pp['sources']+old['sources'],
        inputs=e.bind(DEST/'INPUTS.json'),model_bundle=old['model_bundle'],component_origin=old['component_origin'],
        baseline_outputs=e.bind(prior.DEST/'full/OUTPUTS_FROZEN.json'),baseline_scores=e.bind(DEST/'BASELINE_V464_SCORES.json'),
        connection_cache=e.bind(DEST/'CONNECTION_CACHE.json'),controls=e.bind(e.REC/'endcap_controls_v473/CLOSURE.json'),
        quick_scope='all22TRAIN+5exposedVAL',quick_VAL_ids=sorted(QUICK_VAL),full_scope=old['full_scope'],same_primary_version='V60',
        single_change='Model-qualified common finite endcap source connection with same opposite complete unit and own true gaps/coverage retained.',
        original_model_features_and_geometry_unchanged=True,
        scoring_uses_original_V60_with_explicit_published_source_provenance=True,
        original_source_groups_must_remain_exact=True,
        no_new_fit_mining_or_VALthreshold_calibration=True,no_answers_in_forward=True,cached_precision_not_cold=True,
        time_optimization_canceled=True,epoch=time.time()))
    print('ENDCAP_RECOVERY_PROTOCOL_FROZEN',len(jobs),flush=True)


def one(job,phase,cache):
    before=e.read(common.valid(job['current_prediction'])['path']);packet=e.read(common.valid(job['geometry'])['path'])
    key=connection_key(job);observed=next(v['observation'] for v in cache['entries'] if v['context_sha256']==key)
    result=rules.correct(before,packet,None,None,observed);audit=result['endcap_connection_recovery_rule']
    removed=set(audit['replaced_upstream_parent_indices']);kept=[p for i,p in enumerate(before['parents']) if i not in removed]
    for a,b in zip(kept,result['parents']):assert common.full.detail.equal({k:v for k,v in a.items() if k!='parent_id'},{k:v for k,v in b.items() if k!='parent_id'})
    if not removed:assert common.full.detail.equal(before['parents'],result['parents'])
    target=OUTPUT/phase/'outputs'/job['source_id'];target.mkdir(parents=True,exist_ok=True)
    published_view=rules.published_source_view(result,packet)
    if published_view is packet:source_geometry=job['geometry']
    else:
        chains,groups=common.full.grouping.source_groups(packet);newchains,newgroups=common.full.grouping.source_groups(published_view)
        assert chains==newchains[:len(chains)] and groups==newgroups[:len(groups)]
        e.write(target/'PUBLISHED_SOURCE_GEOMETRY.json',published_view);source_geometry=e.bind(target/'PUBLISHED_SOURCE_GEOMETRY.json')
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',job['source_id'],result)
    e.write(target/'RULE_TRACE.json',dict(source_id=job['source_id'],input=job,audit=audit,
        connection_context_sha256=key,published_source_geometry=source_geometry,
        original_model_geometry_and_features_unchanged=True,original_source_groups_exactly_preserved=True,
        retained_fields_exact_except_parent_id=True,cached_model_and_bound_geometry_not_cold=True,epoch=time.time()))
    print('ENDCAP_FINAL_OUTPUT',phase,job['source_id'],len(before['parents']),len(result['parents']),sorted(removed),flush=True)
    return dict(source_id=job['source_id'],split=job['split'],prediction=e.bind(target/'curves.json'),csv=e.bind(target/'curves.csv'),
        trace=e.bind(target/'RULE_TRACE.json'),source_geometry=source_geometry,input_failure=False)


def predict(phase):
    p=check();guards.install_fence();target=DEST/phase;target.mkdir(exist_ok=True)
    assert not (target/'OUTPUTS_FROZEN.json').exists();alljobs=e.read(p['inputs']['path'])['cases'];jobs=alljobs;out=[]
    if phase=='quick':jobs=[j for j in jobs if j['split']=='TRAIN' or j['source_id'] in QUICK_VAL]
    elif phase=='full':
        assert e.read(DEST/'quick/CLOSURE.json')['proceed_full'];out=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases'];seen={j['source_id'] for j in out};jobs=[j for j in jobs if j['source_id'] not in seen]
    else:raise ValueError(phase)
    resource=common.shared.resources.resources();assert .9*resource['maximum']-resource['memory']>.8*1024**3 and resource['cpu']+1<.8*resource['quota'],resource
    cache=e.read(common.valid(p['connection_cache'])['path'])
    for job in jobs:out.append(one(job,phase,cache));gc.collect()
    out.sort(key=lambda c:c['source_id']);outputs={c['source_id']:c for c in out}
    score_jobs=[dict(j,model_input_geometry=j['geometry'],geometry=outputs[j['source_id']]['source_geometry'],
        geometry_view='Published source provenance only; original model input separately bound') for j in alljobs if j['source_id'] in outputs]
    e.write(target/'SCORING_SOURCE_INPUTS.json',dict(cases=score_jobs,derived_from_frozen_input_and_published_geometry_only=True,
        original_model_input_unchanged=True,no_truth_or_scores=True))
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=out,all_outputs_closed_before_scores=True,
        source_geometry_inputs=e.bind(target/'SCORING_SOURCE_INPUTS.json'),cached_precision_not_cold=True,
        protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))


def score(phase):
    def score_check():
        p=check();binding=e.bind(DEST/phase/'SCORING_SOURCE_INPUTS.json')
        assert e.read(DEST/phase/'OUTPUTS_FROZEN.json')['source_geometry_inputs']==binding
        return dict(p,inputs=binding)
    types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=score_check))(phase)
    result=e.read(DEST/phase/'RESULT.json');result['baseline']='V464 adopted full output'
    result['original_V60_numeric_evaluator_and_truth_unchanged']=True
    result['post_model_output_provenance_view_not_fabricated_features']=True;e.write(DEST/phase/'RESULT.json',result)
    if phase=='quick':
        gained=sum(len(c['gained']) for c in result['cases']);lost=sum(len(c['lost']) for c in result['cases']);delta=sum(c['metrics']['FP']-c['baseline']['FP'] for c in result['cases'])
        e.write(DEST/'quick/CLOSURE.json',dict(analysis_closed=True,proceed_full=lost==0 and (gained>0 or delta<0),
            gained_TP=gained,lost_TP=lost,FP_delta=delta,quick_scope='all22TRAIN+5exposedVAL',not_full_score=True,epoch=time.time()))


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
