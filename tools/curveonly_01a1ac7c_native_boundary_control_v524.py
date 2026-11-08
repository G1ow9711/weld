"""Full32 native-boundary export control with unchanged frozen evaluations."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import copy,gc,multiprocessing,pickle,subprocess,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from tools import curveonly_01a1ac7c_final_face_control_v516 as prior
from tools import curveonly_01a1ac7c_paired_face_report_v411 as metrics
from tools import curveonly_01a1ac7c_paired_face_control_v409 as guards
from weld_pipeline import curveonly_01a1ac7c_native_boundary_runtime_v523 as runtime

e,common,full=prior.e,prior.common,prior.common.full
DEST=e.REC/'native_boundary_control_v524'
OUTPUT=e.OUT/'native_boundary_control_v524'


def check(for_scoring=False):
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    bindings=p['sources']+[p['inputs'],p['model_bundle']]
    if for_scoring:bindings += [p['baseline_scores'],p['baseline_raw']]
    for b in bindings:common.valid(b)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(prior.DEST/'full/CLOSURE.json')['analysis_closed']
    for folder,count in [('native_boundary_controls_v522',12),('native_boundary_nested_controls_v526',1)]:
        assert e.read(e.REC/folder/'ACTUAL_TERMINAL.json')['exit_code']==0
        c=e.read(e.REC/folder/'CLOSURE.json');assert c['all_passed'] and c['count']==count
    p=prior.check();previous={c['source_id']:c for c in e.read(prior.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    jobs=[]
    for j in e.read(p['inputs']['path'])['cases']:
        out=previous[j['source_id']];prep=e.read(common.valid(j['prepared'])['path'])
        jobs.append(dict(j,current_prediction=out['prediction'],model_geometry=out['model_input_geometry'],
            model_features=out['features'],source_geometry=out['source_geometry'],
            final_inventory=j.get('final_inventory') or j.get('actual_inventory') or prep['inventory']))
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(runtime.__file__).resolve(),Path(runtime.rules.__file__).resolve()]
    for source in sources:assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(source.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truth_or_scores=True))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(p,source_commit=commit,
        sources=[e.bind(s) for s in sources]+p['sources']+[e.bind(runtime.rules.native.__file__),e.bind(runtime.rules.native.sampler.__file__)],
        inputs=e.bind(DEST/'INPUTS.json'),baseline_scores=e.bind(prior.DEST/'full/RESULT.json'),
        baseline_raw=e.bind(prior.DEST/'full/RAW_RECORD_DIAGNOSTICS.json'),
        baseline_categories=e.bind(prior.DEST/'full/CATEGORIES.json'),
        controls=[e.bind(e.REC/f/'CLOSURE.json') for f in ['native_boundary_controls_v522','native_boundary_nested_controls_v526']],
        change='Preserve exact-replayed native edge and nested material-path boundaries by exact analytic/polyline subdivision, retaining unknown process semantics.',
        no_new_fit_or_VALcalibration=True,all32_native_banks_frozen_before_any_score=True,
        quick_gate='All121TRAIN TP andFP0 retained, raw process TP net increase without loss of any formerly correct raw process truth.',
        cached_upstream_precision_not_cold=True,epoch=time.time()))


def generate_one(job):
    guards.install_fence()
    packet=e.read(common.valid(job['model_geometry'])['path'])
    before=e.read(common.valid(job['current_prediction'])['path'])
    if job.get('STEP'):common.valid(job['STEP'])
    evidence=runtime.rules.bank(packet,before,record=job)
    target=OUTPUT/'native_inputs'/job['source_id'];target.mkdir(parents=True,exist_ok=True)
    e.write(target/'NATIVE_BOUNDARY_BANK.json',evidence)
    row=dict(source_id=job['source_id'],split=job['split'],bank=e.bind(target/'NATIVE_BOUNDARY_BANK.json'),
        model_geometry=job['model_geometry'],current_prediction=job['current_prediction'],
        source_geometry=job['source_geometry'],STEP=job['STEP'],available=evidence['available'],
        no_labels_or_weights_read=True)
    e.write(target/'INPUT_RECEIPT.json',row)
    print('NATIVE_BOUNDARY_INPUT',job['source_id'],evidence['available'],sum(map(len,evidence['events'])),flush=True)
    return row


def batches(jobs,fn,phase):
    remaining=sorted(jobs,key=lambda j:j['model_geometry']['bytes'],reverse=True);rows=[];observations=[]
    while remaining:
        state=common.shared.resources.resources();available=.9*state['maximum']-state['memory']-.5*1024**3
        slots=min(2,int(.8*state['quota']-state['cpu']));chosen=[];reserved=0.
        for job in remaining:
            cost=.8*1024**3+8*job['model_geometry']['bytes']+5*(job['STEP']['bytes'] if job.get('STEP') else 0)
            if len(chosen)<slots and reserved+cost<=available:chosen.append(job);reserved+=cost
        if not chosen:
            e.write(DEST/f'{phase}_RESOURCE_WAIT.json',dict(resource=state,remaining=len(remaining),epoch=time.time()));time.sleep(5);continue
        observations.append(dict(resource=state,admitted=[j['source_id'] for j in chosen],estimated_bytes=reserved))
        with ProcessPoolExecutor(max_workers=len(chosen),mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            for future in as_completed([pool.submit(fn,j) for j in chosen]):
                rows.append(future.result());e.write(DEST/f'{phase}_PROGRESS.json',dict(cases=rows,resources=observations,epoch=time.time()))
        remaining=[j for j in remaining if j not in chosen];gc.collect()
    return sorted(rows,key=lambda r:r['source_id']),observations


def generate():
    p=check();assert not (DEST/'NATIVE_INPUTS_FROZEN.json').exists()
    jobs=e.read(p['inputs']['path'])['cases']
    rows,resources=batches(jobs,generate_one,'NATIVE_INPUT')
    assert len(rows)==32
    e.write(DEST/'NATIVE_INPUTS_FROZEN.json',dict(cases=rows,resources=resources,
        all_native_inputs_closed_before_new_predictions_and_scores=True,no_labels_or_weights_loaded=True,epoch=time.time()))


def predict(phase):
    p=check();native=e.read(DEST/'NATIVE_INPUTS_FROZEN.json');assert len(native['cases'])==32
    lookup={r['source_id']:r for r in native['cases']};alljobs=e.read(p['inputs']['path'])['cases'];outputs=[]
    if phase=='quick':jobs=[j for j in alljobs if j['split']=='TRAIN']
    else:
        assert phase=='full' and e.read(DEST/'quick/CLOSURE.json')['proceed_full']
        jobs=[j for j in alljobs if j['split']=='VAL'];outputs=e.read(DEST/'quick/OUTPUTS_FROZEN.json')['cases']
    target=DEST/phase;target.mkdir(exist_ok=True);assert not (target/'OUTPUTS_FROZEN.json').exists()
    guards.install_fence()
    for job in jobs:
        bankrow=lookup[job['source_id']]
        assert all(bankrow[k]==job[k] for k in ['model_geometry','current_prediction','source_geometry','STEP'])
        packet=e.read(common.valid(job['model_geometry'])['path']);before=e.read(common.valid(job['current_prediction'])['path'])
        evidence=e.read(common.valid(bankrow['bank'])['path'])
        result=runtime.rules.correct(before,packet,evidence)
        assert len(result['parents'])==len(before['parents'])
        for a,b in zip(before['parents'],result['parents']):
            assert {k:v for k,v in a.items() if k not in ['segments','native_representation_boundaries']}=={
                k:v for k,v in b.items() if k not in ['segments','native_representation_boundaries']}
        out=OUTPUT/phase/'outputs'/job['source_id'];out.mkdir(parents=True,exist_ok=True)
        e.write(out/'curves.json',result);common.shared.write_csv(out/'curves.csv',job['source_id'],result)
        e.write(out/'RULE_TRACE.json',dict(input=job,native_bank=bankrow['bank'],audit=result['native_boundary_export_rule'],
            original_complete_parent_geometry_and_fields_retained=True,no_label_or_score_reads=True,epoch=time.time()))
        outputs.append(dict(source_id=job['source_id'],split=job['split'],prediction=e.bind(out/'curves.json'),
            csv=e.bind(out/'curves.csv'),trace=e.bind(out/'RULE_TRACE.json'),source_geometry=job['source_geometry'],
            model_input_geometry=job['model_geometry'],features=job['model_features'],input_failure=False))
        print('NATIVE_BOUNDARY_FINAL',phase,job['source_id'],len(result['native_boundary_export_rule']['changes']),flush=True)
        gc.collect()
    outputs.sort(key=lambda c:c['source_id']);ids={c['source_id'] for c in outputs}
    scoring=[dict(j,geometry=j['source_geometry']) for j in alljobs if j['source_id'] in ids]
    e.write(target/'SCORING_SOURCE_INPUTS.json',dict(cases=scoring,no_truth_or_scores=True))
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=outputs,all_outputs_closed_before_scores=True,
        source_geometry_inputs=e.bind(target/'SCORING_SOURCE_INPUTS.json'),native_banks=e.bind(DEST/'NATIVE_INPUTS_FROZEN.json'),
        cached_upstream_precision_not_cold=True,epoch=time.time()))


def score(phase):
    p=check(for_scoring=True);frozen=e.read(DEST/phase/'OUTPUTS_FROZEN.json');assert frozen['all_outputs_closed_before_scores']
    jobs={j['source_id']:j for j in e.read(frozen['source_geometry_inputs']['path'])['cases']}
    baseline={c['source_id']:c for c in e.read(p['baseline_scores']['path'])['cases']}
    rawbase={c['source_id']:c for c in e.read(p['baseline_raw']['path'])['cases']}
    cases=[];rawrows=[];categories=[]
    for output in frozen['cases']:
        sid=output['source_id'];job=jobs[sid];old=baseline[sid]
        pred=e.read(common.valid(output['prediction'])['path']);packet=e.read(common.valid(job['geometry'])['path'])
        exported=[full.raw_score.exported(parent) for parent in pred['parents']];assert all(r['_valid_export'] for r in exported)
        truth=e.read(common.valid(old['truth_binding'])['path'])['units']
        chains,groups=full.grouping.source_groups(packet)
        parts=[full.evaluation.annotate(c,groups[chains[c['source_candidate']]],chains[c['source_candidate']],
            packet['candidates'][c['source_candidate']].get('material'),c['source_candidate']) for c in exported]
        units=full.geometry.units(parts);m=full.grouping.matching(units,truth)
        oldset={v['truth'] for v in old['metrics']['matches']};nowset={v['truth'] for v in m['matches']}
        cases.append(dict(source_id=sid,split=job['split'],metrics=m,gained=sorted(nowset-oldset),lost=sorted(oldset-nowset),
            baseline=old['metrics'],prediction=output['prediction'],truth_binding=old['truth_binding']))
        rawtruth=e.read(common.valid(rawbase[sid]['truth'])['path'])
        rawm={kind:full.raw_score.matching(exported,rawtruth['runs'],rawtruth if kind=='process' else None) for kind in ['geometry','process']}
        rawrows.append(dict(source_id=sid,split=job['split'],truth=rawbase[sid]['truth'],metrics=rawm,previous_V516=rawbase[sid]['metrics']))
        categories.append(dict(source_id=sid,split=job['split'],categories={name:full.grouping.matching(
            [c for c in units if c.get(key)==value],[c for c in truth if c.get(key)==value])
            for name,key,value in [('open','closed',False),('closed','closed',True),('horizontal','weld_type','horizontal'),('vertical','weld_type','vertical')]}))
        print('NATIVE_BOUNDARY_SCORE',sid,m['TP'],m['G'],m['FP'],'rawprocess',rawm['process']['TP'],rawm['process']['G'],flush=True)
    numeric=lambda row:row['source_id']!=metrics.ALT
    totals={s:common.shared.total([c for c in cases if c['split']==s and numeric(c)]) for s in ['TRAIN','VAL']}
    result=dict(complete=True,same_primary_version='V60',cases=cases,totals=totals,
        VAL_fileversions=common.shared.total([c for c in cases if c['split']=='VAL']),
        baseline='V516_previous_full_and_frozen_historical_best',same3mm_denominators_real_gaps=True,
        no_new_model_fit_or_VALcalibration=True,no_contract_acceptance=True,epoch=time.time())
    rawtotals={s:{kind:common.shared.total([dict(metrics=c['metrics'][kind]) for c in rawrows if c['split']==s and numeric(c)])
                  for kind in ['geometry','process']} for s in ['TRAIN','VAL']}
    cattotals={s:{kind:common.shared.total([dict(metrics=c['categories'][kind]) for c in categories if c['split']==s and numeric(c)])
                  for kind in ['open','closed','horizontal','vertical']} for s in ['TRAIN','VAL']}
    e.write(DEST/phase/'RESULT.json',result);e.write(DEST/phase/'SCOPE_TOTALS.json',metrics.scopes(result))
    e.write(DEST/phase/'RAW_RECORD_DIAGNOSTICS.json',dict(cases=rawrows,totals=rawtotals,separate_diagnostics_not_primary=True))
    e.write(DEST/phase/'CATEGORIES.json',dict(cases=categories,totals=cattotals,missing_categories_not_passed=True))
    if phase=='quick':
        m=totals['TRAIN'];old=e.read(prior.DEST/'full/SCOPE_TOTALS.json')['TRAIN']
        losses=[];gains=[]
        for c in rawrows:
            a={m['truth'] for m in c['metrics']['process']['matches']};b={m['truth'] for m in c['previous_V516']['process']['matches']}
            losses.extend(dict(source_id=c['source_id'],truth=i) for i in b-a)
            gains.extend(dict(source_id=c['source_id'],truth=i) for i in a-b)
        proceed=m['TP']>=old['TP'] and m['FP']<=old['FP'] and bool(gains) and not losses
        e.write(DEST/phase/'CLOSURE.json',dict(analysis_closed=True,proceed_full=proceed,metrics=m,
            raw_process=rawtotals['TRAIN']['process'],raw_gained=gains,raw_lost=losses,epoch=time.time()))
        print('NATIVE_BOUNDARY_QUICK_PROCEED',proceed,m,rawtotals['TRAIN']['process'],flush=True)


def integration():
    p=check();assert e.read(DEST/'FULL_SCORE_ACTUAL_TERMINAL.json')['exit_code']==0
    jobs={j['source_id']:j for j in e.read(p['inputs']['path'])['cases']}
    outputs={r['source_id']:r for r in e.read(DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    banks={r['source_id']:r for r in e.read(DEST/'NATIVE_INPUTS_FROZEN.json')['cases']}
    guards.install_fence();rows=[]
    for sid in ['curve_07661c4b6a52be6a','curve_0ccbe50eaf3cb217']:
        job=jobs[sid];features=np.load(common.valid(job['model_features'])['path'])
        packet=e.read(common.valid(job['model_geometry'])['path']);inventory=e.read(common.valid(job['final_inventory'])['path'])
        solids=e.core.load_material(common.valid(job['STEP'])['path'])
        with Path(common.valid(p['model_bundle'])['path']).open('rb') as stream:model=pickle.load(stream)
        actual,before,evidence=runtime.decode_existing(features,packet,model,solids,inventory,return_before=True)
        assert before['parents']==e.read(common.valid(job['current_prediction'])['path'])['parents']
        expected=e.read(common.valid(outputs[sid]['prediction'])['path'])
        from tools import curveonly_01a1ac7c_coplanar_source_report_v459 as compare
        differences=compare.compare_geometry(actual['parents'],expected['parents'])
        bankdiff=compare.compare_geometry(evidence,e.read(common.valid(banks[sid]['bank'])['path']))
        prep=e.read(common.valid(job['prepared'])['path']);original=e.read(common.valid(prep['original_forward'])['path'])
        assert model.last_states==original['source_decisions']
        rows.append(dict(source_id=sid,actual_R35_and_frozen_experts_forward=True,upstream_V516_parents_exact=True,
            new_original_native_edges_and_boundaries_replayed=True,final_parent_numeric_diffs=differences,
            input_bank_numeric_diffs=bankdiff,original_neural_states_exact=True))
        e.write(DEST/'full/INTEGRATION_PROGRESS.json',dict(cases=rows,complete=False,epoch=time.time()))
        print('NATIVE_BOUNDARY_ACTUAL_MODEL_EQUAL',sid,len(differences),len(bankdiff),flush=True)
        del solids,features,packet,inventory,model,actual,before;gc.collect()
    e.write(DEST/'full/ACTUAL_MODEL_RULE_INTEGRATION.json',dict(passed=True,cases=rows,
        model=p['model_bundle'],runtime=e.bind(runtime.__file__),no_label_or_score_reads=True,cached_input_observations_not_cold=True,epoch=time.time()))


if __name__=='__main__':
    {'prepare':prepare,'generate':generate,'integration':integration}[sys.argv[1]]() if sys.argv[1] in ['prepare','generate','integration'] else {
        'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
