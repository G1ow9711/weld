"""Input/model-only finite identity pilot on the new-source TRAIN contexts."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import gc,json,subprocess,sys,time
from pathlib import Path
from tools import curveonly_01a1ac7c_mixed_source_control_v493 as previous
from tools import curveonly_01a1ac7c_paired_face_control_v409 as guards
from weld_pipeline import curveonly_01a1ac7c_finite_curve_alias_v500 as rules

e,common=previous.e,previous.common
DEST=e.REC/'finite_curve_alias_pilot_v501';OUTPUT=e.OUT/'finite_curve_alias_pilot_v501'


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists();assert e.read(previous.DEST/'CLOSURE.json')['analysis_closed']
    p=previous.check();outputs={c['source_id']:c for c in e.read(previous.DEST/'quick/OUTPUTS_FROZEN.json')['cases']}
    source=e.read(previous.DEST/'SOURCES_FROZEN.json');selected={c['source_id'] for c in source['cases'] if c['new_mixed']}
    jobs=[]
    for job in e.read(p['inputs']['path'])['cases']:
        if job['source_id'] not in selected:continue
        assert job['split']=='TRAIN';out=outputs[job['source_id']]
        jobs.append(dict(source_id=job['source_id'],split='TRAIN',STEP=job['STEP'],geometry=out['model_input_geometry'],
            prediction=out['prediction'],features=out['features'],
            inventory=e.bind(Path(out['prediction']['path']).parent/'MATERIAL_INVENTORY.json')))
    assert len(jobs)==3
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    paths=[Path(__file__).resolve(),Path(rules.__file__).resolve()]
    for pth in paths:assert pth.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(pth.relative_to(e.ROOT))])
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_labels_or_scores=True))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,sources=[e.bind(x) for x in paths],
        dependencies=[e.bind(m.__file__) for m in [rules.step,rules.step.aliases,rules.conversion,rules.continuity,rules.publication,rules.sampler]],
        inputs=e.bind(DEST/'INPUTS.json'),previous_model_outputs=e.bind(previous.DEST/'quick/OUTPUTS_FROZEN.json'),
        original_source_mm=rules.SOURCE_MM,original_conversion_mm=rules.publication.MAX_CONVERSION_ERROR_MM,
        existing_model_endpoint_uncertainty_mm=rules.ENDPOINT_MM,no_new_fit_or_VALcalibration=True,
        no_changed_features_sources_models_or_geometry=True,cached_actual_model_postrule_precision_not_cold=True,
        previous_goal_turn_classification='PROGRESS: new native mixed source repaired BM79A capacity and actual modelTP; fullTRAIN control proved3newFP and changed the next action.',epoch=time.time()))


def observe():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+p['dependencies']+[p['inputs']]:common.valid(b)
    guards.install_fence();resource=common.shared.resources.resources()
    assert .9*resource['maximum']-resource['memory']>1.*1024**3 and resource['cpu']+1<.8*resource['quota'],resource
    rows=[]
    for job in e.read(p['inputs']['path'])['cases']:
        packet=e.read(common.valid(job['geometry'])['path']);before=e.read(common.valid(job['prediction'])['path'])
        inventory=e.read(common.valid(job['inventory'])['path']);solids=e.core.load_material(common.valid(job['STEP'])['path'])
        frozen=json.dumps(packet,sort_keys=True);print('FINITE_ALIAS_BEGIN',job['source_id'],flush=True)
        result=rules.correct(before,packet,inventory,solids);audit=result['finite_curve_alias_scope_rule']
        assert json.dumps(packet,sort_keys=True)==frozen
        removed=set(audit['removed_parent_indices']);retained=[x for i,x in enumerate(before['parents']) if i not in removed]
        assert len(retained)==len(result['parents'])
        for a,b in zip(retained,result['parents']):assert {k:v for k,v in a.items() if k!='parent_id'}=={k:v for k,v in b.items() if k!='parent_id'}
        target=OUTPUT/job['source_id'];target.mkdir(exist_ok=True);e.write(target/'curves.json',result)
        e.write(target/'RULE_TRACE.json',dict(input=job,audit=audit,no_labels_or_scores=True))
        rows.append(dict(source_id=job['source_id'],split='TRAIN',input=job,prediction=e.bind(target/'curves.json'),trace=e.bind(target/'RULE_TRACE.json'),
            before=len(before['parents']),after=len(result['parents']),removed=sorted(removed),actual_finite_queries=True))
        e.write(DEST/'PROGRESS.json',dict(cases=rows,total=3))
        print('FINITE_ALIAS_FROZEN',job['source_id'],len(before['parents']),len(result['parents']),sorted(removed),flush=True)
        del solids,packet,inventory,result;gc.collect()
    e.write(DEST/'OBSERVATIONS.json',dict(cases=rows,no_labels_scores_or_models_read=True,
        original_model_features_geometry_and_retained_fields_exact=True,not_new_TP_or_full_scope_claim=True,epoch=time.time()))


if __name__=='__main__':{'prepare':prepare,'observe':observe}[sys.argv[1]]()
