"""TRAIN-first final output control of model-native join guarded recovery."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys
import json
import time
import types
import subprocess
import gc
from pathlib import Path
import numpy as np
from tools import curveonly_01a1ac7c_modality_control_v415 as prior
from tools import curveonly_01a1ac7c_wall_base_control_v400 as historical
from weld_pipeline import curveonly_01a1ac7c_native_join_rules_v418 as rules
from weld_pipeline import curveonly_01a1ac7c_native_join_runtime_v419 as runtime

e,common=prior.e,prior.common
DEST,OUTPUT=e.REC/'native_join_control_v420',e.OUT/'native_join_control_v420'


def check():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p['inputs'],p['baseline_outputs'],p['model_bundle'],p['observations_cache']]:common.valid(b)
    return p


def prepare():
    assert not (DEST/'FROZEN_PROTOCOL.json').exists()
    assert e.read(prior.DEST/'full/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'native_join_pilot_v417/CLOSURE.json')['analysis_closed']
    assert e.read(e.REC/'native_join_geometry_controls_v422/CLOSURE.json')['all_passed']
    p=prior.check();commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    sources=[Path(__file__).resolve(),Path(rules.__file__).resolve(),Path(runtime.__file__).resolve()]
    for s in sources:assert s.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(s.relative_to(e.ROOT))])
    before={c['source_id']:c for c in e.read(prior.DEST/'full/OUTPUTS_FROZEN.json')['cases']}
    jobs=[dict(c,current_prediction=before[c['source_id']]['prediction']) for c in e.read(p['inputs']['path'])['cases']]
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    e.write(DEST/'INPUTS.json',dict(cases=jobs,no_truths_or_scores=True))
    result=e.read(prior.DEST/'full/RESULT.json')
    e.write(DEST/'BASELINE_V415_SCORES.json',dict(original_cases=result['cases'],additional_cases=[],source=e.bind(prior.DEST/'full/RESULT.json')))
    oldp=historical.check();cache=e.read(common.valid(oldp['observations_cache'])['path'])
    oldjobs={c['source_id']:c for c in e.read(oldp['inputs']['path'])['cases']};newcache=[]
    for job in jobs:
        oldjob=oldjobs[job['source_id']];prepared=e.read(common.valid(job['prepared'])['path'])
        oldprepared=e.read(common.valid(oldjob['prepared'])['path']);key=historical.observation_key(oldjob,oldprepared['inventory'])
        cached=next((c for c in cache['entries'] if c['context_sha256']==key),None)
        if not cached:continue
        for name in ('STEP','geometry'):assert job[name]['sha256']==oldjob[name]['sha256']
        assert prepared['inventory']['sha256']==oldprepared['inventory']['sha256']
        newparents=e.read(common.valid(job['current_prediction'])['path'])['parents']
        oldparents=e.read(common.valid(oldjob['current_prediction'])['path'])['parents']
        assert json.dumps(newparents,sort_keys=True)==json.dumps(oldparents,sort_keys=True)
        newcache.append(dict(context_sha256=historical.observation_key(job,prepared['inventory']),
            observations=cached['observations'],original_context_sha256=key,
            all_geometry_and_model_parent_values_exact=True))
    e.write(DEST/'OBSERVATION_CACHE.json',dict(entries=newcache,original_cache=oldp['observations_cache'],
        original_context_rebound_only_after_exact_input_and_model_parent_identity=True))
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(p,source_commit=commit,
        sources=[e.bind(s) for s in sources]+p['sources']+oldp['sources'],
        inputs=e.bind(DEST/'INPUTS.json'),baseline_outputs=e.bind(prior.DEST/'full/OUTPUTS_FROZEN.json'),
        baseline_scores=e.bind(DEST/'BASELINE_V415_SCORES.json'),model_bundle=p['model_bundle'],
        observations_cache=e.bind(DEST/'OBSERVATION_CACHE.json'),
        design=e.bind(e.REC/'NATIVE_JOIN_FINAL_DESIGN_V418.md'),
        geometry_controls=e.bind(e.REC/'native_join_geometry_controls_v422/CLOSURE.json'),
        single_change='Require an existing single model-native curve to cross a true source join by more than the original3mmendpoint uncertainty on both sides before applying original finite wall/base recovery. Separate neighbouring parents do not qualify. Full support, real gaps and parent guards stay in force.',
        quick_is_all22TRAIN_only=True,no_new_fit_or_VALcalibration=True,cached_model_outputs_not_cold=True,epoch=time.time()))
    print('NATIVE_JOIN_PROTOCOL_FROZEN',len(jobs),'exact_context_caches',len(newcache),flush=True)


def one(job,phase):
    p=check();prior.prior.install_fence();sid=job['source_id']
    before=e.read(common.valid(job['current_prediction'])['path']);packet=e.read(common.valid(job['geometry'])['path'])
    features=np.load(common.valid(job['features'])['path']);prepared=e.read(common.valid(job['prepared'])['path'])
    inventory=e.read(common.valid(prepared['inventory'])['path'])
    solids=e.core.load_material(common.valid(job['STEP'])['path']) if job['STEP'] and packet.get('pose',{}).get('passed') else None
    cache=e.read(common.valid(p['observations_cache'])['path']);key=historical.observation_key(job,prepared['inventory'])
    cached=next((c for c in cache['entries'] if c['context_sha256']==key),None)
    observations={int(ci):value for ci,value in cached['observations'].items()} if cached else None
    result=rules.correct(before,features,packet,inventory,solids,observations);audit=result['wall_base_chain_rule']
    remove=set(audit.get('replaced_upstream_parent_indices',[]))
    retained=[v for i,v in enumerate(before['parents']) if i not in remove]
    for a,b in zip(retained,result['parents']):
        assert common.full.detail.equal({k:v for k,v in a.items() if k!='parent_id'},
            {k:v for k,v in b.items() if k!='parent_id'})
    if not audit.get('certificates'):assert common.full.detail.equal(before['parents'],result['parents'])
    target=OUTPUT/phase/'outputs'/sid;target.mkdir(parents=True,exist_ok=True)
    e.write(target/'curves.json',result);common.shared.write_csv(target/'curves.csv',sid,result)
    e.write(target/'RULE_TRACE.json',dict(source_id=sid,input=job,audit=audit,exact_bound_dense_cache_used=bool(cached),
        all_untargeted_parent_fields_exact=True,current_V415_model_output_cached_not_labels=True,epoch=time.time()))
    print('NATIVE_JOIN_FINAL_OUTPUT_FROZEN',phase,sid,len(before['parents']),len(result['parents']),
        [c['candidate'] for c in audit.get('certificates',[])],flush=True)
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
    e.write(target/'RUNNING.json',dict(workers=1,resource_start=resource,reason='Most inputs exit at model-native guard; only one TRAIN candidate needs finite geometry. Single process avoids redundant heavy import memory.',epoch=time.time()))
    for job in jobs:outputs.append(one(job,phase));gc.collect()
    outputs.sort(key=lambda c:c['source_id'])
    e.write(target/'OUTPUTS_FROZEN.json',dict(cases=outputs,all_outputs_closed_before_scores=True,
        cached_precision_not_cold=True,workers=1,protocol=e.bind(DEST/'FROZEN_PROTOCOL.json'),epoch=time.time()))


def score(phase):
    return types.FunctionType(common.score.__code__,dict(common.__dict__,DEST=DEST,OUTPUT=OUTPUT,check=check))(phase)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    else:{'predict':predict,'score':score}[sys.argv[1]](sys.argv[2])
