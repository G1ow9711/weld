"""Fresh shared-initialization111/134 ordered-span TRAIN-family comparison."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_ordered_span_train_v236 as original_training
from tools import curveonly_01a1ac7c_direction_span_train_v248 as augmented_training
from tools import curveonly_01a1ac7c_proposal_quality_control_v241 as export
from weld_pipeline import curveonly_01a1ac7c_ordered_spans_v235 as original
from weld_pipeline import curveonly_01a1ac7c_direction_spans_v247 as augmented

DEST=e.REC/'direction_span_family_v249'
OUTPUT=e.OUT/'direction_span_family_v249'
VARIANTS={'original111':(original,original_training),'direction134':(augmented,augmented_training)}


def valid(binding):
    assert e.sha(binding['path'])==binding['sha256'],binding['path']
    return binding


def auth():
    p=e.read(DEST/'FROZEN_PROTOCOL.json')
    for b in p['sources']+[p[k] for k in ('audit','audit_closure','input_cases','targets','baseline_closure')]:
        valid(b)
    return p


def prepare():
    audit=e.read(e.REC/'input_boundary_audit_v246/RESULT.json')
    assert audit['all32_cache_identity_exact'] and audit['TRAIN_ranges']==169
    closed=e.REC/'input_boundary_audit_v246/CLOSURE.json';assert e.read(closed)['complete']
    DEST.mkdir(exist_ok=True);OUTPUT.mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','refs/heads/codex/curve-only-01a1ac7c'],text=True).strip()
    modules=(augmented,augmented_training,sys.modules[__name__])
    for module in modules:
        path=Path(module.__file__).resolve()
        assert path.read_bytes()==subprocess.check_output(['git','show',commit+':'+str(path.relative_to(e.ROOT))])
    original_cases=e.read(e.REC/'round34/INPUTS.json')['cases']
    prior={c['source_id']:c for c in e.read(e.REC/'material_direction_v161/ALL_INPUTS_FROZEN.json')['cases']}
    target={c['source_id']:c for c in e.read(e.REC/'round34/FIT_TARGETS.json')['cases']}
    records=[];fitted=[]
    for c in audit['cases']:
        sid=c['source_id'];packet=e.read(valid(c['geometry'])['path'])
        cached=prior[sid];values=np.load(valid(cached['features'])['path'])
        packet=dict(packet,local_material_direction_v160=dict(values=values.tolist(),receipt=cached['receipt']))
        folder=OUTPUT/'features'/sid;folder.mkdir(parents=True,exist_ok=True)
        e.write(folder/'geometry.json',packet)
        if not (folder/'features.npy').exists():(folder/'features.npy').symlink_to(Path(c['raw_features']['path']).resolve())
        e.write(folder/'RECEIPT.json',dict(complete=True,input_only=True,geometry=e.bind(folder/'geometry.json'),
            features=c['raw_features'],source_direction=cached['features'],native_geometry_unchanged=True))
        rows=augmented.candidate_rows(np.load(c['raw_features']['path']),packet)
        baseline=original.candidate_rows(np.load(c['raw_features']['path']),packet)
        certified=np.load(valid(c['direction_grid'])['path'])
        for i,(a,b) in enumerate(zip(rows,baseline)):
            assert np.array_equal(a['grid'][:,:111],b['grid'])
            lo,hi=certified['offsets'][i:i+2]
            assert np.array_equal(a['grid'][:,111:],certified['values'][lo:hi])
            for key in ('length_mm','grid_step_mm','closed'):
                assert a[key]==b[key]
        record=dict(c,geometry=e.bind(folder/'geometry.json'),features=c['raw_features'])
        records.append(record)
        if c['split']=='TRAIN':
            fitted.append(dict(target[sid],geometry=record['geometry'],features=record['features']))
        print('DIRECTION_SPAN_INPUT_READY',sid,len(rows),flush=True)
    assert len(records)==32 and len(fitted)==22
    e.write(DEST/'INPUTS.json',dict(cases=records,no_labels_or_models_used_for_observations=True))
    e.write(DEST/'FIT_TARGETS.json',dict(cases=fitted,TRAIN_only=True,original169ranges_unchanged=True))
    families=e.read(e.REC/'proposal_quality_control_v241/FROZEN_PROTOCOL.json')['families']
    assert len(families)==18
    cases={c['source_id']:c for c in original_cases};folds=[]
    for held in range(4):
        train=[c for c in fitted if families[cases[c['source_id']]['family']]!=held]
        test=[c for c in records if c['split']=='TRAIN' and families[c['family']]==held]
        assert train and test and not ({c['source_id'] for c in train}&{c['source_id'] for c in test})
        path=DEST/'folds'/f'{held:02d}';path.mkdir(parents=True,exist_ok=True)
        e.write(path/'TRAIN.json',dict(cases=train))
        e.write(path/'HELD_INPUTS.json',dict(cases=test))
        folds.append(dict(index=held,train=e.bind(path/'TRAIN.json'),held=e.bind(path/'HELD_INPUTS.json'),
            held_families=sorted(f for f,k in families.items() if k==held)))
    old=e.read(e.REC/'round34/FROZEN_PROTOCOL.json')
    e.write(DEST/'FROZEN_PROTOCOL.json',dict(source_commit=commit,
        sources=old['sources']+[e.bind(m.__file__) for m in modules],
        audit=e.bind(e.REC/'input_boundary_audit_v246/RESULT.json'),audit_closure=e.bind(closed),
        baseline_closure=e.bind(e.REC/'round34/run/CLOSURE.json'),
        input_cases=e.bind(DEST/'INPUTS.json'),targets=e.bind(DEST/'FIT_TARGETS.json'),folds=folds,
        families=families,variants=list(VARIANTS),settings={k:t.SETTINGS for k,(m,t) in VARIANTS.items()},
        design='Fresh111/134ordered span networks on identical fourTRAINfamily complements, same128epochs/169original representations/objective. All shared initial parameters identical; new23input weights zero. Append directions while retaining solid74, unlike oldfailed399localHGB. ExistingR34fittedqualityhead is NOTused: it has seen heldTRAINlabels.',
        labels_released_only_after_all44_exports_frozen=True,original_G121_retained=True,
        previous_direction399_failure='34/121FP73 vs238 41FP70 andsolid756 44FP80; no claim this feature was previously successful.',
        observed_motivation='TRAIN56native excluded intervals/9families;3families have distance/direction variation where sampledsolid74is equal. Existing111is not proven physically insufficient; variation is not a welding/torch permission rule.',
        nonoverlap_grammar_not_changed='Two literal alias representations contain2.5994e-6mmnative truth projection overlaps. No new disjointness/tolerance rule.',
        decision='Proceed to full22fit/all32VAL only if direction134 improves complete-outputF1 over fresh111 with moreTP or fewerFP; retain all gained/lost and category outcomes. NoVALfit/mining/calibration.',
        CPU_threads_per_job=2,maximum_workers=3,reserve_per_job_bytes=3*1024**3,
        other_future_reserve_bytes=2*1024**3,CPU_fraction_ceiling=.8,memory_fraction_ceiling=.9,
        no_speed_or_contract_claim=True,epoch=time.time()))
    print('DIRECTION_SPAN_FAMILY_PROTOCOL',commit,flush=True)


def init_control():
    p=auth();cases=e.read(p['targets']['path'])['cases']
    augmented_training.configure();rows,mean,scale,masses,details=augmented_training.assemble(cases)
    # Same native support and same original labels: inherit the237 full teacher
    # geometry proof, with an exact all169 encoding identity check here.
    target={c['source_id']:c['chosen'] for c in cases};count=0
    for row in rows:
        chosen=[c for c in target[row['source_id']] if c['candidate']==row['candidate']]
        for reverse in (False,True):
            assert augmented.sequence(row,chosen,reverse)==original.sequence(row,chosen,reverse)
        count+=len(chosen)
    assert count==169 and augmented.sequence is original.sequence and augmented.codec is original.codec
    indices=[]
    for predicate in [lambda r:bool(r['sequence']),lambda r:r['closed'] and bool(r['sequence']),
            lambda r:len(r['sequence'])>1,lambda r:not r['sequence']]:
        options=[i for i,r in enumerate(rows) if predicate(r)]
        if options:
            i=max(options,key=lambda k:(len(rows[k]['sequence']),len(rows[k]['grid'])))
            if i not in indices:indices.append(i)
    torch.manual_seed(20261106);baseline=original.OrderedSpanNetwork().eval()
    torch.manual_seed(20261106);new=augmented.OrderedSpanNetwork().eval()
    for key,value in baseline.state_dict().items():
        compare=new.state_dict()[key][:,:111] if key=='input.weight' else new.state_dict()[key]
        assert torch.equal(value,compare),key
    assert torch.count_nonzero(new.input.weight[:,111:])==0
    selected=[rows[i] for i in indices]
    legacy=[dict(r,grid=r['grid'][:,:111]) for r in selected]
    maximum=0.
    with torch.no_grad():
        for reverse in (False,True):
            a=baseline.teacher(original.pack(legacy,mean[:111],scale[:111],'cpu',reverse))
            b=new.teacher(augmented.pack(selected,mean,scale,'cpu',reverse))
            maximum=max(maximum,max(float((a[k]-b[k]).abs().max()) for k in a))
    assert maximum<1e-5,maximum
    augmented_training.assembly_control(cases,DEST/'functional')
    e.write(DEST/'INIT_CONTROL.json',dict(passed=True,all_shared_initial_parameters_bit_exact=True,
        additional_input_weights_zero=True,initial_teacher_field_max_delta=maximum,comparison_tolerance=1e-5,
        original169codes_and_native_grammar_identical=True,inherited_full_teacher_geometry=e.bind(e.REC/'ordered_span_controls_v237/CLOSURE.json'),
        actual_new_GPU_backward=e.bind(DEST/'functional/ASSEMBLY_CONTROL.json'),
        baseline_parameters=sum(x.numel() for x in baseline.parameters()),new_parameters=sum(x.numel() for x in new.parameters()),
        no_optimizer_step_or_saved_random_model=True))
    print('DIRECTION_SPAN_INITIALIZATION_CONTROL_PASSED',maximum,flush=True)


def raw_control():
    p=auth();sid='curve_0e1e2c230f78c505'
    case=next(c for c in e.read(e.REC/'round34/INPUTS.json')['cases'] if c['source_id']==sid)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            value=os.fsdecode(args[0])
            if any(s in value for s in ('/truth/','/targets/','/score_coordinate_maps/')):
                raise PermissionError('Raw134input control cannot read answers: '+value)
        e.fence(event,args)
    sys.addaudithook(guard)
    start=time.perf_counter();features,packet=augmented.input_curves(case)
    cached=next(c for c in e.read(p['input_cases']['path'])['cases'] if c['source_id']==sid)
    expected=np.load(cached['features']['path']);reference=e.read(cached['geometry']['path'])
    assert np.array_equal(features,expected)
    assert packet['offsets']==reference['offsets']
    assert all(all(a.get(k)==b.get(k) for k in ('points','stations','station_s_mm','material','source_material_path'))
        for a,b in zip(packet['candidates'],reference['candidates']))
    actual=np.asarray(packet['local_material_direction_v160']['values'],np.float32)
    prior=np.asarray(reference['local_material_direction_v160']['values'],np.float32)
    assert np.array_equal(actual,prior)
    a,b=augmented.candidate_rows(features,packet),augmented.candidate_rows(expected,reference)
    assert all(np.array_equal(x['grid'],y['grid']) for x,y in zip(a,b))
    e.write(DEST/'RAW_INPUT_CONTROL.json',dict(passed=True,source_id=sid,raw_input=case['raw'],STEP=case['STEP'],
        raw108_and_direction23_and_all134grids_bit_exact=True,candidates=len(a),stations=len(features),
        actual_raw_input_builder_executed=True,seconds=time.perf_counter()-start,
        no_NN_or_CSV_or_full_cold_pipeline_time_claim=True,no_labels_or_models_read=True))
    print('DIRECTION_SPAN_RAW134_INPUT_CONTROL_PASSED',len(features),flush=True)


def fit_job(index,variant):
    p=auth();fold=p['folds'][index];runtime,training=VARIANTS[variant]
    assert e.read(DEST/'INIT_CONTROL.json')['passed'] and e.read(DEST/'RAW_INPUT_CONTROL.json')['passed']
    held=e.read(valid(fold['held'])['path'])['cases'];cases=e.read(valid(fold['train'])['path'])['cases']
    forbidden={c['source_id'] for c in held}|{c['source_id'] for c in e.read(p['input_cases']['path'])['cases'] if c['split']=='VAL'}
    active=[True]
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            path=Path(os.fsdecode(args[0]));value=str(path)
            if '/truth/' in value or '/score_coordinate_maps/' in value or '/supplement_complete_eval_v130/' in value:
                raise PermissionError('Family fit/forward cannot read answers: '+value)
            if active[0] and any(sid in value for sid in forbidden):
                raise PermissionError('Family fit cannot read held or VAL input: '+value)
            if path.name=='model.pkl' and not str(path.resolve()).startswith(str(DEST.resolve())+'/'):
                raise PermissionError('Fresh family model cannot read previous fitted models: '+value)
        e.fence(event,args)
    sys.addaudithook(guard)
    target=DEST/'folds'/f'{index:02d}'/variant;target.mkdir(exist_ok=True)
    network,mean,scale,fit=training.fit(cases,target/'fit')
    model=runtime.OrderedSpanSet(network,mean,scale)
    with (target/'model.pkl').open('wb') as file:pickle.dump(model,file,protocol=5)
    with (target/'model.pkl').open('rb') as file:model=pickle.load(file)
    active[0]=False
    outputs=[]
    for case in held:
        sid=case['source_id'];observed=np.load(valid(case['features'])['path']);packet=e.read(valid(case['geometry'])['path'])
        decoded=runtime.decode(observed,packet,model)
        output=export.publish(OUTPUT/'folds'/f'{index:02d}'/variant/sid,sid,decoded)
        outputs.append(output)
    e.write(target/'OUTPUTS_FROZEN.json',dict(complete=True,variant=variant,fold=index,cases=outputs,
        model=e.bind(target/'model.pkl'),fit=e.bind(target/'fit/FIT.json'),held_families=fold['held_families'],
        fit_ids=sorted(c['source_id'] for c in cases),held_ids=sorted(c['source_id'] for c in held),
        no_held_or_VALfit_or_quality_head_reuse=True,epoch=time.time()))
    print('DIRECTION_SPAN_FOLD_OUTPUTS_FROZEN',index,variant,len(outputs),flush=True)


def resources():
    maximum=int(Path('/sys/fs/cgroup/memory.max').read_text());memory=int(Path('/sys/fs/cgroup/memory.current').read_text())
    quota,period=map(int,Path('/sys/fs/cgroup/cpu.max').read_text().split())
    def usage():return int(dict(x.split() for x in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())['usage_usec'])
    a,begin=usage(),time.monotonic();time.sleep(2)
    return dict(memory=memory,maximum=maximum,cpu=(usage()-a)/1e6/(time.monotonic()-begin),quota=quota/period)


def family():
    p=auth();assert e.read(DEST/'RAW_INPUT_CONTROL.json')['passed']
    pending=[(index,variant) for index in range(4) for variant in VARIANTS]
    active={};completed=[];failed=[];not_started=[]
    while pending or active:
        for key,item in list(active.items()):
            code=item['process'].poll()
            if code is None:continue
            item['log'].close();info=dict(item['launch'],exit_code=code,actual_subprocess_wait=True,ended=time.time())
            e.write(item['folder']/'ACTUAL_TERMINAL.json',info)
            if code:
                failed.append((key,code));not_started.extend(pending);pending=[]
            else:completed.append(key)
            del active[key]
            print('FAMILY_JOB_ACTUAL_TERMINAL',key,code,flush=True)
        if pending and len(active)<p['maximum_workers']:
            resource=resources()
            # Reserve outstanding growth of every currently live job as well.
            admitted=(resource['memory']+(len(active)+1)*p['reserve_per_job_bytes']+p['other_future_reserve_bytes']<.9*resource['maximum']
                and resource['cpu']+p['CPU_threads_per_job']<.8*resource['quota'])
            if admitted:
                index,variant=pending.pop(0);folder=DEST/'folds'/f'{index:02d}'/variant;folder.mkdir(exist_ok=True)
                assert not (folder/'LAUNCH.json').exists(),'Existing job requires actual process/receipt recovery, never duplicate'
                command=[sys.executable,'-B','-u','-m',__name__ if __name__!='__main__' else 'tools.curveonly_01a1ac7c_direction_span_family_v249','fit_job',str(index),variant]
                log=(folder/'run.log').open('w');process=subprocess.Popen(command,cwd=e.ROOT,stdout=log,stderr=subprocess.STDOUT)
                launch=dict(pid=process.pid,command=command,started=time.time(),resource_admission=resource)
                e.write(folder/'LAUNCH.json',launch)
                active[(index,variant)]=dict(process=process,launch=launch,log=log,folder=folder)
                print('FAMILY_JOB_STARTED',index,variant,process.pid,flush=True)
        e.write(DEST/'FAMILY_PROGRESS.json',dict(completed=completed,pending=pending,
            active=[dict(index=k[0],variant=k[1],**v['launch']) for k,v in active.items()],epoch=time.time()))
        if pending or active:time.sleep(5)
    if failed:
        e.write(DEST/'FAMILY_FAILURE_REVIEW_REQUIRED.json',dict(failed=failed,not_started=not_started,all_started_jobs_have_actual_wait_receipts=True))
        raise RuntimeError('Family control failed; completed evidence preserved, all live jobs allowed to close: '+str(failed))
    frozen={variant:[] for variant in VARIANTS}
    for index in range(4):
        for variant in VARIANTS:
            folder=DEST/'folds'/f'{index:02d}'/variant
            assert e.read(folder/'ACTUAL_TERMINAL.json')['exit_code']==0
            output=e.read(folder/'OUTPUTS_FROZEN.json');assert output['complete']
            for c in output['cases']:valid(c['prediction']);valid(c['csv'])
            frozen[variant]+=output['cases']
    assert all(len(c)==22 and len({x['source_id'] for x in c})==22 for c in frozen.values())
    e.write(DEST/'OUTPUTS_FROZEN.json',dict(complete=True,variants=frozen,total_file_outputs=44,
        all44_outputs_closed_before_final_scoring=True,no_VALoutputs_or_quality_head_reuse=True,epoch=time.time()))
    print('FAMILY_ALL44_OUTPUTS_FROZEN',flush=True)


def score():
    p=auth();frozen=e.read(DEST/'OUTPUTS_FROZEN.json');assert frozen['all44_outputs_closed_before_final_scoring']
    targets={c['source_id']:c for c in e.read(p['targets']['path'])['cases']}
    results={};totals={}
    for variant,cases in frozen['variants'].items():
        rows=[]
        for c in cases:
            row=export.score_one(dict(c,truth=targets[c['source_id']]['truth']))
            rows.append(row);print('DIRECTION_SPAN_HELD_SCORE',variant,c['source_id'],row['metrics']['TP'],row['metrics']['FP'],flush=True)
        results[variant]=rows;totals[variant]=export.full.total(rows)
        t=totals[variant];t['F1']=2*t['TP']/(t['G']+t['P'])
    before={(c['source_id'],m['truth']) for c in results['original111'] for m in c['metrics']['matches']}
    after={(c['source_id'],m['truth']) for c in results['direction134'] for m in c['metrics']['matches']}
    a,b=totals['original111'],totals['direction134']
    proceed=b['F1']>a['F1'] and (b['TP']>a['TP'] or b['FP']<a['FP'])
    e.write(DEST/'RESULT.json',dict(complete=True,totals=totals,cases=results,gained=sorted(after-before),lost=sorted(before-after),
        proceed_full22fit_and_all32VAL=proceed,all_original_G121_and_roles_retained=True,
        no_VALfit_mining_or_calibration=True,quality_head_not_used=True,epoch=time.time()))
    print('DIRECTION_SPAN_FAMILY_DECISION',proceed,totals,flush=True)


if __name__=='__main__':
    action=sys.argv[1]
    with threadpool_limits(limits=2):
        if action=='fit_job':fit_job(int(sys.argv[2]),sys.argv[3])
        else:dict(prepare=prepare,init_control=init_control,raw_control=raw_control,family=family,score=score)[action]()
