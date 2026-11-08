"""Whole-output first failures for the local finite-interval family control."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import sys,time,pickle,csv,numbers,multiprocessing
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_local_family_v99 as experiment
from tools import curveonly_01a1ac7c_score_v2 as geometry_score
from tools import curveonly_01a1ac7c_complete_eval_v60 as evaluation
from weld_pipeline import curveonly_01a1ac7c_local_runtime_v96 as runtime
from weld_pipeline import curveonly_01a1ac7c_local_intervals_v95 as local
R,O=e.REC,e.OUT;DEST=experiment.DEST


def equal(a,b):
    if isinstance(a,dict)and isinstance(b,dict):return set(a)==set(b)and all(equal(a[k],b[k])for k in a)
    if isinstance(a,list)and isinstance(b,list):return len(a)==len(b)and all(equal(x,y)for x,y in zip(a,b))
    if isinstance(a,numbers.Real)and isinstance(b,numbers.Real):return abs(float(a)-float(b))<=1e-9
    return a==b


def exports(binding):
    assert e.sha(binding['prediction']['path'])==binding['prediction']['sha256'];assert e.sha(binding['csv']['path'])==binding['csv']['sha256']
    decoded=e.read(binding['prediction']['path']);expected=[(p,s)for p in decoded['parents']for s in p['segments']]
    with Path(binding['csv']['path']).open(encoding='utf-8',newline='')as f:rows=list(csv.DictReader(f))
    assert len(rows)==len(expected)
    for row,(parent,segment)in zip(rows,expected):
        assert row['parent_id']==parent['parent_id']and int(row['segment_index'])==segment['segment_index']
        for key,index in [('start',0),('end',-1)]:
            assert np.allclose([float(row[key+'_'+c])for c in 'xyz'],segment['points'][index],atol=1e-9,rtol=0)
        assert abs(float(row['length_mm'])-segment['length_mm'])<=1e-9
    return dict(source_id=binding['source_id'],prediction=binding['prediction'],csv=binding['csv'],published_segments=len(rows),hash_and_CSV_readback=True)


def one(fold,case,baseline_trace):
    sid=case['source_id'];name=fold['name'];packet=e.read(O/name/'features'/sid/'geometry.json');features=np.load(O/name/'features'/sid/'features.npy')
    with (R/name/'run/model.pkl').open('rb')as f:model=pickle.load(f)
    truth=e.read(evaluation.DEST/'truth'/sid/'COMPLETE_UNITS.json')['units'];baseline_entry=next(q for q in e.read(fold['baseline_outputs']['path'])['cases']if q['source_id']==sid)
    with threadpool_limits(limits=2):
        baseline_replay=runtime.decode(features,packet,model.baseline);baseline=e.read(baseline_entry['prediction']['path'])
        assert equal(baseline_replay['parents'],baseline['parents']),(sid,'baseline changed')
        raw=model.predict(features,packet);old=model.baseline.predict(features,packet);new=model.local.predict(features,packet)
        decoded=runtime.decode(features,packet,model);frozen=e.read(case['prediction']['path'])
        assert equal(decoded['parents'],frozen['parents']),(sid,'refinement replay changed')
        assert len(raw)==frozen['finite_source_selection']['input_proposals']
        probability=model.local.classifier.predict_proba(local.encode(features,packet))
    counts=Counter();traces=[];old_groups={};local_groups={};chosen_groups={}
    for rows,group in [(old,old_groups),(new,local_groups),(raw,chosen_groups)]:
        for q in rows:group.setdefault(q['candidate'],[]).append(q)
    score=e.read(evaluation.DEST/'scores'/name/'cases'/sid/'SCORE.json');matched={q['truth']for q in score['metrics']['matches']}
    source_decisions=[]
    for ci in range(len(packet['candidates'])):
        lo,hi=packet['offsets'][ci:ci+2];p=probability[lo:hi];cls=model.local.classifier.classes_
        source_decisions.append(dict(candidate=ci,baseline_raw_count=len(old_groups.get(ci,[])),local_raw_count=len(local_groups.get(ci,[])),
            selected='LOCAL'if len(local_groups.get(ci,[]))>len(old_groups.get(ci,[]))else'BASELINE',
            selected_count=len(chosen_groups.get(ci,[])),local_intervals=model.local.last_states[ci]['intervals'],
            local_probability_mean=dict(zip(map(str,cls),map(float,p.mean(0)))),
            local_argmax_rows=dict(Counter(map(str,cls[p.argmax(1)])))))
    legal={};old_trace=e.read(baseline_trace['path'])['traces']
    assert len(old_trace)==len(truth)
    for entry in old_trace:
        ti=entry['truth'];options=[q['legal_interval']for q in entry['legal_candidates']];legal[ti]={q['candidate']for q in options}
        evidence=[]
        for ci in sorted(legal[ti]):
            evidence.append(dict(source_decisions[ci],legal_intervals=[q for q in options if q['candidate']==ci],
                selected_raw_complete_match=any(geometry_score.geometry_error(q,truth[ti])is not None for q in chosen_groups.get(ci,[])),
                local_raw_complete_match=any(geometry_score.geometry_error(q,truth[ti])is not None for q in local_groups.get(ci,[]))))
        reason='UNKNOWN'
        if ti in matched:reason='COMPLETE_GEOMETRY_TP'
        elif not options:reason='NO_SINGLE_LEGAL_COMPLETE_CARRIER'
        elif any(q['selected_raw_complete_match']for q in evidence):reason='SOURCE_SELECTION_OR_EXPORT_REJECTS_COMPLETE_RAW_INTERVAL'
        elif any(q['local_raw_complete_match']and q['selected']=='BASELINE'for q in evidence):reason='COUNT_GROWTH_POLICY_RETAINS_WRONG_GLOBAL_RANGE'
        elif not any(q['selected_count']for q in evidence):reason='BOTH_MODELS_REJECT_ALL_LEGAL_SOURCES'
        else:reason='SELECTED_INTERVAL_RANGE_OR_TYPE_MISMATCH'
        counts[reason]+=1;traces.append(dict(truth=ti,reason=reason,evidence=evidence,physical_or_process_cause='UNKNOWN'))
    matched_predictions={q['prediction']for q in score['metrics']['matches']};false=[];all_legal=set().union(*legal.values())
    for pi,unit in enumerate(score['prediction_groups']):
        if pi in matched_predictions:continue
        candidates=sorted({q['source_candidate']for q in unit['members']})
        false.append(dict(prediction_group=pi,candidates=candidates,
            source_decisions=[source_decisions[ci]for ci in candidates],
            geometry_class='LEGAL_COMPLETE_GT_CARRIER_WRONG_RANGE_TYPE_OR_SELECTION'if any(ci in all_legal for ci in candidates)else'NO_FULL_LEGAL_GT_ON_PUBLISHED_SOURCE',
            physical_or_process_cause='UNKNOWN'))
    dest=R/name/'run/first_failure'/sid;e.write(dest/'TRACE_LOCAL_V101.json',dict(source_id=sid,counts=dict(counts),traces=traces,
        false_positives=false,source_decisions=source_decisions,baseline_forward_exact_semantics_numeric1e9=True,
        refined_forward_exact_semantics_numeric1e9=True,raw_proposal_count_parity=True,
        physical_or_business_causes_not_assumed=True,source=e.bind(__file__)))
    return dict(source_id=sid,name=name,counts=dict(counts),false_positives=false,trace=e.bind(dest/'TRACE_LOCAL_V101.json'))


def run():
    result=e.read(DEST/'RESULT.json');frozen=e.read(DEST/'ALL_OUTPUTS_FROZEN.json');assert result['complete']and frozen['complete']
    baseline=e.read(R/'family_graph_v67/FIRST_FAILURE_REVIEW.json');trace={q['source_id']:q['precise_trace']for q in baseline['cases']if q['name'].endswith('/graph')}
    rows=[];checks=[];jobs=[]
    for fold in frozen['folds']:
        for name in ('baseline_outputs','frozen_outputs'):
            for case in e.read(fold[name]['path'])['cases']:checks.append(exports(case))
        for case in e.read(fold['frozen_outputs']['path'])['cases']:jobs.append((fold,case,trace[case['source_id']]))
    assert len(checks)==44 and len(jobs)==22
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1)as pool:
        for f in as_completed([pool.submit(one,*q)for q in jobs]):
            rows.append(f.result());e.write(DEST/'REVIEW_PROGRESS.json',dict(complete=len(rows),total=22))
    totals=Counter();geometry=Counter();source=Counter()
    for row in rows:
        totals.update(row['counts'])
        for fp in row['false_positives']:
            geometry[fp['geometry_class']]+=1
            changed=any(q['selected']=='LOCAL'for q in fp['source_decisions'])
            source['LOCAL_REFINEMENT_SELECTED'if changed else'BASELINE_INTERVAL_RETAINED']+=1
    assert sum(totals.values())==121
    e.write(DEST/'FIRST_FAILURE_REVIEW.json',dict(complete=True,cases=rows,first_failure_totals=dict(totals),
        false_positive_geometry=dict(geometry),false_positive_source=dict(source),all22_baseline_and_refined_forward_replayed=True,
        all22_raw_proposal_counts_replayed=True,all44_export_hashes_and_CSV_readback=True,exports=checks,
        result=e.bind(DEST/'RESULT.json'),no_model_output_or_threshold_change=True,no_VALread=True,source=e.bind(__file__),epoch=time.time()))
    print('LOCAL_FIRST_FAILURES',dict(totals),'FP',dict(geometry),dict(source),flush=True)


def visual():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=e.read(DEST/'RESULT.json');frozen=e.read(DEST/'ALL_OUTPUTS_FROZEN.json');assert e.read(DEST/'FIRST_FAILURE_REVIEW.json')['complete']
    lookup={case['source_id']:fold['name']for fold in frozen['folds']for case in e.read(fold['frozen_outputs']['path'])['cases']}
    before={c['source_id']:c for c in result['cases']['baseline']};after={c['source_id']:c for c in result['cases']['local']}
    selected=[(sid,ti,'GAIN')for sid,ti in result['gained'][:2]]+[(sid,ti,'LOSS')for sid,ti in result['lost'][:1]]
    if not selected:
        order=sorted(after,key=lambda sid:after[sid]['metrics']['FP']-before[sid]['metrics']['FP'],reverse=True)
        selected=[(sid,0,'FP_DIAGNOSTIC')for sid in order[:3]]
    fig,axes=plt.subplots(1,len(selected),figsize=(6*len(selected),5),squeeze=False);details=[]
    for ax,(sid,ti,kind)in zip(axes.ravel(),selected):
        truth=e.read(evaluation.DEST/'truth'/sid/'COMPLETE_UNITS.json')['units'];reference=np.asarray(truth[ti]['points'])
        score=e.read(evaluation.DEST/'scores'/lookup[sid]/'cases'/sid/'SCORE.json');match=next((q for q in score['metrics']['matches']if q['truth']==ti),None)
        pi=match['prediction']if match else next((j for j in range(len(score['prediction_groups']))if j not in {q['prediction']for q in score['metrics']['matches']}),None)
        prediction=np.asarray(score['prediction_groups'][pi]['points'])if pi is not None else reference[:0]
        focus=np.vstack([reference,prediction]);center=focus.mean(0);_,_,basis=np.linalg.svd(focus-center,full_matrices=False);project=lambda x:(np.asarray(x)-center)@basis[:2].T
        for j,q in enumerate(truth):
            xy=project(q['points']);ax.plot(xy[:,0],xy[:,1],c='0.7',linewidth=3,label='Reference'if j==0 else None)
        if len(prediction):
            xy=project(prediction);ax.plot(xy[:,0],xy[:,1],c='royalblue'if match else'firebrick',linewidth=1.5,label='New TP'if match else'New actual FP')
        old=before[sid]['metrics'];new=after[sid]['metrics'];ax.set_title(f"{kind} {sid[-8:]} / truth {ti}\nTP {old['TP']} -> {new['TP']}; FP {old['FP']} -> {new['FP']}")
        ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8);ax.set_xlabel('Display plane / mm');ax.set_ylabel('Display plane / mm')
        details.append(dict(source_id=sid,truth=ti,state=kind,displayed_prediction_group=pi,actual_match=match))
    fig.suptitle('TRAIN-family complete final outputs: local finite interval refinement\nProjection for display only; strict original 3D V60 scoring',fontsize=12);fig.tight_layout()
    path=DEST/'FINAL_GEOMETRY_VISUAL.png';fig.savefig(path,dpi=140);plt.close(fig)
    e.write(DEST/'VISUAL_INTERPRETATION.json',dict(figure=e.bind(path),details=details,source=e.bind(__file__),no_evaluation_registration=True))
    print('LOCAL_FAMILY_VISUAL',flush=True)


if __name__=='__main__':dict(run=run,visual=visual)[sys.argv[1]]()
