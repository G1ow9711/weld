"""Frozen all-curve preparation, fresh compact fit, full outputs then scoring."""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
import argparse, concurrent.futures, csv, json, math, multiprocessing, pickle, time, traceback
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from tools.curveonly_01a1ac7c_inventory import ROOT,REC,read,write,bind,sha
from weld_pipeline import curveonly_01a1ac7c as core

OUT=ROOT/'server_rebuild/CURVEONLY_01a1ac7c_20261106_v1'

def excel_types(record):
    import openpyxl
    result={};evidence=[];curve_rows=[]
    for binding in record['excel']:
        wb=openpyxl.load_workbook(binding['path'],read_only=True,data_only=True)
        for sheet in wb:
            rows=iter(sheet.values);headers=[str(v or '')for v in next(rows,[])]
            ids=[i for i,h in enumerate(headers)if 'ObjectId'in h]
            ty=[i for i,h in enumerate(headers)if 'WeldType'in h]
            if not ids or not ty:continue
            for row in rows:
                value=str(row[ty[0]])
                result[str(row[ids[0]])]='vertical'if 'Vertical'in value or '立焊'in value else 'horizontal'if 'Horizontal'in value or '平焊'in value else 'UNKNOWN'
                hasmid=[i for i,h in enumerate(headers)if 'HasMid'in h]
                if hasmid and row[hasmid[0]]:
                    def index(text):return next(i for i,h in enumerate(headers)if h.split('\n')[-1]==text)
                    start=[float(row[index('StartMatrix.Translation.'+a)])for a in 'XYZ'];end=[float(row[index('EndMatrix.Translation.'+a)])for a in 'XYZ']
                    length=float(row[index('Length')]);curve_rows.append(dict(id=str(row[ids[0]]),ends=[start,end],length=length,weld_type=result[str(row[ids[0]])]))
        wb.close()
    value=read(record['label']['path'])
    if 'welds'in value:
        missing=[w for w in value['welds']if w['id']not in result and any(s['kind']!='line'for s in w['segments'])]
        if missing and curve_rows:
            a=np.array([[w['segments'][0]['points'][0],w['segments'][-1]['points'][-1]]for w in missing]);b=np.array([r['ends']for r in curve_rows])
            costs=np.minimum(np.linalg.norm(a[:,None]-b[None],axis=3).max(2),np.linalg.norm(a[:,None]-b[None,:,::-1],axis=3).max(2))
            lengths=np.array([sum(s['lengthMm']for s in w['segments'])for w in missing]);other=np.array([r['length']for r in curve_rows]);valid=(costs<=1e-4)&(abs(lengths[:,None]-other)<=1e-4)
            for i,w in enumerate(missing):
                js=np.flatnonzero(valid[i])
                if len(js)!=1 or valid[:,js[0]].sum()!=1:continue
                j=int(js[0]);result[w['id']]=curve_rows[j]['weld_type'];evidence.append(dict(JSON_id=w['id'],Excel_id=curve_rows[j]['id'],endpoint_max_mm=float(costs[i,j]),length_error_mm=float(abs(lengths[i]-other[j])),unique_one_to_one=True,source_ids_retained=True))
    if evidence:write(REC/'label_association'/record['source_id']/'EXCEL_JSON_ID_ASSOCIATION.json',dict(source=record['label'],excel=record['excel'],associations=evidence,not_a_prediction_transform=True,coordinate_transform_fitted=False))
    return result

def representation_points(rep):
    if rep['kind']=='line':return np.array(rep['endpoints_mm'],float)
    if 'arc_parameters'in rep:
        a=rep['arc_parameters'];c=np.array(a['center_mm']);u=np.array(a['start_direction']);n=np.array(a['plane_normal']);v=np.cross(n,u);r=a['radius_mm'];sweep=a['sweep_rad']
        theta=np.linspace(0,sweep,max(5,int(np.ceil(abs(sweep)*r/.5))+1))
        return c+r*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)
    return np.array(rep.get('points_mm',rep.get('points')),float)

def truth_record(r):
    typed=excel_types(r);binding=r.get('reviewed_target',r['label']);value=read(binding['path']);runs=[];children=[]
    canonical='parents'in value and 'welds'not in value
    parents=value['parents'if canonical else 'welds']
    for parent in parents:
        pid=parent.get('original_parent_id',parent.get('id'));current=[];current_children=[];types=[]
        def flush():
            nonlocal current,current_children,types
            if current:
                points=np.concatenate(current);runs.append(dict(id=str(pid)+':'+str(len(runs)),parent_id=pid,points=points.tolist(),weld_type=types[0],children=list(current_children),closed=bool(np.linalg.norm(points[0]-points[-1])<.1),circle=core.circle_fit(points)))
            current=[];current_children=[];types=[]
        for i,child in enumerate(parent['children'if canonical else 'segments']):
            kind=child['original_kind'if canonical else 'kind']
            if kind=='line':flush();continue
            p=representation_points(child['representation'])if canonical else np.array(child['points'],float)
            wt=child.get('weld_type',typed.get(str(pid),'UNKNOWN'))
            if wt=='UNKNOWN':raise ValueError('Unknown reviewed weld type '+str(pid))
            gap=np.linalg.norm(current[-1][-1]-p[0]) if current else 0.
            if current and (gap>.2 or wt!=types[-1]):flush()
            ci=len(children);children.append(dict(id=child.get('original_child_id',str(pid)+':'+str(i)),parent_id=pid,points=p.tolist(),weld_type=wt,circle=core.circle_fit(p)))
            current.append(p);current_children.append(ci);types.append(wt)
        flush()
    return dict(source_id=r['source_id'],runs=runs,children=children,source=binding,unit='mm',no_predicted_geometry_in_truth=True)

def prepare():
    inv=read(REC/'SOURCE_INVENTORY.json');cases=[];truth=[]
    oldroot=ROOT/'experiment_records/STEPINPUT_01a19c46_round52_source_gate_20261105_v1/full_evaluation_v1/input_plans'
    for r in inv['inputs']:
        if r['split']=='PENDING_AUDIT':raise ValueError('No difficult input silently removed')
        T=None;source=None;plan=oldroot/r['key']/'INPUT_ONLY_PLAN.json'
        if plan.exists():p=read(plan);T=p.get('transform');source=bind(plan)
        elif r.get('pose')and r['pose'].get('coordinate_config'):
            T=r['pose']['coordinate_config'].get('transform_STEP_to_XYZ');source=r['pose']['coordinate_config'].get('source')
        record=dict(source_id=r['source_id'],name=r['name'],key=r['key'],split=r['split'],family=r['split_family'],raw=r['raw'],STEP=r['STEP'],transform=T,transform_source=source)
        # Workers receive this manifest only, with no labels or old models.
        cases.append(record);t=truth_record(r);target=REC/'truth'/r['source_id']/'CURVE_TRUTH.json';write(target,t)
        truth.append(dict(source_id=r['source_id'],split=r['split'],family=r['split_family'],truth=bind(target),geometry_units=len(t['runs']),segments=len(t['children']),historical_roles=r['previous_roles']))
    assert not ({c['family']for c in cases if c['split']=='TRAIN'}&{c['family']for c in cases if c['split']=='VAL'})
    write(REC/'INPUTS.json',dict(cases=cases,labels_or_fitted_models=False,materials_and_XYZ_only=True))
    write(REC/'TRUTH_INDEX.json',dict(cases=truth,census_labels_exposed=True,never_read_by_inference=True))
    summary={split:dict(inputs=sum(c['split']==split for c in cases),families=len({c['family']for c in cases if c['split']==split}),geometry_units=sum(c['geometry_units']for c in truth if c['split']==split),segments=sum(c['segments']for c in truth if c['split']==split))for split in ('TRAIN','VAL')}
    write(REC/'FROZEN_PROTOCOL.json',dict(schema='CURVEONLY_COMPACT_V1',seed=20261106,scope=summary,source_inventory=bind(REC/'SOURCE_INVENTORY.json'),inputs=bind(REC/'INPUTS.json'),truth_index=bind(REC/'TRUTH_INDEX.json'),
        feature_names=core.NAMES,model=dict(kind='HistGradientBoostingClassifier',max_iter=80,max_leaf_nodes=15,max_depth=4,min_samples_leaf=20,l2_regularization=1.,learning_rate=.1),
        threshold=.5,station_spacing_mm=2.,sample_curve_chord_error_mm=.05,curve_whole_tolerance_mm=3.,endpoint_tolerance_mm=3.,
        train_only_supervision=True,validation_in_fit_or_threshold_calibration=False,old_weights_reused=False,blind_validation=False,
        segmentation_policy='ARC<=30degrees; real gaps and non-arc shared process cuts remain checked. Raw exact-child counts are also reported.',
        resources=dict(maximum_workers=3,CPU_fraction=.8,memory_fraction=.9),sources=[bind(__file__),bind(core.__file__)],
        nominal_geometry_and_observation_separate=True,full_contract_pass=False))
    print('CURVEONLY_SPLIT',json.dumps(summary),flush=True)

def check_source():
    p=read(REC/'FROZEN_PROTOCOL.json')
    for b in p['sources']:
        assert sha(b['path'])==b['sha256'],'Numerical source changed during frozen experiment'
    return p

def fence(event,args):
    if event!='open'or not isinstance(args[0],(str,bytes,Path)):return
    path=Path(os.fsdecode(args[0]));s=str(path)
    if path.name in ('SOURCE_INVENTORY.json','TRUTH_INDEX.json','CURVE_TRUTH.json')or path.suffix.lower()in ('.xlsx','.cbf')or 'welds.json'in path.name:
        raise PermissionError('Curve-only inference label fence: '+s)

def input_one(c):
    check_source();sysmodule=__import__('sys');sysmodule.addaudithook(fence);dest=OUT/'features'/c['source_id'];dest.mkdir(parents=True,exist_ok=True)
    x,packet=core.input_curves(c);np.save(dest/'features.npy',x);write(dest/'geometry.json',packet)
    receipt=dict(source_id=c['source_id'],split=c['split'],rows=len(x),features=bind(dest/'features.npy'),geometry=bind(dest/'geometry.json'),timing=packet['timing'],pose=packet['pose'],raw=c['raw'],STEP=c['STEP'],labels_read=False,historical_weights_or_predictions_read=False)
    write(dest/'RECEIPT.json',receipt);print('CURVEONLY_INPUT',c['source_id'],len(packet['candidates']),len(x),round(packet['timing']['input_to_features'],2),flush=True);return receipt

def inputs(workers=3,only=None,resume_short=False):
    check_source();cases=read(REC/'INPUTS.json')['cases']
    if only:cases=[c for c in cases if c['source_id']==only]
    records=[];reused=[]
    if resume_short:
        previous=read(REC/'ALL_INPUTS.json');assert previous['failures']>0
        for c in list(cases):
            dest=OUT/'features'/c['source_id'];path=dest/'RECEIPT.json'
            if not path.exists():continue
            r=read(path);assert r['raw']==c['raw']and r['STEP']==c['STEP']
            assert sha(r['features']['path'])==r['features']['sha256']and sha(r['geometry']['path'])==r['geometry']['sha256']
            assert all(len(v['stations'])>=3 for v in read(r['geometry']['path'])['candidates'])
            reused.append(dict(source_id=c['source_id'],receipt=bind(path),proof='Completed old path has >=3 stations for every curve. New len<3 guard and two-station refinement branches are unreachable; every original numerical operation remains unchanged.'))
            records.append(r);cases.remove(c)
        write(REC/'SHORT_CURVE_REPAIR_CACHE_PROOF.json',dict(cases=reused,original_failure_run=previous,cached_precision_not_cold=True,no_training_or_prediction_yet=True))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))as pool:
        fs={pool.submit(input_one,c):c for c in cases}
        for future in concurrent.futures.as_completed(fs):
            c=fs[future]
            try:records.append(future.result())
            except Exception as exc:
                records.append(dict(source_id=c['source_id'],split=c['split'],failed=True,error=repr(exc),trace=traceback.format_exc()));print('CURVEONLY_INPUT_FAILED',c['source_id'],repr(exc),flush=True)
            write(REC/'INPUT_PROGRESS.json',dict(cases=records,complete=len(records),total=len(cases)))
    write(REC/('INPUT_CONTROL.json'if only else 'ALL_INPUTS.json'),dict(cases=records,full_scope=not bool(only),failures=sum(bool(r.get('failed'))for r in records)))

def labels_for(packet,truth):
    points=[];labels=[]
    for r in truth['runs']:
        p=core.resample(r['points'],.5);points.extend(p);labels.extend([1 if r['weld_type']=='horizontal'else 2]*len(p))
    if not points:return np.zeros(packet['offsets'][-1],np.int8)
    tree=cKDTree(points);types=np.array(labels);rows=[]
    for c in packet['candidates']:
        p=np.array(c['stations']);d,j=tree.query(p);y=np.where(d<=1.5,types[j],0);rows.append(y.astype(np.int8))
    return np.concatenate(rows)if rows else np.empty(0,np.int8)

def fit(round_name):
    protocol=check_source();start=time.perf_counter();cases=read(REC/'INPUTS.json')['cases'];truth={c['source_id']:c for c in read(REC/'TRUTH_INDEX.json')['cases']};xx=[];yy=[];ww=[];details=[]
    for c in cases:
        if c['split']!='TRAIN':continue
        dest=OUT/'features'/c['source_id'];receipt=dest/'RECEIPT.json'
        if not receipt.exists():details.append(dict(source_id=c['source_id'],failed_input=True));continue
        packet=read(dest/'geometry.json');x=np.load(dest/'features.npy');t=read(truth[c['source_id']]['truth']['path']);y=labels_for(packet,t)
        weight=np.zeros(len(y))
        for label in np.unique(y):weight[y==label]=1/max(1,(y==label).sum())
        xx.append(x);yy.append(y);ww.append(weight);details.append(dict(source_id=c['source_id'],rows=len(y),classes=dict(Counter(map(int,y)))))
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);settings=protocol['model'].copy();settings.pop('kind')
    model=HistGradientBoostingClassifier(**settings,random_state=20261106,early_stopping=False)
    with threadpool_limits(limits=2):model.fit(x,y,sample_weight=w*len(w)/w.sum())
    dest=REC/round_name;dest.mkdir(exist_ok=True)
    with (dest/'model.pkl').open('wb')as f:pickle.dump(model,f,protocol=5)
    write(dest/'FIT.json',dict(model=bind(dest/'model.pkl'),TRAIN_inputs=sum(c['split']=='TRAIN'for c in cases),cases=details,classes=dict(Counter(map(int,y))),rows=len(y),features=len(core.NAMES),seconds=time.perf_counter()-start,validation_labels_read=False,old_weights_reused=False,protocol=bind(REC/'FROZEN_PROTOCOL.json')))
    print('CURVEONLY_FIT',len(x),dict(Counter(map(int,y))),round(time.perf_counter()-start,2),flush=True)

def predict(round_name):
    check_source();sysmodule=__import__('sys');sysmodule.addaudithook(fence);dest=REC/round_name
    with (dest/'model.pkl').open('rb')as f:model=pickle.load(f)
    cases=read(REC/'INPUTS.json')['cases'];out=[]
    for c in cases:
        start=time.perf_counter();d=OUT/'features'/c['source_id'];target=OUT/round_name/'outputs'/c['source_id'];target.mkdir(parents=True,exist_ok=True)
        if not (d/'RECEIPT.json').exists():decoded=dict(parents=[],input_failure=True)
        else:
            x=np.load(d/'features.npy');packet=read(d/'geometry.json')
            with threadpool_limits(limits=2):decoded=core.decode(x,packet,model)
        write(target/'curves.json',decoded)
        with (target/'curves.csv').open('w',newline='',encoding='utf8')as f:
            fields=['workpiece_id','weld_id','parent_id','segment_index','weld_type','geometry_type','is_closed','start_x','start_y','start_z','end_x','end_y','end_z','length_mm','confidence','curve_points_file','status'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for p in decoded['parents']:
                for s in p['segments']:
                    a,b=s['points'][0],s['points'][-1];writer.writerow(dict(workpiece_id=c['source_id'],weld_id=p['parent_id']+'_'+str(s['segment_index']),parent_id=p['parent_id'],segment_index=s['segment_index'],weld_type=p['weld_type'],geometry_type=s['geometry_type'],is_closed=p['is_closed'],start_x=a[0],start_y=a[1],start_z=a[2],end_x=b[0],end_y=b[1],end_z=b[2],length_mm=s['length_mm'],confidence=p['confidence'],curve_points_file='curves.json',status=s['status']))
        out.append(dict(source_id=c['source_id'],split=c['split'],prediction=bind(target/'curves.json'),csv=bind(target/'curves.csv'),cached_predict_export_seconds=time.perf_counter()-start,input_failure=decoded.get('input_failure',False)))
    write(dest/'OUTPUTS_FROZEN.json',dict(cases=out,complete_scope=len(cases),all_outputs_closed=True,validation_scoring_started=False,model=bind(dest/'model.pkl'),cached_execution_not_cold_timing=True))
    print('CURVEONLY_OUTPUTS_FROZEN',len(out),flush=True)

def continuous_distance(query,polyline):
    q=np.asarray(query,float);p=core.resample(polyline,1.);a=p[:-1];v=np.diff(p,axis=0);den=np.maximum((v*v).sum(1),1e-18);maximum=0.
    for start in range(0,len(q),64):
        delta=q[start:start+64,None,:]-a[None,:,:];t=np.clip((delta*v).sum(2)/den,0,1)
        d=np.sqrt(((delta-t[:,:,None]*v)**2).sum(2)).min(1);maximum=max(maximum,float(d.max()))
        if maximum>3.000001:return maximum
    return maximum

def geometry_error(p,t):
    if p['weld_type']!=t['weld_type']:return None
    a,b=np.array(p['points']),np.array(t['points']);pc=np.linalg.norm(a[0]-a[-1])<.1;tc=np.linalg.norm(b[0]-b[-1])<.1
    if pc!=tc:return None
    if not pc:
        end=min(np.linalg.norm(a[[0,-1]]-b[[0,-1]],axis=1).max(),np.linalg.norm(a[[0,-1]]-b[[-1,0]],axis=1).max())
        if end>3:return None
    else:end=0.
    if np.any(a.max(0)<b.min(0)-3)or np.any(b.max(0)<a.min(0)-3):return None
    error=max(continuous_distance(core.resample(a,1.),b),continuous_distance(core.resample(b,1.),a))
    if error>3:return None
    return dict(endpoint_mm=float(end),whole_curve_mm=error)

def match(pred,truth):
    cost=np.full((len(pred),len(truth)),1e6);edges={}
    for i,p in enumerate(pred):
        for j,t in enumerate(truth):
            error=geometry_error(p,t)
            if error is not None:edges[i,j]=error;cost[i,j]=sum(error.values())
    matched=[]
    if cost.size:
        ii,jj=linear_sum_assignment(cost)
        matched=[dict(prediction=int(i),truth=int(j),**edges[i,j])for i,j in zip(ii,jj)if (i,j)in edges]
    tp=len(matched);return dict(G=len(truth),P=len(pred),TP=tp,FP=len(pred)-tp,FN=len(truth)-tp,matches=matched,extraction=tp/len(truth)if truth else None)

def score(round_name):
    check_source();dest=REC/round_name;frozen=read(dest/'OUTPUTS_FROZEN.json');truth={c['source_id']:c for c in read(REC/'TRUTH_INDEX.json')['cases']};cases=[]
    for c in frozen['cases']:
        assert sha(c['prediction']['path'])==c['prediction']['sha256'];decoded=read(c['prediction']['path']);t=read(truth[c['source_id']]['truth']['path']);pred=decoded['parents'];whole=match(pred,t['runs'])
        predchildren=[dict(points=s['points'],weld_type=p['weld_type'])for p in pred for s in p['segments']];seg=match(predchildren,t['children']);valid=0;fail=[]
        for m in whole['matches']:
            p=pred[m['prediction']];target=t['runs'][m['truth']];legal=True
            if target['circle']:
                legal=all((not(sfit:=core.circle_fit(s['points'])))or abs(sfit['sweep'])<=math.pi/6+1e-6 for s in p['segments'])
            else:
                expected=[np.array(t['children'][j]['points'])[-1]for j in target['children'][:-1]]
                actual=[np.array(s['points'])[-1]for s in p['segments'][:-1]]
                legal=all(any(np.linalg.norm(a-b)<=3 for b in actual)for a in expected)
            if legal:valid+=1
            else:fail.append(m)
        process=dict(G=whole['G'],P=whole['P'],TP=valid,FP=whole['P']-valid,FN=whole['G']-valid,extraction=valid/whole['G']if whole['G']else None)
        packetpath=OUT/'features'/c['source_id']/'geometry.json';packet=read(packetpath)if packetpath.exists()else dict(candidates=[])
        carrier=[dict(points=p['points'],weld_type=wt)for p in packet['candidates']for wt in ('horizontal','vertical')]
        coverage=sum(any(geometry_error(p,tg)is not None for p in carrier)for tg in t['runs'])
        row=dict(source_id=c['source_id'],split=c['split'],whole_curve=whole,process=process,raw_exact_child_diagnostic=seg,full_edge_candidate_coverage=coverage,candidate_count=len(packet['candidates']),input_failure=c['input_failure'],unmatched_truth=[dict(index=i,id=r['id'],children=r['children'],first_failure='NO_COMPLETE_NATIVE_CARRIER_OR_INTERVAL_OR_SELECTION_UNKNOWN')for i,r in enumerate(t['runs'])if i not in {m['truth']for m in whole['matches']}])
        cases.append(row);write(dest/'scores'/c['source_id']/'SCORE.json',row);print('CURVEONLY_SCORE',c['source_id'],c['split'],whole['TP'],whole['G'],'FP',whole['FP'],'carrier',coverage,flush=True)
    totals={}
    for split in ('TRAIN','VAL'):
        selected=[c for c in cases if c['split']==split];totals[split]={}
        for metric in ('whole_curve','process','raw_exact_child_diagnostic'):
            s={k:sum(c[metric][k]for c in selected)for k in ('G','P','TP','FP','FN')};s['extraction']=s['TP']/s['G']if s['G']else None;s['FP_per_G']=s['FP']/s['G']if s['G']else None;totals[split][metric]=s
    write(dest/'RESULT.json',dict(cases=cases,totals=totals,outputs_frozen=bind(dest/'OUTPUTS_FROZEN.json'),score_policy='Frozen curve-only geometry and arc30/shared-process evaluation. Raw exact child diagnostic separately. No E attributes acceptance claim.',full_contract_pass=False))
    print('CURVEONLY_TOTALS',json.dumps(totals),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','inputs','fit','predict','score']);ap.add_argument('--round',default='round01');ap.add_argument('--workers',type=int,default=3);ap.add_argument('--only');ap.add_argument('--resume-short-curve-repair',action='store_true');a=ap.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='inputs':inputs(a.workers,a.only,a.resume_short_curve_repair)
    elif a.action=='fit':fit(a.round)
    elif a.action=='predict':predict(a.round)
    else:score(a.round)

if __name__=='__main__':main()
