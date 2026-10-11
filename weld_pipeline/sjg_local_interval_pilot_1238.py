"""Learn local weld occupancy, then decode every positive interval.

Only fixed TRAIN labels are read by training. Presence admissions remain384;
the prior interval-count/ratio heads are replaced by local occupancy runs.
This remains a proposal frontend, not assembled final parent output.
"""
import argparse,gc,hashlib,json,pickle,time,traceback
from collections import defaultdict
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_dense_joint_sequence_1237 import ordered_features,SCHEMA as INPUT_SCHEMA
from weld_pipeline.sjg_whole_joint_targets_1229 import TRAIN
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_finite_face_domains_1235 import decode_shared
from weld_pipeline.sjg_joint_graph_capacity_1105 import ContinuousGraphBank
from weld_pipeline.sjg_joint_forest_train_1109 import attributes
INPUT_RUN='run386'
EXPECTED_SCHEMA=INPUT_SCHEMA


def load_inputs(key,input_run=None):
    folder=OUT/f'{input_run or INPUT_RUN}/source_{key}';r=read(folder/'receipt.json')
    assert r['schema']==EXPECTED_SCHEMA and digest(folder/'features.npy')==r['features_sha256']
    assert digest(folder/'sequence_map.npz')==r['map_sha256']
    with np.load(folder/'sequence_map.npz') as z:m={k:z[k] for k in z.files}
    return np.load(folder/'features.npy',mmap_mode='r'),m,r


def intervals(q,positive,closed=False):
    edges=np.r_[q[0],(q[:-1]+q[1:])/2,q[-1]]
    starts=np.flatnonzero(positive & ~np.r_[False,positive[:-1]])
    ends=np.flatnonzero(positive & ~np.r_[positive[1:],False])+1
    result=[dict(lo=float(edges[a]),hi=float(edges[b]),closed=False,wrap=False) for a,b in zip(starts,ends) if edges[b]>edges[a]]
    if closed and positive.all():return [dict(lo=float(q[0]),hi=float(q[-1]),closed=True,wrap=False)]
    if closed and len(result)>1 and positive[0] and positive[-1]:
        wrap=dict(lo=result[0]['hi'],hi=result[-1]['lo'],closed=False,wrap=True)
        result=result[1:-1]+[wrap]
    return result


def controls(dest):
    q=np.arange(0.,21.,5.);p=np.array([False,True,True,False,True])
    original=intervals(q,p);reverse=intervals(q,p[::-1])
    mapped=sorted([(q[-1]-r['hi'],q[-1]-r['lo']) for r in reverse])
    assert mapped==[(r['lo'],r['hi']) for r in original]
    assert intervals(q,np.ones(5,bool),True)[0]['closed']
    r=intervals(q,np.array([True,True,False,True,True]),True)
    assert len(r)==1 and r[0]['wrap'] and (r[0]['lo'],r[0]['hi'])==(7.5,12.5)
    write(dest/'interval_controls.json',dict(line_reversal=True,closed_and_wrapped_circle=True,
        all_positive_runs_retained=True,no_minimum_length_filter=True,
        perfect_site_label_boundary_discretization_bound_mm=2.5,not_model_error_guarantee=True))


def train(dest):
    start=time.perf_counter();controls(dest);total=0;plans=[]
    for key in TRAIN:
        x,m,r=load_inputs(key);x._mmap.close();folder=OUT/f'run380/source_{key}'
        sr=read(folder/'joint_supervision_receipt.json')
        assert digest(folder/'joint_count_targets.npz')==sr['labels_sha256']
        assert digest(folder/'joint_interval_instances.json')==sr['instances_sha256']
        with np.load(folder/'joint_count_targets.npz') as z:labels=z['labels']
        lookup={int(p):i for i,p in enumerate(m['profile_ids'])}
        rows=[int(row) for row in m['admitted_rows'] if labels[row]>=0]
        count=sum(int(m['offsets'][lookup[row//2]+1]-m['offsets'][lookup[row//2]]) for row in rows)
        plans.append((key,m,rows,count,r));total+=count
    columns=plans[0][4]['columns'];assert all(p[4]['columns']==columns for p in plans)
    x=np.empty((total,columns),np.float32);y=np.zeros(total,np.uint8);w=np.zeros(total,float);offset=0;audits=[]
    for key,m,rows,count,input_receipt in plans:
        source,_,_=load_inputs(key);raw=read(OUT/f'run380/source_{key}/joint_interval_instances.json')['instances']
        events=defaultdict(list);original=set();covered=set()
        for event in raw:
            original.add(tuple(event['owner']))
            if event['count_is_known']:events[event['row']].append(event)
        lookup={int(p):i for i,p in enumerate(m['profile_ids'])};begin=offset
        for row in rows:
            j=lookup[row//2];a,b=m['offsets'][j:j+2];q=m['q_mm'][a:b];n=b-a
            x[offset:offset+n]=ordered_features(source[a:b],row%2)
            for event in events[row]:
                lo,hi=np.array(event['ratio'])*event['length_mm'];mask=(q>=lo)&(q<=hi)
                if mask.any():
                    y[offset:offset+n][mask]=1;w[offset:offset+n][mask]+=event['weight']/int(mask.sum());covered.add(tuple(event['owner']))
            offset+=n
        yy=y[begin:offset];ww=w[begin:offset];pos=yy>0;neg=~pos
        if pos.any():ww[pos]/=(len(TRAIN)*2*ww[pos].sum())
        if neg.any():ww[neg]=1/(len(TRAIN)*2*int(neg.sum()))
        audit=dict(source=key,known_admitted_rows=len(rows),point_rows=int(offset-begin),positive_sites=int(pos.sum()),
            original_children=len(original),covered_children=len(covered),missing_owners=[list(v) for v in sorted(original-covered)],
            feature_sha256=input_receipt['features_sha256'],unknown_admitted_rows_excluded_from_supervision=True)
        audits.append(audit);source._mmap.close();print('INTERVAL_SUPERVISION',audit,flush=True)
    assert offset==total
    write(dest/'supervision_receipt.json',dict(cases=audits,TRAIN=list(TRAIN),G=sum(a['original_children'] for a in audits),
        point_rows=total,columns=columns,fit_calls=0,DEV_reads=0,STEP_reads=0))
    if any(a['missing_owners'] for a in audits):raise ValueError('Missing original TRAIN child supervision; no fit')
    seen={};conflicts=[];duplicate_rows=0
    for i in range(total):
        sig=hashlib.sha256(x[i].tobytes()).digest();previous=seen.get(sig)
        if previous is None:seen[sig]=(i,int(y[i]))
        else:
            duplicate_rows+=1
            if previous[1]!=y[i]:conflicts.append(dict(first=previous[0],second=i,labels=[previous[1],int(y[i])]))
    del seen;gc.collect()
    write(dest/'input_control.json',dict(point_rows=total,positive_sites=int(y.sum()),exact_feature_duplicate_rows=duplicate_rows,
        exact_opposite_label_conflicts=conflicts,fit_calls=0,DEV_reads=0))
    if conflicts:raise ValueError('Exact local features have opposite occupancy labels; no fit')
    check_resources();model=ExtraTreesClassifier(n_estimators=16,max_depth=18,min_samples_leaf=4,max_features=.75,
        bootstrap=False,n_jobs=2,random_state=1031,warm_start=True);trace=[]
    for trees in range(16,65,16):
        check_resources();model.set_params(n_estimators=trees);model.fit(x,y,sample_weight=w)
        trace.append(dict(trees=trees,seconds=time.perf_counter()-start));write(dest/'training_progress.json',dict(trace=trace))
        print('INTERVAL_TRAIN TREES',trees,'/64 SECONDS',round(time.perf_counter()-start,2),flush=True)
    model.set_params(n_jobs=1,warm_start=False)
    old=read(OUT/'run384/training_receipt.json');assert digest(OUT/'run384/whole_joint_model.pkl')==old['model_sha256']
    with (OUT/'run384/whole_joint_model.pkl').open('rb') as f:prior=pickle.load(f)
    payload=dict(occupancy=model,category=prior['models']['category'],K=prior['K'],threshold=.5)
    atomic_binary_write(dest/'local_interval_model.pkl',lambda f:pickle.dump(payload,f,protocol=5))
    with (dest/'local_interval_model.pkl').open('rb') as f:reloaded=pickle.load(f)
    assert np.array_equal(model.predict_proba(x[:256]),reloaded['occupancy'].predict_proba(x[:256]))
    write(dest/'training_receipt.json',dict(status='LOCAL_OCCUPANCY_MODEL_TRAINED',TRAIN=list(TRAIN),G=213,
        point_rows=total,columns=columns,input_run=INPUT_RUN,reload_exact=True,trees=64,trace=trace,threshold=.5,category_sha256=forest_state_sha256(payload['category']),
        model_sha256=digest(dest/'local_interval_model.pkl'),DEV_reads=0,STEP_reads=0,full_contract_pass=False,seconds=time.perf_counter()-start))
    print('INTERVAL_TRAIN_COMPLETE',round(time.perf_counter()-start,2),flush=True)


def infer(dest,key):
    start=time.perf_counter();receipt=read(dest/'training_receipt.json')
    assert receipt['input_run']==INPUT_RUN
    assert digest(dest/'local_interval_model.pkl')==receipt['model_sha256']
    with (dest/'local_interval_model.pkl').open('rb') as f:payload=pickle.load(f)
    x,m,r=load_inputs(key);case=load_case(key);case['x']._mmap.close();g=case['graph']
    old=np.load(OUT/f'run380/source_{key}/joint_features.npy',mmap_mode='r')
    lookup={int(p):i for i,p in enumerate(m['profile_ids'])};choices=[];probability_blocks=[];probability_offsets=[0]
    counts=[]
    for row in m['admitted_rows']:
        row=int(row);j=lookup[row//2];a,b=m['offsets'][j:j+2];q=m['q_mm'][a:b];curve=g.curve(row//2)
        probability=payload['occupancy'].predict_proba(ordered_features(x[a:b],row%2))[:,1]
        probability_blocks.append(probability.astype(np.float32));probability_offsets.append(probability_offsets[-1]+len(q))
        domains=intervals(q,probability>=.5,curve.kind=='arc' and curve.arc.is_closed);counts.append(len(domains))
        if not domains:continue
        category_x=np.column_stack((np.tile(old[row],(len(domains),1)),np.arange(1,len(domains)+1)/payload['K'])).astype(np.float32)
        categories=payload['category'].predict(category_x)
        for domain,category in zip(domains,categories):
            attr=attributes(category);attr.update(forward=True,closed=domain['closed'],wrap=domain['wrap'])
            choices.append(dict(query=len(choices),probability=1.,child_count=1,closed=domain['closed'],
                children=[dict(profile=row//2,intervals=[0,0],fractions=[domain['lo']/curve.length,domain['hi']/curve.length],
                    side='UNKNOWN',connection='UNKNOWN',**attr)]))
    result=decode_shared(choices,ContinuousGraphBank(g));result.update(scope='LOCAL_INTERVAL_RAW_PROPOSALS_NOT_FINAL_PARENT_OUTPUT',GT_reads=0,STEP_reads=0)
    folder=dest/f'proposal_{key}';folder.mkdir(exist_ok=False)
    with (folder/'decoded.json').open('w',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    np.savez(folder/'occupancy.npz',admitted_rows=m['admitted_rows'],offsets=np.array(probability_offsets),probabilities=np.concatenate(probability_blocks))
    write(folder/'prediction_receipt.json',dict(status='PURE_INPUT_PROPOSALS_FROZEN_BEFORE_EVALUATION',source=key,
        decoded_sha256=digest(folder/'decoded.json'),model_sha256=receipt['model_sha256'],admitted_rows=len(m['admitted_rows']),
        proposed_intervals=len(choices),maximum_intervals_per_carrier=max(counts,default=0),GT_reads=0,STEP_reads=0,
        seconds=time.perf_counter()-start,full_contract_pass=False))
    x._mmap.close();old._mmap.close();print('INTERVAL_PREDICT',key,'INTERVALS',len(choices),'SECONDS',round(time.perf_counter()-start,2),flush=True)


def main():
    global INPUT_RUN,EXPECTED_SCHEMA
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',choices=['train','infer'],required=True);ap.add_argument('--source');ap.add_argument('--input-run',choices=['run386','run388','run390'],default='run386');a=ap.parse_args()
    INPUT_RUN=a.input_run
    if INPUT_RUN=='run388':
        from weld_pipeline.sjg_global_neighborhood_sequence_1239 import SCHEMA
        EXPECTED_SCHEMA=SCHEMA
    elif INPUT_RUN=='run390':
        from weld_pipeline.sjg_finite_extent_sequence_1240 import SCHEMA
        EXPECTED_SCHEMA=SCHEMA
    dest=OUT/a.run
    if a.stage=='train':
        dest.mkdir(exist_ok=False);write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),TRAIN=list(TRAIN),DEV=['64'],
            feature_schema=EXPECTED_SCHEMA,input_run=INPUT_RUN,step_mm=5.,trees=64,threshold=.5,frozen_admission='run384',frozen_category='run384',
            implicit_interval_count_no_truncation=True,best379_preserved=True,external_consultation_paused=True))
    try:train(dest) if a.stage=='train' else infer(dest,a.source)
    except BaseException:write(dest/f'{a.stage}_{a.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
