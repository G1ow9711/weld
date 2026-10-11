"""Same existing pair through frozen interval and validity heads. Diagnostic only."""
import argparse,collections,gc,hashlib,pickle,time
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_joint_pair_model_1248 import PairModel,load_inputs,tensors,CHUNK
from weld_pipeline.sjg_conditional_diagnostics_1284 import DEST,TRAIN

def feature_matrix(s,r,ii,jj,cats,margin):
    a=r['q_offset'];n=r['n'];q=s['m']['q_mm'][a:a+n];span=q[jj]-q[ii]
    mean=(s['prefix'][a+jj]-s['prefix'][a+ii])/span
    extra=np.column_stack((np.log1p(span/10),span/r['curve_length_mm'],mean,margin,
                           (cats%2==1).astype(float),(cats%2==0).astype(float)))
    return np.concatenate((s['x'][2*a+ii],s['x'][2*a+n+jj],
                           np.broadcast_to(s['context'][r['row']],(len(ii),160)),extra),axis=1).astype(np.float32)

def dev_records(s):
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    from weld_pipeline.sjg_recall_set_worker_1162 import target_for
    from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
    from weld_pipeline.world_set_evaluation_285 import reference_segment
    case=load_case('64');graph=case['graph'];case['x']._mmap.close()
    target,binding=target_for('64');native=native_targets(target)
    assert len(native['parents'])==26 and all(len(p['children'])==1 and p['children'][0]['original_kind']=='line' for p in native['parents'])
    ends=[np.asarray(reference_segment(p['children'][0])['points'])[[0,-1]] for p in native['parents']]
    records=[]
    for ri,row in enumerate(s['m']['rows']):
        row=int(row);a,b=map(int,s['m']['offsets'][ri:ri+2]);n=b-a;curve=graph.curve(row//2)
        if curve.kind!='line' or n<2:continue
        q=s['m']['q_mm'][a:b];points=curve.at(q);groups=[]
        for owner,ep in enumerate(ends):
            left=np.flatnonzero(np.linalg.norm(points-ep[0],axis=1)<=3)
            right=np.flatnonzero(np.linalg.norm(points-ep[1],axis=1)<=3)
            pairs=sorted({min(int(i),int(j))*n+max(int(i),int(j)) for i in left for j in right if i!=j})
            if pairs:
                vertical=native['parents'][owner]['children'][0]['weld_type']=='vertical'
                groups.append(dict(owner=[owner,0],allowed_categories=[2,4] if vertical else [1,3],pairs=pairs,
                    category_parent_parities=None))
        if groups:records.append(dict(source='64',row=row,q_offset=a,feature_offset=2*a,n=n,
            curve_length_mm=curve.length,is_closed=False,gold_groups=groups))
    return records,binding['target_sha256']

def run(key):
    start=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.enabled=False;check_resources();dest=DEST/f'same_pair_{key}';dest.mkdir(exist_ok=False)
    p=read(DEST/'registered_protocol.json')
    for path,h in p['frozen_models'].items():assert digest(OUT/path)==h
    routes=read(OUT/'run433/inputs_frozen_before_train_labels.json')['routes']
    s=load_inputs(key,**routes[key]['pair_inputs'])
    payload=torch.load(OUT/'run432/pair_model.pt',map_location='cpu')
    device='cuda' if torch.cuda.is_available() else 'cpu';model=PairModel(payload['classes']).to(device).eval()
    model.load_state_dict(payload['state_dict'])
    forests=[]
    for arm in ('run445','run446'):
        with (OUT/arm/'models.pkl').open('rb') as f:forests.append(pickle.load(f)['models']['validity'])
    if key in TRAIN:
        targets=read(OUT/f'run431/source_{key}_set_targets.json')
        assert digest(OUT/f'run431/source_{key}_set_targets.json')==next(c['target_sha256'] for c in read(OUT/'run431/receipt.json')['cases'] if c['source']==key)
        records=[r for r in targets['records'] if r['gold_groups']];target_sha=targets['original_target_sha256']
        known={r['row']:r['complete_background_supervision_allowed'] for r in read(OUT/f'run422/background_{key}.json')['rows']}
    else:records,target_sha=dev_records(s);known={}
    raw=read(OUT/f'run432/proposal_{key}/selected_intervals.json')['intervals']
    raw_by_row=collections.defaultdict(list)
    for i,v in enumerate(raw):raw_by_row[v['row']].append((i,v))
    original_features=np.load(OUT/f'run434/source_{key}/features.npy',mmap_mode='r')
    expected=[]
    for arm in ('run445','run446'):
        ds=read(OUT/arm/f'test_{key}/decisions.json')['decisions']
        expected.append({d['candidate_index']:d['validity_probability'] for d in ds})
    # Full factory parity on every actual raw candidate residing on a positive row,
    # including the existing seven correct DEV copies. Diagnostic scores not clamped.
    parity=dict(rows=0,max_feature_error=0.,max_probability_error=0.,max_margin_error=0.)
    total=sum(len(g['pairs'])*len(g['allowed_categories']) for r in records for g in r['gold_groups'])
    dtype=np.dtype([('row','<i4'),('parent','<i4'),('child','<i4'),('i','<i4'),('j','<i4'),('category','i1'),
                    ('margin','<f4'),('C','<f8'),('D','<f8'),('negative_strictly_higher','<i8'),
                    ('negative_equal','<i8'),('actual_raw','?')])
    table=np.lib.format.open_memmap(dest/'existing_pair_scores.npy',mode='w+',dtype=dtype,shape=(total,))
    feature_sha=hashlib.sha256();offset=0;summaries={};parities=[]
    print('SAME_PAIR_START',key,'ROWS',len(records),'DIRECTION_EXPLANATIONS',total,flush=True)
    for ri,r in enumerate(records):
        n=r['n'];a=r['q_offset'];q=s['m']['q_mm'][a:a+n]
        x,c,qq,prefix=tensors(s,r,payload['norm'],device)
        with torch.no_grad():
            h=model.encode(x,c);deltas=np.full((n,n,len(payload['classes'])-1),-np.inf,np.float32)
            for lo in range(0,n-1,CHUNK):
                hi=min(lo+CHUNK,n-1);scores=model.scores(h[:n],h[n:],qq,prefix,r['curve_length_mm'],r['is_closed'],lo,hi)
                deltas[lo:hi]=(scores[:,:,1:]-scores[:,:,:1]).cpu().numpy()
        actual={}
        for raw_index,v in raw_by_row[r['row']]:
            i=int(np.argmin(abs(q-v['lo_mm'])));j=int(np.argmin(abs(q-v['hi_mm'])));cat=v['category']
            assert q[i]==v['lo_mm'] and q[j]==v['hi_mm'];actual[(i,j,cat)]=raw_index
            margin=float(deltas[i,j,cat-1]);err=abs(margin-v['margin']);assert err<1e-4
            xx=feature_matrix(s,r,np.array([i]),np.array([j]),np.array([cat]),np.array([margin]))
            fe=float(abs(xx[0]-original_features[raw_index]).max());assert fe<1e-4,(key,raw_index,fe)
            pe=max(abs(forest.predict_proba(xx)[0,1]-expected[k][raw_index]) for k,forest in enumerate(forests));assert pe<1e-10
            parity['rows']+=1;parity['max_feature_error']=max(parity['max_feature_error'],fe)
            parity['max_probability_error']=max(parity['max_probability_error'],pe);parity['max_margin_error']=max(parity['max_margin_error'],err)
        # Only rows with authorized complete background supervision have exhaustive
        # negative rankings. UNKNOWN is never converted to negative. DEV uses only
        # already-confirmed incompatible raw candidates, explicitly a limited domain.
        if known.get(r['row'],False):
            mask=np.triu(np.ones((n,n),bool),1)
            for g in r['gold_groups']:
                ids=np.asarray(g['pairs']);mask[ids//n,ids%n]=False
            negative=np.sort(deltas.max(-1)[mask]);rank_scope='complete_background_known_noncompatible_intervals'
        elif key=='64':
            diag=read(OUT/'run435/test_64/postfreeze_stage_diagnosis.json')
            bad={d['candidate_index'] for d in diag['candidates'] if not d['compatible_original_parent_indices']}
            negative=np.sort(np.asarray([deltas[i,j,cat-1] for (i,j,cat),index in actual.items() if index in bad],np.float32))
            rank_scope='confirmed_incompatible_actual_raw_only_NOT_exhaustive_unknown_pairs'
        else:negative=None;rank_scope='UNKNOWN_background_not_ranked'
        for g in r['gold_groups']:
            owner=tuple(g['owner']);summary=summaries.setdefault(owner,dict(parent=owner[0],child=owner[1],
                explanation_count=0,M=None,max_C=0.,max_D=0.,J_C=False,J_D=False,J_D_ge_half=False,
                positive_interval_count=0,actual_raw_correct_copies=0,best_interval_witness=None,
                complete_parent_path_NOT_certified=True))
            ids=np.asarray(g['pairs'],np.int64)
            parities.append(dict(row=r['row'],owner=list(owner),allowed_categories=g['allowed_categories'],
                                 category_parent_parities=g.get('category_parent_parities'),rank_scope=rank_scope,
                                 confirmed_negative_intervals=None if negative is None else len(negative)))
            for cat in g['allowed_categories']:
                for begin in range(0,len(ids),8192):
                    pair=ids[begin:begin+8192];ii,jj=pair//n,pair%n;cats=np.full(len(pair),cat,np.int8)
                    margins=deltas[ii,jj,cat-1];xx=feature_matrix(s,r,ii,jj,cats,margins)
                    feature_sha.update(xx.tobytes(order='C'));pc,pd=[f.predict_proba(xx)[:,1] for f in forests]
                    z=table[offset:offset+len(pair)];z['row']=r['row'];z['parent']=owner[0];z['child']=owner[1]
                    z['i']=ii;z['j']=jj;z['category']=cats;z['margin']=margins;z['C']=pc;z['D']=pd
                    z['actual_raw']=[(int(i),int(j),cat) in actual for i,j in zip(ii,jj)]
                    if negative is None:z['negative_strictly_higher']=-1;z['negative_equal']=-1
                    else:
                        left=np.searchsorted(negative,margins,side='left');right=np.searchsorted(negative,margins,side='right')
                        z['negative_strictly_higher']=len(negative)-right;z['negative_equal']=right-left
                    summary['explanation_count']+=len(pair);summary['positive_interval_count']+=int((margins>0).sum())
                    summary['actual_raw_correct_copies']+=int(z['actual_raw'].sum())
                    summary['J_C']|=bool(((margins>0)&(pc>.5)).any());summary['J_D']|=bool(((margins>0)&(pd>.5)).any())
                    summary['J_D_ge_half']|=bool(((margins>0)&(pd>=.5)).any())
                    summary['max_C']=max(summary['max_C'],float(pc.max()));summary['max_D']=max(summary['max_D'],float(pd.max()))
                    best=int(margins.argmax())
                    if summary['M'] is None or margins[best]>summary['M']:
                        summary['M']=float(margins[best]);summary['best_interval_witness']=dict(row=r['row'],i=int(ii[best]),j=int(jj[best]),
                            category=cat,C=float(pc[best]),D=float(pd[best]),rank_scope=rank_scope,
                            negatives_strictly_higher=int(z['negative_strictly_higher'][best]),negatives_equal=int(z['negative_equal'][best]))
                    offset+=len(pair)
        del x,c,qq,prefix,h,deltas;gc.collect()
        if ri%25==0:
            table.flush();print('SAME_PAIR_PROGRESS',key,ri+1,'/',len(records),'EXPLANATIONS',offset,'/',total,flush=True)
            write(dest/'progress.json',dict(completed_rows=ri+1,total_rows=len(records),explanations=offset,total=total,elapsed=time.perf_counter()-start))
            reclaim_own_idle_pages();check_resources()
    assert offset==total;table.flush();table._mmap.close()
    expected_G=26 if key=='64' else targets['original_G'];assert len(summaries)==expected_G
    if key=='64':assert sum(v['actual_raw_correct_copies'] for v in summaries.values())==7
    write(dest/'factory_parity.json',dict(status='PASS',**parity))
    write(dest/'direction_and_negative_scope.json',dict(rows=parities))
    write(dest/'summary.json',dict(source=key,G=expected_G,children=list(summaries.values()),
        explanations=total,feature_columns=526,feature_storage='exact reconstruction from immutable source arrays and recorded indices/category/margin',
        ordered_feature_stream_sha256=feature_sha.hexdigest(),pair_table_sha256=digest(dest/'existing_pair_scores.npy'),
        target_sha256=target_sha,seconds=time.perf_counter()-start,fit_calls=0,
        NOT_new_extraction_TP=True,whole_parent_path_NOT_certified=True,full_contract_pass=False))
    print('SAME_PAIR_COMPLETE',key,'G',expected_G,'J_D',sum(v['J_D'] for v in summaries.values()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args();run(a.source)
