"""One frozen-model full-domain search; native masks are training-only."""
import collections,gc,hashlib,json,time,os
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,digest,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_owned_continuation_watchdog_1295 import atomic_json
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_carrier_commit_1319 import publish,recover,durable_json
from weld_pipeline.sjg_local_runtime_904 import resources

def memory_point(folder,position,phase,arrays=None):
    entry=dict(pid=os.getpid(),parent_pid=os.getppid(),role='single_scientific_worker',
        position=position,phase=phase,monotonic=time.monotonic(),resources=resources(),
        arrays_nbytes=arrays or {})
    with (folder/'memory_events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(entry)+'\n')

def binding_for(key):
    p=read(DEST/'registered_protocol.json')
    return dict(source=key,protocol_sha256=digest(DEST/'registered_protocol.json'),
        miner_sha256=digest(__file__),model_sha256=digest(OUT/'run456/unified_model.npz'),
        native_sha256=digest(OUT/f'run422/source_{key}/receipt.json'),
        input_sha256=digest(OUT/f'run430/source_{key}/receipt.json'),
        zero_choices_sha256=digest(OUT/f'run432/proposal_{key}/selected_intervals.json'),
        float64_accumulation=True)

DEST=OUT/'run458';TRAIN=('21','BM79A_native_local','10','25','36')

def event(name,**v):
    entry=dict(event=name,monotonic=time.monotonic(),**v)
    with (DEST/'mining_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,allow_nan=False)+'\n')
    print('DOMAIN_MINING',name,v,flush=True)

def guard():
    if time.monotonic()>read(DEST/'mining_budget.json')['deadline_monotonic']-10:raise TimeoutError('Original full-domain mining deadline')

class StreamingDP:
    """Original skip-on-tie interval DP, processed as topological outgoing edges."""
    def __init__(self,n):
        self.n=n;self.best=np.zeros(n);self.prev=np.full(n,-1,int)
        self.cat=np.zeros(n,int);self.edge=np.zeros(n);self.next=0
    def carry(self,i):
        if i and self.best[i-1]>=self.best[i]:
            self.best[i]=self.best[i-1];self.prev[i]=i-1;self.cat[i]=0
    def feed(self,i,margin,category):
        assert i==self.next;self.next+=1;self.carry(i)
        # Match the original float64 best-vector + margin-vector addition.
        # NumPy may otherwise keep float32 when adding a float64 scalar.
        ids=np.arange(i+1,self.n);values=self.best[i]+margin[ids].astype(np.float64);take=values>self.best[ids]
        ids=ids[take];self.best[ids]=values[take];self.prev[ids]=i;self.cat[ids]=category[ids];self.edge[ids]=margin[ids]
    def finish(self):
        assert self.next==self.n-1;self.carry(self.n-1);chosen=[];j=self.n-1
        while j>0:
            i=int(self.prev[j]);assert 0<=i<j
            if self.cat[j]:chosen.append((i,j,int(self.cat[j]),float(self.edge[j])))
            j=i
        return chosen[::-1]

def controls():
    from weld_pipeline.sjg_joint_pair_model_1248 import select_intervals
    rng=np.random.default_rng(1315);cases=0
    for n in (2,3,9,25):
        for _ in range(12):
            for score in (rng.integers(-2,3,(n,n)).astype(float),rng.normal(size=(n,n)).astype(np.float32),rng.normal(size=(n,n))):
                score[np.tril_indices(n)]=-np.inf
                cat=rng.integers(1,5,(n,n));ref=select_intervals(score,cat);dp=StreamingDP(n)
                for i in range(n-1):dp.feed(i,score[i],cat[i])
                assert dp.finish()==ref;cases+=1
    atomic_json(DEST/'streaming_DP_control.json',dict(status='PASS',exact_tie_and_empty_parity_cases=cases,
       float32_and_float64_margins_use_original_float64_accumulation=True,
       original_open_closed_candidate_domain_unchanged=True,supervision_mask_not_used_in_deployment=True))

def native_mask(row,ii,jj,n,profiles,known,indices):
    wrong=np.full((len(ii),4),bool(known.get(row,False)),bool)
    if not wrong.any():return wrong
    encoded=ii*n+jj
    for group in profiles[row//2]['groups']:
        assert not group['implicit_all_pairs']
        allowed=indices[group['pair_start']:group['pair_stop']]
        if not len(allowed):continue
        pos=np.searchsorted(allowed,encoded);match=(pos<len(allowed))&(allowed[np.minimum(pos,len(allowed)-1)]==encoded)
        for c in (group['native_category'],group['reversed_parent_category']):wrong[match,c-1]=False
    return wrong

class ExistingPool:
    def __init__(self,key):
        folder=OUT/f'run454/workset_{key}';sr=read(folder/'structure_receipt.json');r=read(OUT/f'run450/workset_{key}/receipt.json')
        for f,h in {**sr['files'],'features.npy':r['features_sha256'],'base_margins.npy':r['base_sha256']}.items():assert digest(folder/f)==h
        self.pairs=np.load(folder/'pool_pairs.npy');self.lookup={tuple(map(int,p)):i for i,p in enumerate(self.pairs)}
        self.x=[np.load(folder/f,mmap_mode='r') for f in ('features.npy','extra_features.npy')]
        self.b=[np.load(folder/f,mmap_mode='r') for f in ('base_margins.npy','extra_base.npy')];self.cut=len(self.x[0])
    def get(self,i):
        k=int(i>=self.cut);j=i-self.cut if k else i
        return self.x[k][j],self.b[k][j]
    def close(self):
        for a in self.x+self.b:a._mmap.close()

def signature(x,b,c):
    return (c,hashlib.sha256(np.asarray(x,np.float32).tobytes()+np.float32(b[c]).tobytes()).hexdigest())

def positives():
    identities=read(OUT/'run454/identity_control.json')['original_children'];result=collections.defaultdict(list)
    for key in TRAIN:
        f=OUT/f'run450/workset_{key}';x=np.load(f/'features.npy',mmap_mode='r');b=np.load(f/'base_margins.npy',mmap_mode='r')
        for v in identities:
            if v['source']==key:
                i=v['pool_index'];c=v['native_identity'][-1]-1;result[signature(x[i],b[i],c)].append(v)
        x._mmap.close();b._mmap.close()
    assert sum(map(len,result.values()))==269
    return result

def mine(key,start,stop):
    assert key in TRAIN;begin=time.monotonic();guard()
    import torch
    from weld_pipeline import sjg_unified_interval_workset_1288 as old
    from weld_pipeline.sjg_joint_pair_model_1248 import tensors,CHUNK
    from weld_pipeline.sjg_resumable_unified_workset_1293 import sparse
    old.DEST=OUT/'run450';folder=DEST/f'mined_{key}';folder.mkdir(exist_ok=True);blocks=folder/'blocks';blocks.mkdir(exist_ok=True)
    commits=folder/'commits';commits.mkdir(exist_ok=True);binding=binding_for(key)
    memory_point(folder,start,'before_piece_load')
    cache={}
    if key=='10':
        audit=read(OUT/'run457/partial_feature_block_integrity.json')
        assert audit['status']=='COMMITTED_FEATURE_BLOCK_INTEGRITY_PASS'
        progress=read(OUT/'run457/mined_10/progress.json')
        assert digest(OUT/'run457/mined_10/progress.json')==audit['progress_sha256']
        cache={b['row']:b for b in progress['committed_blocks']}

    with np.load(OUT/'run456/unified_model.npz') as z:theta=z['theta']
    assert digest(OUT/'run456/unified_model.npz')==read(OUT/'run456/training_receipt.json')['model_sha256']
    pos=positives();pool=ExistingPool(key)
    native=OUT/f'run422/source_{key}';nr=read(native/'receipt.json')
    assert digest(native/'compatibility.json')==nr['compatibility_sha256'] and digest(native/'pairs.npz')==nr['pairs_sha256']
    relation=read(native/'compatibility.json');profiles={p['profile']:p for p in relation['profiles']}
    known={r['row']:r['complete_background_supervision_allowed'] for r in relation['rows']}
    with np.load(native/'pairs.npz') as z:native_indices=z['indices']
    rt=old.load_runtime(key)
    # Original runtime has already converted curve lengths/closed flags into records.
    # Scoring uses frozen endpoint/context/sequence arrays; no later function uses graph.
    del rt['graph'];gc.collect();reclaim_own_idle_pages();check_resources()
    memory_point(folder,start,'after_piece_load',dict(endpoint=int(np.prod(rt['s']['x'].shape))*4,
        sequence=int(rt['seq'].nbytes),context=int(rt['s']['context'].nbytes)))
    assert 0<=start<stop<=len(rt['records']) and stop-start<=256

    original=collections.defaultdict(list);prior=collections.defaultdict(list)
    for v in read(OUT/f'run432/proposal_{key}/selected_intervals.json')['intervals']:original[v['row']].append(v)
    if key in ('21','BM79A_native_local'):
        for v in read(OUT/f'run456/trained/proposal_{key}/selected_intervals.json')['intervals']:prior[v['row']].append(v)
    ledger=[];total=negative_carriers=negative_constraints=0;deployment=[];exposed=[];max_error=0.
    for position in range(start,stop):
        guard()
        if recover(commits,position,binding) is not None:continue
        r=rt['records'][position];n=r['n'];row=r['row'];a=r['q_offset'];qraw=rt['s']['m']['q_mm'][a:a+n]
        memory_point(folder,position,'before_carrier')
        ledger=[];deployment=[];exposed=[]
        if n<2:
            publish(commits,position,binding,dict(position=position,row=row,pairs=0,negative_constraints=0,
                negative_carrier=False,deployment=[],known_wrong_selection=[],zero_selection=[],
                zero_parity=True,H=[],blocks=[]))
            continue
        dense_control=key=='10' and position<2
        dense={}
        if dense_control:
            for name,dtype in (('normal',float),('bad',float),('zero',np.float32)):
                dense[name]=[np.full((n,n),-np.inf,dtype),np.zeros((n,n),int)]

        x,c,q,p=tensors(rt['s'],r,rt['payload']['norm'],rt['device']);model=rt['model']
        evidence=old.evidence(rt,r);project=evidence.project(theta[:,360:600].reshape(4,4,60))
        endpoints=x.cpu().numpy();left=endpoints[:n]@theta[:,:180].T;right=endpoints[n:]@theta[:,180:360].T
        normal=StreamingDP(n);zero=StreamingDP(n);bad=StreamingDP(n);row_negative=0
        with torch.no_grad():
            h=model.encode(x,c)
            for lo in range(0,n-1,CHUNK):
                guard();hi=min(lo+CHUNK,n-1)
                output=model.scores(h[:n],h[n:],q,p,r['curve_length_mm'],r['is_closed'],lo,hi)
                base=(output[:,:,1:]-output[:,:,:1]).cpu().numpy()
                normal_m=np.full((hi-lo,n),-np.inf);normal_c=np.zeros((hi-lo,n),int)
                bad_m=np.full_like(normal_m,-np.inf);bad_c=np.zeros_like(normal_c)
                ii,jj=np.where(np.arange(lo,hi)[:,None]<np.arange(n)[None,:]);ii+=lo
                for pair_first in range(0,len(ii),8192):
                    i,j=ii[pair_first:pair_first+8192],jj[pair_first:pair_first+8192]
                    score=base[i-lo,j].astype(float)+left[i]+right[j]+project.scores(qraw[i],qraw[j])+old.scalars(rt,r,i,j)@theta[:,600:604].T+theta[:,604]
                    assert np.isfinite(score).all()
                    normal_m[i-lo,j]=score.max(1);normal_c[i-lo,j]=score.argmax(1)+1
                    mask=native_mask(row,i,j,n,profiles,known,native_indices);row_negative+=int(mask.sum())
                    neg=np.where(mask,1+score,-np.inf)
                    bad_m[i-lo,j]=neg.max(1);bad_c[i-lo,j]=neg.argmax(1)+1
                if dense_control:
                    dense['normal'][0][lo:hi]=normal_m;dense['normal'][1][lo:hi]=normal_c
                    dense['bad'][0][lo:hi]=bad_m;dense['bad'][1][lo:hi]=bad_c
                    dense['zero'][0][lo:hi]=base.max(2);dense['zero'][1][lo:hi]=base.argmax(2)+1
                for k in range(lo,hi):
                    normal.feed(k,normal_m[k-lo],normal_c[k-lo]);bad.feed(k,bad_m[k-lo],bad_c[k-lo])
                    zero.feed(k,base[k-lo].max(1),base[k-lo].argmax(1)+1)
        chosen=normal.finish();wrong=bad.finish();baseline=zero.finish();ref=original[row]
        if dense_control:
            from weld_pipeline.sjg_joint_pair_model_1248 import select_intervals
            for name,chosen_ref in (('normal',chosen),('bad',wrong),('zero',baseline)):
                assert select_intervals(*dense[name])==chosen_ref,(key,position,name)
            durable_json(folder/f'actual_dense_control_{position}.json',dict(status='PASS',position=position,row=row,
                three_DP_exact_choices_and_scores=True,float32_zero_float64_cumulative=True))
        memory_point(folder,position,'after_scoring',dict(endpoint=int(x.numel()*x.element_size()),
            encoding=int(h.numel()*h.element_size()),evidence_y=int(evidence.y.nbytes),
            evidence_prefix=int(evidence.prefix.nbytes),project_values=int(project.values.nbytes),
            project_prefix=int(project.prefix.nbytes),current_base=int(base.nbytes)))

        assert len(baseline)==len(ref)
        for (i,j,cat,s),v in zip(baseline,ref):
            assert (float(qraw[i]),float(qraw[j]),cat)==(v['lo_mm'],v['hi_mm'],v['category']) and abs(s-v['margin'])<1e-4
        if key in ('21','BM79A_native_local'):
            assert len(chosen)==len(prior[row])
            for (i,j,cat,s),v in zip(chosen,prior[row]):
                assert (float(qraw[i]),float(qraw[j]),cat)==(v['lo_mm'],v['hi_mm'],v['category']) and abs(s-v['unified_score'])<1e-4
        row_H={(i,j,c) for i,j,c,s in wrong}
        for i,j,cat,s in chosen:
            mask=native_mask(row,np.array([i]),np.array([j]),n,profiles,known,native_indices)
            if mask[0,cat-1]:row_H.add((i,j,cat))
            deployment.append(dict(row=row,left=i,right=j,category=cat,unified_score=s,known_wrong=bool(mask[0,cat-1])))
        negative_carriers+=int(row_negative>0);negative_constraints+=row_negative;total+=n*(n-1)//2
        if row_H:
            unique=sorted({(i,j) for i,j,c in row_H});lookup={v:i for i,v in enumerate(unique)}
            pair=np.array([(row,i,j) for i,j in unique],np.int64);indices=np.array([pool.lookup.get(tuple(v),-1) for v in pair],np.int64)
            mask=np.zeros((len(pair),4),bool)
            for i,j,cat in row_H:mask[lookup[i,j],cat-1]=True
            assert np.all(~mask|native_mask(row,pair[:,1],pair[:,2],n,profiles,known,native_indices))
            missing=np.flatnonzero(indices<0);features=np.empty((len(missing),605),np.float32);bases=np.empty((len(missing),4),np.float32)
            cached=cache.get(row)
            if cached:
                cached_path=OUT/'run457/mined_10/blocks'/cached['file'];assert digest(cached_path)==cached['sha256']
                with np.load(cached_path) as z:
                    assert np.array_equal(pair,z['pairs']) and np.array_equal(indices,z['pool_indices'])
                    assert np.array_equal(mask,z['negative']) and np.array_equal(missing,z['new_indices'])
                    features[:]=z['new_features'];bases[:]=z['new_base']
            else:
                for first in range(0,len(missing),4096):
                    ids=missing[first:first+4096];i,j=pair[ids,1],pair[ids,2]
                    features[first:first+len(ids)]=old.features(rt,r,i,j,evidence)
                    bases[first:first+len(ids)],control=sparse(rt,r,i,j,False)
            missing_lookup={int(v):i for i,v in enumerate(missing)}
            for idx in range(len(pair)):
                fx,fb=pool.get(indices[idx]) if indices[idx]>=0 else (features[missing_lookup[idx]],bases[missing_lookup[idx]])
                for cat in np.flatnonzero(mask[idx]):
                    conflict=pos.get(signature(fx,fb,int(cat)))
                    if conflict:
                        atomic_json(folder/'exact_positive_negative_conflict.json',dict(source=key,pair=pair[idx].tolist(),category=int(cat+1),positive_identities=conflict))
                        raise RuntimeError('Identical scoring input has opposite supervised labels; preserved both, stopped')
            file=blocks/f'row_{row}.npz'
            atomic_binary_write(file,lambda f:np.savez(f,pairs=pair,pool_indices=indices,negative=mask,new_indices=missing,new_features=features,new_base=bases))
            ledger.append(dict(file=file.name,row=row,rows=len(pair),negative_count=int(mask.sum()),new_feature_rows=len(missing),sha256=digest(file)))
            exposed.extend((row,i,j,c) for i,j,c in sorted(row_H))
        payload=dict(position=position,row=row,pairs=n*(n-1)//2,negative_constraints=row_negative,
            negative_carrier=bool(row_negative),deployment=deployment,known_wrong_selection=wrong,
            zero_selection=baseline,zero_parity=True,H=exposed,blocks=ledger,
            actual_dense_control=dense_control,cached_features_reused=bool(row in cache and row_H))
        publish(commits,position,binding,payload,blocks/ledger[0]['file'] if ledger else None)
        memory_point(folder,position,'after_commit',dict(new_features=int(features.nbytes) if row_H else 0))
        if position%25==0 or position==stop-1:
            event('CARRIER_COMMITTED',source=key,position=position+1,total=len(rt['records']),batch_stop=stop)
        del x,c,q,p,h,evidence,project,normal,zero,bad,dense,chosen,wrong,baseline,ref,payload
        del endpoints,left,right,output,base,normal_m,normal_c,bad_m,bad_c,score,neg
        if row_H:del features,bases,pair,indices,mask,fx,fb
        gc.collect();check_resources()
    durable_json(folder/f'batch_{start}_{stop}.json',dict(status='COMPLETE',start=start,stop=stop,
        pid=os.getpid(),binding=binding,seconds=time.monotonic()-begin))
    pool.close()
    event('BATCH_COMPLETE',source=key,start=start,stop=stop,seconds=time.monotonic()-begin)

def finalize(key):
    begin=time.monotonic();guard();folder=DEST/f'mined_{key}';blocks=folder/'blocks';commits=folder/'commits'
    binding=binding_for(key);pool=ExistingPool(key)
    with np.load(OUT/f'run430/source_{key}/role_map.npz') as z:record_rows=z['rows']
    total=negative_carriers=negative_constraints=0;ledger=[];deployment=[];exposed=[]
    for position,row in enumerate(record_rows):
        guard();v=recover(commits,position,binding)
        assert v is not None and v['row']==int(row) and v['zero_parity']
        total+=v['pairs'];negative_carriers+=int(v['negative_carrier']);negative_constraints+=v['negative_constraints']
        ledger.extend(v['blocks']);deployment.extend(v['deployment']);exposed.extend(v['H'])
    native=OUT/f'run422/source_{key}'
    atomic_json(folder/'deployment_selected.json',dict(source=key,selections=deployment,training_search_not_final_test=True))
    atomic_json(folder/'exposed_identities.json',dict(identities=exposed))
    new_count=sum(v['new_feature_rows'] for v in ledger)
    if new_count:
        nx=np.lib.format.open_memmap(folder/'exposed_features.npy',mode='w+',dtype=np.float32,shape=(new_count,605))
        nb=np.lib.format.open_memmap(folder/'exposed_base.npy',mode='w+',dtype=np.float32,shape=(new_count,4))
    else:
        nx=np.empty((0,605),np.float32);nb=np.empty((0,4),np.float32)
    old_mask=np.zeros((len(pool.pairs),4),bool);new_mask=np.zeros((new_count,4),bool);offset=0
    for block in ledger:
        guard();assert digest(blocks/block['file'])==block['sha256']
        with np.load(blocks/block['file']) as z:
            existing=z['pool_indices']>=0;old_mask[z['pool_indices'][existing]]|=z['negative'][existing]
            k=len(z['new_indices']);nx[offset:offset+k]=z['new_features'];nb[offset:offset+k]=z['new_base']
            new_mask[offset:offset+k]=z['negative'][z['new_indices']];offset+=k
    assert offset==new_count and int(old_mask.sum()+new_mask.sum())==len(exposed)
    if new_count:nx.flush();nb.flush();nx._mmap.close();nb._mmap.close()
    else:np.save(folder/'exposed_features.npy',nx);np.save(folder/'exposed_base.npy',nb)
    atomic_binary_write(folder/'exposed_masks.npz',lambda f:np.savez(f,old_negative=old_mask,new_negative=new_mask))
    if key=='BM79A_native_local':
        audit=read(OUT/'run456/BM79A_raw_native_identity_audit.json');assert len(deployment)==len(audit['rows'])==50
        observed=collections.Counter(v['supervision'] for v in audit['rows']);assert observed==dict(known_wrong=37,native_compatible=10,UNKNOWN=3)
        H=set(exposed)
        for v,auditrow in zip(deployment,audit['rows']):
            assert v['row']==auditrow['row'] and v['category']==auditrow['category']
            identity=(v['row'],v['left'],v['right'],v['category'])
            assert (identity in H)==(auditrow['supervision']=='known_wrong')
        atomic_json(folder/'BM_previous_raw_coverage_control.json',dict(status='PASS',known_wrong_in_H=37,compatible_excluded=10,UNKNOWN_excluded=3,original_final_FP_covered=28))
    output=dict(source=key,all_pairs_scanned=total,complete_carriers=len(record_rows),known_negative_carriers=negative_carriers,
        full_domain_known_negative_constraints=negative_constraints,exposed_negative_identities=len(exposed),blocks=ledger,
        new_feature_rows=sum(v['new_feature_rows'] for v in ledger),original_zero_DP_parity=True,
        old456_raw_DP_parity=key in ('21','BM79A_native_local'),native_receipt_sha256=digest(native/'receipt.json'),
        frozen_model_sha256=digest(OUT/'run456/unified_model.npz'),fit_calls=0,DEV_reads=0,STEP_reads=0,
        supervision_only_for_training_DP=True,feature_conflicts=0,seconds=time.monotonic()-begin,
        files={f:digest(folder/f) for f in ('exposed_features.npy','exposed_base.npy','exposed_masks.npz','exposed_identities.json','deployment_selected.json')})
    atomic_json(folder/'receipt.json',output);pool.close()
    event('PIECE_COMPLETE',source=key,pairs=total,negative_carriers=negative_carriers,exposed=len(exposed),seconds=output['seconds'])
