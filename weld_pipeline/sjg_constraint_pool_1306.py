"""Extend run452 with sealed raw selections and independent known-wrong masks."""
import collections,gc,os,pickle,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_finite_structured_loss_1298 import legal_best,delta,set_value,structured_score_gradient

DEST=OUT/'run454';TRAIN=('21','BM79A_native_local','10','25','36')


def compile_piece(key):
    assert key in TRAIN
    begin=time.perf_counter();src=OUT/f'run450/workset_{key}';dest=DEST/f'workset_{key}';dest.mkdir()
    receipt=read(src/'receipt.json')
    for filename,field in [('features.npy','features_sha256'),('base_margins.npy','base_sha256'),
                           ('indices_and_masks.npz','masks_sha256'),('parents.json','parents_sha256'),
                           ('complete_parent_witnesses.json','native_witness_sha256')]:
        assert digest(src/filename)==receipt[field];os.link(src/filename,dest/filename)
    with np.load(src/'indices_and_masks.npz') as z:original_pairs=z['pairs']
    prior=OUT/f'run453/workset_{key}'
    prior_receipt=read(prior/'structure_receipt.json')
    for f,h in prior_receipt['files'].items():assert digest(prior/f)==h
    prior_pairs=np.load(prior/'pool_pairs.npy');assert np.array_equal(prior_pairs[:len(original_pairs)],original_pairs)
    with (prior/'structure.pkl').open('rb') as f:prior_data=pickle.load(f)
    prior_records={r['row']:r for r in prior_data['records']}
    pairs=[tuple(map(int,p)) for p in prior_pairs];lookup={p:i for i,p in enumerate(pairs)}
    assert len(lookup)==len(pairs)
    route=read(OUT/'run433/inputs_frozen_before_train_labels.json')['routes'][key]['pair_inputs']
    folder=OUT/route['input_run']/f'source_{key}';mr=read(folder/'receipt.json')
    assert digest(folder/'role_map.npz')==mr['map_sha256']
    with np.load(folder/'role_map.npz') as z:m={k:z[k] for k in z.files}
    rowmap={int(r):j for j,r in enumerate(m['rows'])}
    def add(p):
        p=tuple(map(int,p))
        if p not in lookup:lookup[p]=len(pairs);pairs.append(p)
        return lookup[p]
    raw_selections={};selection_files=[('old432',OUT/f'run432/proposal_{key}/selected_intervals.json')]
    if key=='21':selection_files.append(('failed450',OUT/'run450/trained/proposal_21/selected_intervals.json'))
    if key in ('21','BM79A_native_local'):selection_files.append(('latest453',OUT/f'run453/trained/proposal_{key}/selected_intervals.json'))
    bindings={};all_saved=[]
    for label,path in selection_files:
        bindings[label]=digest(path);byrow=collections.defaultdict(list)
        for query,v in enumerate(read(path)['intervals']):
            row=int(v['row']);j=rowmap[row];a,b=m['offsets'][j:j+2];q=m['q_mm'][a:b]
            i=int(np.searchsorted(q,v['lo_mm']));k=int(np.searchsorted(q,v['hi_mm']))
            assert q[i]==v['lo_mm'] and q[k]==v['hi_mm'] and i<k
            ix=add((row,i,k));byrow[row].append((ix,int(v['category'])-1))
            if label=='failed450':all_saved.append(dict(query=query,index=ix,category=int(v['category'])-1,row=row,
                saved_margin=v['margin'],saved_unified_score=v['unified_score']))
        raw_selections[label]=dict(byrow)
    pairs=np.asarray(pairs,np.int64);count=len(original_pairs);prior_count=len(prior_pairs);missing=pairs[prior_count:]
    extra_x=np.empty((len(missing),605),np.float32);extra_base=np.empty((len(missing),4),np.float32)
    if len(missing):
        from weld_pipeline import sjg_resumable_unified_workset_1293 as compiler
        compiler.DEST=DEST;rt=compiler.runtime(key);rmap={r['row']:r for r in rt['records']}
        for number,row in enumerate(np.unique(missing[:,0])):
            ids=np.flatnonzero(missing[:,0]==row);r=rmap[int(row)]
            extra_x[ids]=compiler.old.features(rt,r,missing[ids,1],missing[ids,2])
            extra_base[ids],control=compiler.sparse(rt,r,missing[ids,1],missing[ids,2],number==0)
        compiler.close(rt)
    extra_x=np.concatenate([np.load(prior/'extra_features.npy'),extra_x])
    extra_base=np.concatenate([np.load(prior/'extra_base.npy'),extra_base])
    assert np.isfinite(extra_x).all() and np.isfinite(extra_base).all()
    np.save(dest/'extra_features.npy',extra_x);np.save(dest/'extra_base.npy',extra_base)
    np.save(dest/'pool_pairs.npy',pairs)
    native=OUT/f'run422/source_{key}';nr=read(native/'receipt.json')
    for f,field in [('compatibility.json','compatibility_sha256'),('pairs.npz','pairs_sha256')]:assert digest(native/f)==nr[field]
    relation=read(native/'compatibility.json');profiles={p['profile']:p for p in relation['profiles']}
    known={r['row']:r['complete_background_supervision_allowed'] for r in relation['rows']}
    with np.load(native/'pairs.npz') as z:native_pairs=z['indices']
    owners=relation['owners'];records=[];memory={};masked=collections.Counter();owner_counts=collections.Counter()
    zero=np.zeros((len(pairs),4))
    for row in np.unique(pairs[:,0]):
        row=int(row);ids=np.flatnonzero(pairs[:,0]==row);p=profiles[row//2];n=p['n']
        encoded=pairs[ids,1]*n+pairs[ids,2];groups={};compat=collections.defaultdict(list)
        sort_keys={}
        for g in p['groups']:
            assert not g['implicit_all_pairs']
            oi=g['owner_index'];group=groups.setdefault(oi,dict(owner=oi,parent=owners[oi]['parent'],options=[]))
            native_ids=native_pairs[g['pair_start']:g['pair_stop']]
            sort_keys[oi]=(float(np.median(native_ids//n)),float(np.median(native_ids%n)),oi)
            positions=np.searchsorted(native_ids,encoded)
            matches=np.flatnonzero((positions<len(native_ids))&(native_ids[np.minimum(positions,len(native_ids)-1)]==encoded))
            for cat,parity in ((g['native_category']-1,False),(g['reversed_parent_category']-1,True)):
                for i in matches:
                    option=(int(i),int(cat),parity)
                    if option not in group['options']:group['options'].append(option)
                    if (oi,parity) not in compat[int(i),int(cat)]:compat[int(i),int(cat)].append((oi,parity))
        groups=sorted(groups.values(),key=lambda g:sort_keys[g['owner']])
        for g in groups:g['options'].sort()
        r=dict(row=row,indices=ids,groups=groups,compat=dict(compat),eligible=bool(known.get(row,False)),weight=0.)
        reason=None
        if not r['eligible']:reason='UNKNOWN_background'
        elif any(not g['options'] for g in groups):reason='finite_pool_missing_compatible_original_child'
        else:
            gold,value=legal_best(r,pairs[ids],zero[ids])
            if gold is None:reason='no_complete_nonoverlap_whole_parent_parity_target_in_pool'
        if reason:r['eligible']=False;r['mask_reason']=reason;masked[reason]+=1
        else:
            for g in groups:owner_counts[g['owner']]+=1
        global_to_local={int(v):i for i,v in enumerate(ids)};pool={():len(groups)}
        for label,byrow in raw_selections.items():
            if row in byrow:
                y=tuple(sorted(((global_to_local[i],c) for i,c in byrow[row]),key=lambda z:(pairs[ids[z[0]],1],pairs[ids[z[0]],2],z[1])))
                if r['eligible']:pool[y]=delta(r,y)
        if r['eligible']:pool[gold]=0
        memory[row]=pool;records.append(r)
    # Freeze run452 complete-risk eligibility and numeric carrier weights.
    # The separate deletion loss does not consult this eligibility flag.
    owner_counts=collections.Counter();masked=collections.Counter()
    for r in records:
        old=prior_records.get(r['row']);r['new_pool_full_gold_feasible']=r['eligible']
        if old and old['eligible']:assert r['eligible']
        r['eligible']=bool(old and old['eligible']);r['full_gold_verified']=r['eligible']
        if not r['eligible']:
            r['mask_reason']=old.get('mask_reason','prior_complete_risk_inactive') if old else 'new_carrier_complete_risk_inactive'
            masked[r['mask_reason']]+=1
        else:
            for g in r['groups']:owner_counts[g['owner']]+=1
        wrong=np.full((len(r['indices']),4),bool(known.get(r['row'],False)),bool)
        for (i,c),compatible_owners in r['compat'].items():
            if compatible_owners:wrong[i,c]=False
        r['known_wrong']=wrong
    # These intermediate responsibilities are NOT independent owner hinge weights.
    # The original carrier-level weighted risk is deliberately unchanged.
    G=receipt['G'];background=sum(r['eligible'] and not r['groups'] for r in records)
    mass=collections.defaultdict(float)
    for r in records:
        if not r['eligible']:continue
        if r['groups']:
            r['owner_responsibility']={g['owner']:1/(G*owner_counts[g['owner']]) for g in r['groups']}
            r['weight']=sum(r['owner_responsibility'].values())/len(r['groups'])
            for oi,w in r['owner_responsibility'].items():mass[oi]+=w
        else:r['weight']=1/background
    assert all(abs(v-1/G)<1e-12 for v in mass.values())
    for r in records:
        old=prior_records.get(r['row'])
        r['weight']=float(old['weight']) if old else 0.
        if old and 'owner_responsibility' in old:r['owner_responsibility']=old['owner_responsibility']
    known_wrong_carriers=sum(r['known_wrong'].any() for r in records)
    data=dict(records=records,memory=memory,original_rows=count,all_saved=all_saved,G=G)
    atomic_binary_write(dest/'structure.pkl',lambda f:pickle.dump(data,f,protocol=5))
    output=dict(source=key,G=G,original_rows=count,pool_rows=len(pairs),new_feature_rows=len(missing),prior452_rows=prior_count,known_wrong_carriers=int(known_wrong_carriers),full_structure_weights_and_eligibility_frozen_from452=True,
        eligible_carriers=sum(r['eligible'] for r in records),masked=dict(masked),
        eligible_owners=len(owner_counts),structure_only_uncovered_original_owners=[i for i in range(G) if i not in owner_counts],
        original_positive_loss_and_denominator_unchanged=True,intermediate_owner_partition_mass=dict(mass),
        outer_normalization='frozen run452 carrier-level weighted risk; NOT independent per-owner hinge mass',
        internal_set_score_is_unnormalized_sum=True,selection_sha256=bindings,native_receipt_sha256=digest(native/'receipt.json'),
        original_receipt_sha256=digest(src/'receipt.json'),fit_calls=0,DEV_reads=0,STEP_reads=0,seconds=time.perf_counter()-begin)
    output['files']={f:digest(dest/f) for f in ('extra_features.npy','extra_base.npy','pool_pairs.npy','structure.pkl')}
    write(dest/'structure_receipt.json',output)
    print('STRUCTURE_POOL',key,'ROWS',len(pairs),'EXTRA',len(missing),'ELIGIBLE',output['eligible_carriers'],'KNOWN_WRONG_CARRIERS',known_wrong_carriers,'MASKED',dict(masked),flush=True)
    del extra_x,extra_base;gc.collect();check_resources()


def load_piece(key):
    dest=DEST/f'workset_{key}';receipt=read(dest/'structure_receipt.json')
    for f,h in receipt['files'].items():assert digest(dest/f)==h
    with (dest/'structure.pkl').open('rb') as f:data=pickle.load(f)
    data['pairs']=np.load(dest/'pool_pairs.npy')
    data['x']=np.concatenate([np.load(dest/'features.npy'),np.load(dest/'extra_features.npy')])
    data['base']=np.concatenate([np.load(dest/'base_margins.npy'),np.load(dest/'extra_base.npy')])
    data['receipt']=receipt
    return data

