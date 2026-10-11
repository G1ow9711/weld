"""Compile native complete-parent witnesses and input-only interval evidence.

TRAIN targets are used only for witness/negative masks, never feature values.
No DEV target is accessible from the training entry point.
"""
import argparse,collections,gc,json,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_unified_interval_math_1287 import ArcLengthEvidence,complete_features

TRAIN=('21','BM79A_native_local','10','25','36')
DEST=OUT/'run449'

def load_runtime(key):
    import torch
    from weld_pipeline.sjg_joint_pair_model_1248 import PairModel,load_inputs
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False
    protocol=read(DEST/'registered_protocol.json')
    for path,h in protocol['frozen_models'].items():assert digest(OUT/path)==h
    route=read(OUT/'run433/inputs_frozen_before_train_labels.json')['routes'][key]
    source=load_inputs(key,**route['pair_inputs'])
    payload=torch.load(OUT/'run432/pair_model.pt',map_location='cpu')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=PairModel(payload['classes']).to(device).eval();model.load_state_dict(payload['state_dict'])
    assert payload['classes']==[0,1,2,3,4]
    seq=OUT/f'run421/sequence56/source_{key}';sr=read(seq/'receipt.json')
    assert digest(seq/'features.npy')==sr['features_sha256'] and digest(seq/'sequence_map.npz')==sr['map_sha256']
    with np.load(seq/'sequence_map.npz') as z:sm={k:z[k] for k in z.files}
    sf=np.load(seq/'features.npy',mmap_mode='r')
    aux=torch.load(OUT/'run429/sequence_model.pt',map_location='cpu')
    case=load_case(key);case['x']._mmap.close();graph=case['graph']
    records=[]
    for i,row in enumerate(source['m']['rows']):
        a,b=map(int,source['m']['offsets'][i:i+2]);curve=graph.curve(int(row)//2)
        records.append(dict(row=int(row),q_offset=a,feature_offset=2*a,n=b-a,
            curve_length_mm=curve.length,is_closed=bool(curve.kind=='arc' and curve.arc.is_closed)))
    return dict(key=key,s=source,payload=payload,device=device,model=model,graph=graph,records=records,
        seq=sf,seqmap=sm,seqlookup={int(p):i for i,p in enumerate(sm['profile_ids'])},
        seqmean=aux['mean'],seqscale=aux['scale'],route=route)

def evidence(rt,r):
    from weld_pipeline.sjg_dense_joint_sequence_1237 import ordered_features
    sm=rt['seqmap'];j=rt['seqlookup'][r['row']//2];a,b=sm['offsets'][j:j+2]
    return ArcLengthEvidence(sm['q_mm'][a:b],ordered_features(rt['seq'][a:b],r['row']%2),
        rt['seqmean'],rt['seqscale'],r['is_closed'])

def scalars(rt,r,ii,jj):
    s=rt['s'];a=r['q_offset'];q=s['m']['q_mm'][a:a+r['n']];span=q[jj]-q[ii]
    return np.c_[np.log1p(span)/10,span/r['curve_length_mm'],
        (s['prefix'][a+jj]-s['prefix'][a+ii])/span,np.full(len(ii),float(r['is_closed']))]

def features(rt,r,ii,jj,e=None):
    s=rt['s'];a=r['q_offset'];n=r['n'];norm=rt['payload']['norm'];q=s['m']['q_mm'][a:a+n]
    left=(np.asarray(s['x'][2*a+ii])-norm['mean'])/norm['scale']
    right=(np.asarray(s['x'][2*a+n+jj])-norm['mean'])/norm['scale']
    return complete_features(left,right,(e or evidence(rt,r)).features(q[ii],q[jj]),scalars(rt,r,ii,jj)).astype(np.float32)

def sparse_margins(rt,r,ii,jj):
    import torch
    from weld_pipeline.sjg_joint_pair_model_1248 import tensors
    x,c,q,p=tensors(rt['s'],r,rt['payload']['norm'],rt['device']);model=rt['model'];n=r['n']
    with torch.no_grad():
        h=model.encode(x,c);i=torch.as_tensor(ii,device=rt['device']);j=torch.as_tensor(jj,device=rt['device'])
        a=h[i];b=h[n+j];span=q[j]-q[i]
        geom=torch.stack((torch.log1p(span)/10,span/r['curve_length_mm'],(p[j]-p[i])/span,
            torch.full_like(span,float(r['is_closed']))),-1)
        score=torch.einsum('ik,ckl,il->ic',a,model.pair,b)/np.sqrt(32)+model.unary(a)+model.unary(b)+model.span(geom)
        result=(score[:,1:]-score[:,:1]).cpu().numpy()
        # Independent comparison with the original full-row scoring path.
        unique=np.unique(ii)
        maximum=0.
        for left in unique[:2]:
            ids=np.flatnonzero(ii==left);full=model.scores(h[:n],h[n:],q,p,r['curve_length_mm'],r['is_closed'],int(left),int(left)+1)
            ref=(full[0,:,1:]-full[0,:,:1]).cpu().numpy()[jj[ids]]
            maximum=max(maximum,float(abs(result[ids]-ref).max()))
        assert maximum<1e-4,(r['row'],maximum)
    return result,maximum

def choice(rt,r,i,j,cat,query=0,margin=0.):
    from weld_pipeline.sjg_joint_forest_train_1109 import attributes
    a=r['q_offset'];q=rt['s']['m']['q_mm'][a:a+r['n']];attr=attributes(cat)
    return dict(query=query,probability=float(np.exp(-np.logaddexp(0.,-margin))),child_count=1,closed=attr['closed'],
        children=[dict(profile=r['row']//2,intervals=[0,0],fractions=[float(q[i]/r['curve_length_mm']),float(q[j]/r['curve_length_mm'])],
        side='UNKNOWN',connection='UNKNOWN',**attr)])

def witnesses(rt,targets):
    from weld_pipeline.sjg_recall_set_worker_1162 import target_for
    from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets,score_native_parents
    from weld_pipeline.world_set_evaluation_285 import reference_segment
    from weld_pipeline.sjg_finite_face_domains_1235 import decode_shared,ContinuousGraphBank
    key=rt['key'];target,binding=target_for(key);assert binding['target_sha256']==targets['original_target_sha256']
    native=native_targets(target);refs={};best={};bank=ContinuousGraphBank(rt['graph'])
    for pi,p in enumerate(native['parents']):
        for ci,c in enumerate(p['children']):refs[pi,ci]=np.asarray(reference_segment(c)['points'])[[0,-1]]
    for r in targets['records']:
        if not r['gold_groups']:continue
        n=r['n'];a=r['q_offset'];q=rt['s']['m']['q_mm'][a:a+n];xyz=rt['graph'].curve(r['row']//2).at(q)
        for group in r['gold_groups']:
            owner=tuple(group['owner']);ids=np.asarray(group['pairs'],np.int64);ii,jj=ids//n,ids%n
            for cat in group['allowed_categories']:
                for parity in group['category_parent_parities'][str(cat)]:
                    ep=refs[owner][::-1] if parity else refs[owner]
                    forward=cat>=3;l,rp=(ii,jj) if forward else (jj,ii)
                    errors=np.linalg.norm(xyz[l]-ep[0],axis=1)+np.linalg.norm(xyz[rp]-ep[1],axis=1)
                    pick=int(errors.argmin());entry=(float(errors[pick]),int(r['row']),int(ii[pick]),int(jj[pick]),int(cat))
                    identity=(*owner,bool(parity))
                    if identity not in best or entry<best[identity]:best[identity]=entry
    byrow={r['row']:r for r in rt['records']};views=[];audits=[]
    for pi,parent in enumerate(target['parents']):
        alternatives=[]
        for parity in (False,True):
            children=range(len(parent['children'])-1,-1,-1) if parity else range(len(parent['children']))
            indices=[best[pi,ci,parity][1:] for ci in children]
            for u,(row,i,j,cat) in enumerate(indices):
                for row2,i2,j2,cat2 in indices[u+1:]:
                    assert row!=row2 or max(i,i2)>=min(j,j2),(key,pi,'native complete witness violates interval nonoverlap')
            decoded=decode_shared([choice(rt,byrow[row],i,j,cat,k) for k,(row,i,j,cat) in enumerate(indices)],bank)
            assert not decoded['invalid_predictions'] and len(decoded['parents'])==len(indices)
            witness=dict(child_count=len(indices),is_closed=parent['is_closed'],
                gaps=parent['gaps'][::-1] if parity else parent['gaps'],segments=[p['segments'][0] for p in decoded['parents']])
            result=score_native_parents(dict(parents=[witness],invalid_predictions=[],total_prediction_children=len(indices)),
                dict(original_children=len(indices),parents=[parent]))
            assert result['categories']['overall']['TP']==len(indices),(key,pi,parity,result['categories']['overall'])
            assert result['categories']['overall']['FP']==0
            alternatives.append(indices)
            audits.append(dict(parent=pi,whole_parent_reverse=parity,children=len(indices),
                max_mm=max(m['bidirectional_sampled_maximum_mm'] for m in result['matches']),
                input_native_pairs=[list(z) for z in indices]))
        views.append(alternatives)
    assert sum(len(p[0]) for p in views)==target['original_children']==targets['original_G']
    return views,dict(source=key,G=target['original_children'],parents=len(views),views=audits,
        status='ALL_COMPLETE_PARENT_NATIVE_WITNESSES_D3_VERIFIED',target_sha256=binding['target_sha256'],
        gap_order_and_closure_are_supervision_NOT_predicted=True,formal_attributes_NOT_certified=True)

def uniform_pairs(records,rng,size):
    counts=np.array([r['n']*(r['n']-1)//2 for r in records],np.int64);cumulative=np.cumsum(counts)
    global_ids=rng.integers(0,int(cumulative[-1]),size=size);ri=np.searchsorted(cumulative,global_ids,side='right')
    before=np.r_[0,cumulative[:-1]];out=[]
    for rowid,k in zip(ri,global_ids-before[ri]):
        r=records[int(rowid)];n=r['n'];starts=np.arange(n,dtype=np.int64);starts=starts*(2*n-starts-1)//2
        i=int(np.searchsorted(starts,k,side='right')-1);j=int(i+1+k-starts[i]);assert 0<=i<j<n
        out.append((r['row'],i,j))
    return out,global_ids,int(cumulative[-1])

def prepare(key):
    assert key in TRAIN;begin=time.perf_counter();check_resources();rt=load_runtime(key)
    targets=read(OUT/f'run431/source_{key}_set_targets.json')
    expected=next(c['target_sha256'] for c in read(OUT/'run431/receipt.json')['cases'] if c['source']==key)
    assert digest(OUT/f'run431/source_{key}_set_targets.json')==expected
    dest=DEST/f'workset_{key}';dest.mkdir();views,audit=witnesses(rt,targets);write(dest/'complete_parent_witnesses.json',audit)
    records=rt['records'];byrow={r['row']:r for r in records};lookup={};pairs=[]
    def index(z):
        z=tuple(map(int,z))
        if z not in lookup:lookup[z]=len(pairs);pairs.append(z)
        return lookup[z]
    parents=[[[[index(z[:3]),z[3]-1] for z in view] for view in p] for p in views]
    positive_rows=sorted({i for p in parents for view in p for i,c in view})
    from weld_pipeline.sjg_conditional_diagnostics_1284 import labels
    raw=read(OUT/f'run432/proposal_{key}/selected_intervals.json')['intervals'];y=labels(key);rawbad=[]
    for k,v in enumerate(raw):
        if y[k]!=0:continue
        r=byrow[v['row']];a=r['q_offset'];q=rt['s']['m']['q_mm'][a:a+r['n']]
        i=int(np.searchsorted(q,v['lo_mm']));j=int(np.searchsorted(q,v['hi_mm']))
        assert q[i]==v['lo_mm'] and q[j]==v['hi_mm'];rawbad.append((index((r['row'],i,j)),v['category']-1,k))
    rng=np.random.default_rng(1031);sampled=[];global_samples=[];rng_states=[]
    for epoch in range(16):
        sample,gids,total=uniform_pairs(records,rng,4096);sampled.append([index(z) for z in sample]);global_samples.append(gids);rng_states.append(rng.bit_generator.state)
    pairs=np.asarray(pairs,np.int64);known={r['row']:r['complete_background_supervision_allowed'] for r in read(OUT/f'run422/background_{key}.json')['rows']}
    groups={r['row']:r['gold_groups'] for r in targets['records']};negative=np.zeros((len(pairs),4),bool)
    # Full native compatibility, including unselected aliases, determines masks.
    for row in np.unique(pairs[:,0]):
        ix=np.flatnonzero(pairs[:,0]==row);r=byrow[int(row)]
        if known.get(int(row),False):negative[ix]=True
        pid=pairs[ix,1]*r['n']+pairs[ix,2]
        for g in groups.get(int(row),[]):
            ids=np.asarray(g['pairs'],np.int64);pos=np.searchsorted(ids,pid);match=(pos<len(ids))&(ids[np.minimum(pos,len(ids)-1)]==pid)
            for c in g['allowed_categories']:negative[ix[match],c-1]=False
    for i,c,rawindex in rawbad:assert negative[i,c],(key,rawindex,'known raw error conflicts with full native compatibility')
    for p in parents:
        for view in p:
            for i,c in view:assert not negative[i,c]
    x=np.lib.format.open_memmap(dest/'features.npy',mode='w+',dtype=np.float32,shape=(len(pairs),605))
    base=np.lib.format.open_memmap(dest/'base_margins.npy',mode='w+',dtype=np.float32,shape=(len(pairs),4));maximum=0.
    for number,row in enumerate(np.unique(pairs[:,0])):
        ix=np.flatnonzero(pairs[:,0]==row);r=byrow[int(row)];ii,jj=pairs[ix,1],pairs[ix,2]
        x[ix]=features(rt,r,ii,jj);base[ix],error=sparse_margins(rt,r,ii,jj);maximum=max(maximum,error)
        if number%250==0:
            x.flush();base.flush();print('WORKSET',key,'CARRIERS',number,'/',len(np.unique(pairs[:,0])),flush=True)
            reclaim_own_idle_pages();check_resources()
    x.flush();base.flush();x._mmap.close();base._mmap.close()
    np.savez(dest/'indices_and_masks.npz',pairs=pairs,negative=negative,sampled=np.asarray(sampled,np.int64),
        global_samples=np.asarray(global_samples,np.int64),positive_rows=positive_rows,
        raw_bad=np.asarray(rawbad,np.int64).reshape(-1,3))
    write(dest/'parents.json',dict(parents=parents,G=targets['original_G'],rng_states=rng_states))
    write(dest/'receipt.json',dict(source=key,G=targets['original_G'],pairs=len(pairs),uniform_domain_pairs=total,
        per_draw_probability=1/total,samples_per_epoch=4096,passes=16,seed=1031,
        native_witness_sha256=digest(dest/'complete_parent_witnesses.json'),
        features_sha256=digest(dest/'features.npy'),base_sha256=digest(dest/'base_margins.npy'),
        masks_sha256=digest(dest/'indices_and_masks.npz'),parents_sha256=digest(dest/'parents.json'),
        sparse_vs_original_max_abs=maximum,all_known_raw_negatives=len(rawbad),
        unknown_not_replenished_or_negative=True,unselected_native_aliases_not_negative=True,
        DEV_reads=0,STEP_reads=0,fit_calls=0,seconds=time.perf_counter()-begin))
    print('WORKSET_COMPLETE',key,'G',targets['original_G'],'PAIRS',len(pairs),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args();prepare(a.source)
