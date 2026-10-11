"""Reuse frozen manifests, repair only the omitted original score clamp, commit blocks."""
import ctypes,gc,hashlib,json,os,shutil,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,ROOT,read,write,digest,check_resources,resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline import sjg_unified_interval_workset_1288 as old

DEST=OUT/'run450';MODEL=None;PAYLOAD=None;AUX=None;DEVICE=None

def event(stage,**details):
    values=(ctypes.c_uint64*6)();api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetProcessIoCounters.argtypes=[ctypes.c_void_p,ctypes.c_void_p];api.GetProcessIoCounters.restype=ctypes.c_int
    assert api.GetProcessIoCounters(ctypes.c_void_p(-1),ctypes.byref(values))
    value=dict(stage=stage,monotonic=time.monotonic(),cpu_seconds=time.process_time(),resources=resources(),
        io=dict(zip(('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes'),map(int,values))),**details)
    with (DEST/'compiler_stages.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    print('COMPILER_STAGE',stage,details,flush=True)

def models():
    global MODEL,PAYLOAD,AUX,DEVICE
    if MODEL is not None:return
    event('IMPORT_START')
    import torch
    from weld_pipeline.sjg_joint_pair_model_1248 import PairModel
    torch.set_num_threads(2);torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False
    PAYLOAD=torch.load(OUT/'run432/pair_model.pt',map_location='cpu')
    AUX=torch.load(OUT/'run429/sequence_model.pt',map_location='cpu')
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    MODEL=PairModel(PAYLOAD['classes']).to(DEVICE).eval();MODEL.load_state_dict(PAYLOAD['state_dict'])
    event('IMPORT_END',device=DEVICE)

def runtime(key):
    models();event('LOAD_ARRAYS_START',source=key)
    from weld_pipeline.sjg_joint_pair_model_1248 import load_inputs
    route=read(OUT/'run433/inputs_frozen_before_train_labels.json')['routes'][key]
    s=load_inputs(key,**route['pair_inputs']);folder=OUT/f'run421/sequence56/source_{key}';sr=read(folder/'receipt.json')
    assert digest(folder/'features.npy')==sr['features_sha256'] and digest(folder/'sequence_map.npz')==sr['map_sha256']
    with np.load(folder/'sequence_map.npz') as z:sm={k:z[k] for k in z.files}
    sf=np.load(folder/'features.npy',mmap_mode='r');sl={int(p):i for i,p in enumerate(sm['profile_ids'])};records=[]
    for j,row in enumerate(s['m']['rows']):
        a,b=map(int,s['m']['offsets'][j:j+2]);k=sl[int(row)//2];u,v=map(int,sm['offsets'][k:k+2])
        records.append(dict(row=int(row),q_offset=a,feature_offset=2*a,n=b-a,
            curve_length_mm=float(sm['q_mm'][v-1]),is_closed=bool(sf[u,27])))
    event('LOAD_ARRAYS_END',source=key,carriers=len(records),point_reconstruction_calls=0)
    return dict(key=key,s=s,payload=PAYLOAD,device=DEVICE,model=MODEL,records=records,seq=sf,seqmap=sm,
        seqlookup=sl,seqmean=AUX['mean'],seqscale=AUX['scale'],route=route)

def close(rt):
    rt['s']['x']._mmap.close();rt['s']['context']._mmap.close();rt['seq']._mmap.close();rt.clear();gc.collect()

def sparse(rt,r,ii,jj,validate=False):
    """Original run449 sparse batch shape, with the missing original clamp restored."""
    import torch
    from weld_pipeline.sjg_joint_pair_model_1248 import tensors
    x,c,q,p=tensors(rt['s'],r,rt['payload']['norm'],rt['device']);model=rt['model'];n=r['n']
    with torch.no_grad():
        h=model.encode(x,c);i=torch.as_tensor(ii,dtype=torch.long,device=rt['device']);j=torch.as_tensor(jj,dtype=torch.long,device=rt['device'])
        a=h[i];b=h[n+j];span=(q[j]-q[i]).clamp_min(0)
        geometry=torch.stack((torch.log1p(span)/10,span/r['curve_length_mm'],(p[j]-p[i])/span.clamp_min(1e-6),
            torch.full_like(span,float(r['is_closed']))),-1)
        score=torch.einsum('ik,ckl,il->ic',a,model.pair,b)/np.sqrt(32)+model.unary(a)+model.unary(b)+model.span(geometry)
        result=(score[:,1:]-score[:,:1]).cpu().numpy();assert np.isfinite(result).all()
        short=(span<=1e-6).cpu().numpy();lefts=np.unique(ii[short])
        if validate:lefts=np.union1d(lefts,np.unique(ii)[:2])
        maximum=0.;reference_pairs=0
        for left in lefts:
            pick=np.flatnonzero(ii==left);full=model.scores(h[:n],h[n:],q,p,r['curve_length_mm'],r['is_closed'],int(left),int(left)+1)
            reference=(full[0,:,1:]-full[0,:,:1]).cpu().numpy()[jj[pick]];assert np.isfinite(reference).all()
            error=float(abs(result[pick]-reference).max());maximum=max(maximum,error);reference_pairs+=n
        assert maximum<1e-4,(r['row'],maximum)
    return result,dict(maximum_original_forward_error=maximum,reference_pairs=reference_pairs,short_pairs=int(short.sum()))

def hardlink(src,dst):
    assert not dst.exists();os.link(src,dst);assert digest(src)==digest(dst)

def reuse(key):
    begin=time.perf_counter();src=OUT/f'run449/workset_{key}';dest=DEST/f'workset_{key}';dest.mkdir()
    r=read(src/'receipt.json');assert r['G']==read(src/'complete_parent_witnesses.json')['G']
    for filename,field in [('features.npy','features_sha256'),('base_margins.npy','base_sha256'),
            ('indices_and_masks.npz','masks_sha256'),('parents.json','parents_sha256'),
            ('complete_parent_witnesses.json','native_witness_sha256')]:
        assert digest(src/filename)==r[field]
        if filename=='base_margins.npy':shutil.copy2(src/filename,dest/filename)
        else:hardlink(src/filename,dest/filename)
    with np.load(dest/'indices_and_masks.npz') as z:pairs=z['pairs']
    mf=OUT/f'run430/source_{key}/role_map.npz';assert digest(mf)==read(mf.parent/'receipt.json')['map_sha256']
    with np.load(mf) as z:m={k:z[k] for k in z.files}
    rows={int(row):j for j,row in enumerate(m['rows'])};base=np.load(dest/'base_margins.npy',mmap_mode='r+')
    repair=[]
    for row in np.unique(pairs[:,0]):
        ids=np.flatnonzero(pairs[:,0]==row);j=rows[int(row)];a,b=m['offsets'][j:j+2];q=m['q_mm'][a:b].astype(np.float32)
        small=(q[pairs[ids,2]]-q[pairs[ids,1]])<1e-6;repair.extend(ids[small].tolist())
    repair=np.asarray(sorted(repair),np.int64);bad=np.flatnonzero(~np.isfinite(base).all(1));assert set(bad)<=set(repair)
    changes=[]
    if len(repair):
        rt=runtime(key);records={v['row']:v for v in rt['records']}
        for row in np.unique(pairs[repair,0]):
            # Preserve the original sparse batch shape for this carrier.
            ids=np.flatnonzero(pairs[:,0]==row);rrow=records[int(row)]
            values,control=sparse(rt,rrow,pairs[ids,1],pairs[ids,2],True)
            keep=np.isin(ids,repair);unchanged=~keep
            assert np.isfinite(base[ids[unchanged]]).all()
            assert np.allclose(base[ids[unchanged]],values[unchanged],rtol=0,atol=1e-4)
            base[ids[keep]]=values[keep];changes.append(dict(row=int(row),repaired_rows=ids[keep].tolist(),**control))
        close(rt)
    base.flush();assert np.isfinite(base).all();base._mmap.close()
    updated=dict(r,base_sha256=digest(dest/'base_margins.npy'),reused_run449_features_and_masks_exact=True,
        short_interval_clamp_restored=True,nonfinite_before=len(bad),repaired_rows=len(repair),fit_calls=0)
    write(dest/'base_repair.json',dict(changes=changes,source_original_not_changed=True,
        unchanged_finite_rows_not_recomputed_or_overwritten=True,seconds=time.perf_counter()-begin))
    write(dest/'receipt.json',updated);event('REUSE_COMMIT',source=key,G=r['G'],repaired_rows=len(repair),bad_before=len(bad))

def manifest36(rt,dest):
    event('DRAW_MANIFEST_START',source='36');w=read(OUT/'run449/workset_36/complete_parent_witnesses.json')
    hardlink(OUT/'run449/workset_36/complete_parent_witnesses.json',dest/'complete_parent_witnesses.json')
    lookup={};pairs=[]
    def index(z):
        z=tuple(map(int,z))
        if z not in lookup:lookup[z]=len(pairs);pairs.append(z)
        return lookup[z]
    parents=[]
    for pi in range(w['parents']):
        views=w['views'][2*pi:2*pi+2];assert all(v['parent']==pi for v in views)
        parents.append([[[index(z[:3]),z[3]-1] for z in v['input_native_pairs']] for v in views])
    positive=sorted({i for p in parents for v in p for i,c in v})
    lp=OUT/'run434/source_36/training_targets.npz'
    assert digest(lp)==next(c['labels_sha256'] for c in read(OUT/'run434/supervision_receipt.json')['cases'] if c['source']=='36')
    with np.load(lp) as z:y=z['validity']
    records={r['row']:r for r in rt['records']};rawbad=[]
    for k,v in enumerate(read(OUT/'run432/proposal_36/selected_intervals.json')['intervals']):
        if y[k]!=0:continue
        r=records[v['row']];a=r['q_offset'];q=rt['s']['m']['q_mm'][a:a+r['n']]
        i=int(np.searchsorted(q,v['lo_mm']));j=int(np.searchsorted(q,v['hi_mm']))
        assert q[i]==v['lo_mm'] and q[j]==v['hi_mm'];rawbad.append([index((v['row'],i,j)),v['category']-1,k])
    rng=np.random.default_rng(1031);sampled=[];global_samples=[];rng_states=[]
    for epoch in range(16):
        sample,global_ids,total=old.uniform_pairs(rt['records'],rng,4096)
        sampled.append([index(z) for z in sample]);global_samples.append(global_ids);rng_states.append(rng.bit_generator.state)
    pairs=np.asarray(pairs,np.int64);negative=np.zeros((len(pairs),4),bool)
    native=OUT/'run422/source_36';nr=read(native/'receipt.json')
    assert digest(native/'compatibility.json')==nr['compatibility_sha256'] and digest(native/'pairs.npz')==nr['pairs_sha256']
    relation=read(native/'compatibility.json');profiles={p['profile']:p for p in relation['profiles']}
    known={r['row']:r['complete_background_supervision_allowed'] for r in relation['rows']}
    with np.load(native/'pairs.npz') as z:native_indices=z['indices']
    for row in np.unique(pairs[:,0]):
        ids=np.flatnonzero(pairs[:,0]==row);n=records[int(row)]['n']
        if known.get(int(row),False):negative[ids]=True
        encoded=pairs[ids,1]*n+pairs[ids,2]
        for group in profiles[int(row)//2]['groups']:
            assert not group['implicit_all_pairs']
            compatible=native_indices[group['pair_start']:group['pair_stop']];positions=np.searchsorted(compatible,encoded)
            if not len(compatible):continue
            assert np.all(np.diff(compatible)>=0)
            mask=(positions<len(compatible))&(compatible[np.minimum(positions,len(compatible)-1)]==encoded)
            for cat in (group['native_category'],group['reversed_parent_category']):negative[ids[mask],cat-1]=False
    for p in parents:
        for view in p:
            for i,c in view:assert not negative[i,c]
    for i,c,k in rawbad:assert negative[i,c]
    np.savez(dest/'indices_and_masks.npz',pairs=pairs,negative=negative,sampled=np.asarray(sampled,np.int64),
        global_samples=np.asarray(global_samples,np.int64),positive_rows=positive,raw_bad=np.asarray(rawbad,np.int64).reshape(-1,3))
    write(dest/'parents.json',dict(parents=parents,G=32,rng_states=rng_states))
    assert len(pairs)==np.load(OUT/'run449/workset_36/features.npy',mmap_mode='r').shape[0]
    event('DRAW_MANIFEST_END',source='36',pairs=len(pairs),domain_pairs=total,original_parent_witness_reselected=False)
    return pairs,negative,total

def prepare36():
    begin=time.perf_counter();dest=DEST/'workset_36';dest.mkdir();rt=runtime('36');pairs,negative,total=manifest36(rt,dest)
    records={r['row']:r for r in rt['records']};rows=np.unique(pairs[:,0]);blocks=dest/'blocks';blocks.mkdir()
    x=np.lib.format.open_memmap(dest/'features.npy',mode='w+',dtype=np.float32,shape=(len(pairs),605))
    base=np.lib.format.open_memmap(dest/'base_margins.npy',mode='w+',dtype=np.float32,shape=(len(pairs),4))
    bindings=dict(compiler_sha256=digest(__file__),feature_source_sha256=digest(old.__file__),
        input_receipt_sha256=digest(OUT/'run430/source_36/receipt.json'),
        sequence_receipt_sha256=digest(OUT/'run421/sequence56/source_36/receipt.json'),
        witness_sha256=digest(dest/'complete_parent_witnesses.json'),manifest_sha256=digest(dest/'indices_and_masks.npz'),
        original432_sha256=digest(OUT/'run432/pair_model.pt'),original429_sha256=digest(OUT/'run429/sequence_model.pt'))
    ledger=[];maximum=0.;reference_pairs=0;computed_pairs=0
    for start in range(0,len(rows),64):
        block_ids=[];event('FEATURE_BLOCK_START',source='36',carrier_start=start)
        for position,row in enumerate(rows[start:start+64],start):
            ids=np.flatnonzero(pairs[:,0]==row);r=records[int(row)];ii,jj=pairs[ids,1],pairs[ids,2]
            x[ids]=old.features(rt,r,ii,jj);values,control=sparse(rt,r,ii,jj,position==0)
            base[ids]=values;maximum=max(maximum,control['maximum_original_forward_error'])
            reference_pairs+=control['reference_pairs'];computed_pairs+=len(ids);block_ids.extend(ids.tolist())
        ids=np.asarray(sorted(block_ids),np.int64);x.flush();base.flush()
        path=blocks/f'block_{start//64:04d}.npz'
        atomic_binary_write(path,lambda f:np.savez(f,indices=ids,features=x[ids],base=base[ids],negative=negative[ids]))
        with np.load(path) as saved:
            assert np.array_equal(saved['features'],x[ids]) and np.array_equal(saved['base'],base[ids])
            assert np.isfinite(saved['features']).all() and np.isfinite(saved['base']).all()
            assert np.array_equal(saved['negative'],negative[ids])
        ledger.append(dict(file=path.name,sha256=digest(path),carrier_start=start,carrier_stop=min(start+64,len(rows)),
            row_min=int(ids.min()),row_max=int(ids.max()),explicit_indices_in_block=True,rows=len(ids)))
        write(dest/'committed_blocks.json',dict(bindings=bindings,blocks=ledger,next_carrier=min(start+64,len(rows)),
            committed_rows=sum(b['rows'] for b in ledger),total_rows=len(pairs)))
        event('COMMIT',source='36',completed_carriers=min(start+64,len(rows)),total_carriers=len(rows),
            completed_rows=sum(b['rows'] for b in ledger));check_resources()
    assert computed_pairs==len(pairs) and np.all(x[:,604]==1) and np.isfinite(base).all()
    x._mmap.close();base._mmap.close();close(rt)
    write(dest/'receipt.json',dict(source='36',G=32,pairs=len(pairs),uniform_domain_pairs=total,per_draw_probability=1/total,
        samples_per_epoch=4096,passes=16,seed=1031,native_witness_sha256=bindings['witness_sha256'],
        features_sha256=digest(dest/'features.npy'),base_sha256=digest(dest/'base_margins.npy'),
        masks_sha256=digest(dest/'indices_and_masks.npz'),parents_sha256=digest(dest/'parents.json'),
        sparse_vs_original_max_abs=maximum,reference_control_pairs=reference_pairs,training_pairs_computed=computed_pairs,
        full_domain_base_expansion_for_workset=False,old_uncommitted_partial_not_reused=True,
        unknown_not_replenished_or_negative=True,unselected_native_aliases_not_negative=True,
        DEV_reads=0,STEP_reads=0,fit_calls=0,seconds=time.perf_counter()-begin))
    event('WORKSET36_COMPLETE',G=32,rows=len(pairs))

def prepare():
    protocol=read(DEST/'registered_protocol.json')
    for path,h in protocol['frozen_models'].items():assert digest(OUT/path)==h
    numerical_repair_control()
    for key in old.TRAIN[:-1]:reuse(key)
    prepare36()
    assert sum(read(DEST/f'workset_{k}/receipt.json')['G'] for k in old.TRAIN)==269
    write(DEST/'all_worksets_verified.json',dict(G=269,TRAIN=list(old.TRAIN),all_base_finite=True,fit_calls=0,
        workset_receipt_sha256={k:digest(DEST/f'workset_{k}/receipt.json') for k in old.TRAIN}))

def numerical_repair_control():
    models()
    import torch
    from weld_pipeline.sjg_joint_pair_model_1248 import tensors
    rawq=np.array([0.,1e-9,10.,500.,500.+1e-8,501.],np.float64);n=len(rawq)
    rng=np.random.default_rng(1293)
    norm=dict(mean=np.zeros(180,np.float32),scale=np.ones(180,np.float32),
        context_mean=np.zeros(160,np.float32),context_scale=np.ones(160,np.float32))
    s=dict(x=rng.normal(size=(2*n,180)).astype(np.float32),context=np.zeros((1,160),np.float32),m=dict(q_mm=rawq),prefix=rawq*.5)
    rt=dict(s=s,payload=dict(norm=norm),device=DEVICE,model=MODEL)
    ii,jj=[a.astype(np.int64) for a in np.triu_indices(n,1)];maximum=0.
    for closed in (False,True):
        r=dict(row=0,q_offset=0,feature_offset=0,n=n,curve_length_mm=501.,is_closed=closed)
        value,c=sparse(rt,r,ii,jj,True);x,ctx,q,p=tensors(s,r,norm,DEVICE)
        with torch.no_grad():
            h=MODEL.encode(x,ctx);full=MODEL.scores(h[:n],h[n:],q,p,501.,closed,0,n-1)
            a=torch.tensor(ii,dtype=torch.long,device=DEVICE);b=torch.tensor(jj,dtype=torch.long,device=DEVICE)
            selected=full[a,b];ref=(selected[:,1:]-selected[:,:1]).cpu().numpy()
        assert np.isfinite(value).all() and np.isfinite(ref).all()
        error=float(abs(value-ref).max());maximum=max(maximum,error);assert error<1e-4
    write(DEST/'numerical_repair_control.json',dict(status='PASS',original_model_frozen=True,
        all_four_direction_type_columns_checked=True,open_and_closed_span_scalar_checked=True,
        native_positive_float32_collapsed_and_sub_1e_6_spans_retained=True,
        corrected_sparse_vs_original_full_max_abs=maximum,original_tolerance_unchanged=1e-4,fit_calls=0))
    event('NUMERICAL_REPAIR_CONTROL_PASS',maximum_error=maximum)

if __name__=='__main__':prepare()
