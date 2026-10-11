"""Structured interval-set hinge training; labels confined to TRAIN paths."""
import argparse
from collections import Counter
import gc
import time
import traceback
import numpy as np
import torch
from weld_pipeline import sjg_joint_pair_model_1248 as base
from weld_pipeline.sjg_interval_set_targets_1249 import gold_set
from weld_pipeline.sjg_density_feature_runtime_1277 import DensityView,overlay_endpoint,ENDPOINT_COLUMNS
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,reclaim_own_idle_pages,resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write

EPOCHS=12
BATCH=64
INPUT_RUN='run402'



def density_tensors(source,record,norm,device,view=None):
    if view is None:return base.tensors(source,record,norm,device)
    a=record['q_offset'];n=record['n'];f=record['feature_offset']
    raw=np.array(source['x'][f:f+2*n],copy=True)
    q=source['m']['q_mm'][a:a+n];prefix=source['prefix'][a:a+n]
    alternate=view.endpoint(record['row'],q,record['curve_length_mm'],record['is_closed'])
    raw=(overlay_endpoint(raw,alternate)-norm['mean'])/norm['scale']
    context=(np.array(source['context'][record['row']],copy=True)-norm['context_mean'])/norm['context_scale']
    return [torch.as_tensor(z,dtype=torch.float32,device=device) for z in (raw,context,q,prefix)]


def density_tensor_controls(records,sources,norm,views,dest):
    reports=[];retained=np.setdiff1d(np.arange(180),ENDPOINT_COLUMNS)
    for key,view in views.items():
        r=next(v for v in records if v['source']==key and v['n']>=2)
        original=base.tensors(sources[key],r,norm,'cpu')
        clean=density_tensors(sources[key],r,norm,'cpu')
        augmented=density_tensors(sources[key],r,norm,'cpu',view)
        assert all(torch.equal(a,b) for a,b in zip(original,clean))
        assert torch.equal(original[0][:,retained],augmented[0][:,retained])
        assert all(torch.equal(a,b) for a,b in zip(original[1:],augmented[1:]))
        delta=float((original[0]-augmented[0]).abs().max());assert delta>0
        again=base.tensors(sources[key],r,norm,'cpu')
        assert all(torch.equal(a,b) for a,b in zip(original,again))
        reports.append(dict(source=key,normalized_max_difference=delta))
    write(dest/'density_tensor_controls.json',dict(status='PASS',sources=reports,
        original_route_exact=True,original_inputs_not_mutated=True,context_q_occupancy_prefix_unchanged=True,
        unchanged_endpoint_columns=135,changed_endpoint_columns=45,DEV_reads=0))


def training_resource_gate(dest,stage,buffered):
    # Preserve the original guard. Reclaim only this worker's idle pages when
    # necessary; no labels, model state, or other applications are discarded.
    if not buffered or resources()['available_physical_bytes']<1342177280:
        write(dest/f'memory_trim_{stage}.json',reclaim_own_idle_pages())
    return check_resources()


def structured_choices(scores,record):
    """Exact FP/FN loss-augmented MAP and best complete legal gold set."""
    n=len(scores);groups=record['gold_groups'];fp=record['false_positive_cost']
    augmented=scores.copy()+fp
    for group in groups:
        pairs=np.asarray(group['pairs'],dtype=np.int64)
        for category in group.get('allowed_categories',[group['category']]):
            augmented[pairs//n,pairs%n,category-1]-=fp+group['miss_cost']
    categories=augmented.argmax(2)+1;margins=augmented.max(2)
    predicted=base.select_intervals(margins,categories)
    gold,gold_value=gold_set(groups,scores,n)
    value=sum(v for i,j,c,v in predicted)+sum(v['miss_cost'] for v in groups)-gold_value
    assert value>=-2e-3,(value,predicted,gold)
    return predicted,gold,max(0.,value)


def selected_scores(model,h,q,prefix,length,closed,edges):
    """Differentiable selected-edge scores without rebuilding the full matrix."""
    n=len(q);device=h.device
    i=torch.tensor([v[0] for v in edges],device=device);j=torch.tensor([v[1] for v in edges],device=device)
    c=torch.tensor([v[2] for v in edges],device=device)
    left=h[i];right=h[n+j]
    score=torch.einsum('pk,ckl,pl->pc',left,model.pair,right)/np.sqrt(32)
    score=score+model.unary(left)+model.unary(right)
    span=(q[j]-q[i]).clamp_min(0)
    geometry=torch.stack((torch.log1p(span)/10,span/length,(prefix[j]-prefix[i])/span.clamp_min(1e-6),
        torch.full_like(span,float(closed))),dim=-1)
    score=score+model.span(geometry)
    return score[torch.arange(len(edges),device=device),c]-score[:,0]


def controls(model,dest):
    n=5
    record=dict(gold_groups=[dict(pairs=[1,2],category=1,miss_cost=.6),
        dict(pairs=[2*n+4,3*n+4],category=2,miss_cost=.4)],false_positive_cost=.5)
    def enumerate_sets(start):
        yield []
        for i in range(start,n-1):
            for j in range(i+1,n):
                for c in (1,2):
                    for tail in enumerate_sets(j):yield [(i,j,c)]+tail
    sets=list(enumerate_sets(0));rng=np.random.default_rng(710);maximum=0.
    lookup={(pair//n,pair%n,g['category']):k for k,g in enumerate(record['gold_groups']) for pair in g['pairs']}
    for _ in range(8):
        scores=rng.normal(size=(n,n,2)).astype(np.float32)
        pred,gold,hinge=structured_choices(scores,record);best_aug=-np.inf;best_gold=-np.inf
        for edges in sets:
            owners=[lookup[e] for e in edges if e in lookup]
            assert len(set(owners))==len(owners)
            score=sum(float(scores[i,j,c-1]) for i,j,c in edges)
            cost=.5*sum(e not in lookup for e in edges)+sum(g['miss_cost'] for k,g in enumerate(record['gold_groups']) if k not in owners)
            best_aug=max(best_aug,score+cost)
            if len(edges)==2 and set(owners)=={0,1}:best_gold=max(best_gold,score)
        error=abs(hinge-(best_aug-best_gold));maximum=max(maximum,error);assert error<2e-5
    torch.manual_seed(104);xx=torch.randn(14,180);context=torch.randn(160)
    q=torch.arange(7,dtype=torch.float32);prefix=q*.5;edges=[(0,3,1),(2,5,2),(5,6,3)]
    coefficients=torch.tensor([1.,-1.,2.])
    h=model.encode(xx,context);all_scores=model.scores(h[:7],h[7:],q,prefix,6,False,0,7)
    full=torch.stack([all_scores[i,j,c]-all_scores[i,j,0] for i,j,c in edges])
    model.zero_grad(set_to_none=True);(full*coefficients).sum().backward()
    grads=[v.grad.clone() for v in model.parameters()]
    model.zero_grad(set_to_none=True);h=model.encode(xx,context)
    sparse=selected_scores(model,h,q,prefix,6,False,edges);(sparse*coefficients).sum().backward()
    error=max(float((v.grad-g).abs().max()) for v,g in zip(model.parameters(),grads));assert error<1e-5
    assert torch.allclose(full.detach(),sparse.detach(),atol=1e-5,rtol=0)
    model.zero_grad(set_to_none=True)
    write(dest/'structured_controls.json',dict(loss_augmented_and_gold_DP_match_exhaustive=True,
        exhaustive_hinge_max_abs=maximum,selected_edge_gradient_max_abs=error,
        gold_adjacent_segments_supported=True,GT_reads=0))


def train(dest,resume=False,supervision_run='run404',input_run=INPUT_RUN,
          input_routes=None,expected_g=213,initial_run='run399',buffer_inputs=False,initialization_file=None):
    from weld_pipeline.sjg_supervision_dependency_guard_1269 import require_clean_occupancy_supervision
    require_clean_occupancy_supervision(
        ['run392'] if input_routes is None else [v.get('occupancy_run','run392') for v in input_routes.values()])
    if (OUT/supervision_run/'confirmed_supervision_conflicts.json').exists():
        raise RuntimeError('run404 has confirmed native-D3 positive omissions; rebuild compatibility before fitting or resuming')
    start=time.perf_counter();torch.set_num_threads(2);torch.backends.cudnn.enabled=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(1031)
    if initialization_file is None:
        initial_path=OUT/initial_run/'pair_model.pt';initial_receipt=read(OUT/initial_run/'training_receipt.json')
        initialization_kind='PREVIOUS_TRAINED_MODEL'
    else:
        initial_path=OUT/initialization_file;initial_receipt=read(initial_path.parent/'initialization_receipt.json')
        assert initial_receipt['status']=='UNTRAINED_INITIALIZATION_CURRENT_TRAIN_NORMALIZATION'
        assert initial_receipt['fit_calls']==0 and initial_receipt['DEV_reads']==0
        initialization_kind='FRESH_WEIGHTS_CURRENT_TRAIN_NORMALIZATION'
    assert digest(initial_path)==initial_receipt['model_sha256']
    payload=torch.load(initial_path,map_location='cpu');norm=payload['norm']
    model=base.PairModel(payload['classes']);model.load_state_dict(payload['state_dict']);controls(model,dest)
    source_receipt=read(OUT/supervision_run/'receipt.json');assert source_receipt['G']==expected_g
    assert source_receipt['status']=='COMPLETE_ORDERED_TRAIN_INTERVAL_SETS_READY'
    train_keys=[c['source'] for c in source_receipt['cases']]
    assert len(train_keys)==len(set(train_keys)) and '64' not in train_keys
    if input_routes is not None:assert set(input_routes)==set(train_keys)
    records=[];sources={};bindings=[];pairs_per_epoch=0
    density_views={key:DensityView(key) for key in train_keys}
    for c in source_receipt['cases']:
        key=c['source'];f=OUT/f'{supervision_run}/source_{key}_set_targets.json';assert digest(f)==c['target_sha256']
        data=read(f);route={} if input_routes is None else input_routes[key]
        if buffer_inputs:
            for record in data['records']:
                for group in record['gold_groups']:
                    old_pairs=group['pairs'];packed=np.asarray(old_pairs,dtype=np.int64)
                    assert packed.tolist()==old_pairs
                    group['pairs']=packed
        s=base.load_inputs(key,input_run,**route)
        if buffer_inputs:
            from weld_pipeline.sjg_lossless_training_buffers_1268 import LosslessRows,controls as buffer_controls
            write(dest/'lossless_buffer_controls.json',buffer_controls())
            buffer_receipts=[]
            for field in ('x','context'):
                mapped=s[field];buffered=LosslessRows(mapped)
                assert buffered.shape==mapped.shape and buffered.dtype==mapped.dtype
                for offset in range(0,len(mapped),8192):
                    assert np.array_equal(buffered[offset:offset+8192],mapped[offset:offset+8192])
                buffer_receipts.append(dict(field=field,raw_bytes=buffered.nbytes,stored_bytes=buffered.stored_bytes,
                    raw_sha256=buffered.raw_sha256,all_blocks_and_slices_exact=True))
                buffered.cache.clear()
                mapped._mmap.close();s[field]=buffered
            write(dest/f'lossless_buffer_{key}.json',dict(source=key,arrays=buffer_receipts))
            print('TRAIN_INPUT_BUFFERED',key,'RAW_BYTES',s['x'].nbytes+s['context'].nbytes,
                'STORED_BYTES',s['x'].stored_bytes+s['context'].stored_bytes,flush=True)
        assert s['receipt']['feature_sha256']==data['feature_sha256'] and s['receipt']['map_sha256']==data['map_sha256']
        assert data['full_gold_sets_feasible'] and data['no_disjoint_same_owner_alternatives'] and data['no_pair_multi_owner']
        sources[key]=s;records.extend(data['records']);pairs_per_epoch+=data['all_pairs']
        item=dict(source=key,set_targets_sha256=digest(f),features_sha256=data['feature_sha256'],
            context_sha256=s['context_sha256'],G=data['original_G'])
        if input_routes is not None:item.update(input_run=input_run,input_route=route)
        item['density_view_sha256']=density_views[key].receipt['alternate_sha256']
        item['density_map_sha256']=density_views[key].receipt['map_sha256']
        bindings.append(item)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0003);rng=np.random.default_rng(1031)
    trace=[];first=1;elapsed=0.;updates=0
    if resume:
        state=torch.load(dest/'latest_training_state.pt',map_location=device)
        assert state.get('supervision_run','run404')==supervision_run
        assert state.get('input_bindings',bindings)==bindings
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['rng'];trace=state['trace'];first=state['epoch']+1
        elapsed=trace[-1]['seconds'];updates=state['updates']
        assert all(np.array_equal(norm[k],state['norm'][k]) for k in norm)
        write(dest/'resume_receipt.json',dict(checkpoint_sha256=digest(dest/'latest_training_state.pt'),next_epoch=first,
            optimizer_and_rng_restored=True))
    density_tensor_controls(records,sources,norm,density_views,dest)
    training_resource_gate(dest,'before_epochs',buffer_inputs)
    print('STRUCTURED_SET_TRAIN',len(records),'CARRIERS',pairs_per_epoch,'PAIRS_PER_EPOCH',device,flush=True)
    for epoch in range(first,EPOCHS+1):
        ep=time.perf_counter();order=rng.permutation(len(records));total_loss=0.;pair_count=0;active_rows=0;half_rows=0
        for base_index in range(0,len(order),BATCH):
            ids=order[base_index:base_index+BATCH];optimizer.zero_grad(set_to_none=True);batch_active=False
            for ix in ids:
                r=records[ix];n=r['n'];pair_count+=n*(n-1)//2
                if n<2:continue
                use_half=bool(rng.integers(0,2));half_rows+=int(use_half)
                x,c,q,p=density_tensors(sources[r['source']],r,norm,device,density_views[r['source']] if use_half else None)
                scores=np.zeros((n,n,len(payload['classes'])-1),np.float32)
                with torch.no_grad():
                    h=model.encode(x,c)
                    for lo in range(0,n-1,base.CHUNK):
                        hi=min(lo+base.CHUNK,n-1)
                        all_scores=model.scores(h[:n],h[n:],q,p,r['curve_length_mm'],r['is_closed'],lo,hi)
                        scores[lo:hi]=(all_scores[:,:,1:]-all_scores[:,:,:1]).cpu().numpy()
                predicted,gold,value=structured_choices(scores,r)
                total_loss+=value*r['row_weight']
                if value>1e-6:
                    counter=Counter((i,j,cat) for i,j,cat,v in predicted)
                    counter.subtract((i,j,cat) for i,j,cat,v in gold)
                    edges=[e for e,v in counter.items() if v]
                    if edges:
                        h=model.encode(x,c)
                        values=selected_scores(model,h,q,p,r['curve_length_mm'],r['is_closed'],edges)
                        coefficients=torch.tensor([counter[e] for e in edges],dtype=torch.float32,device=device)
                        objective=(values*coefficients).sum()*r['row_weight']*len(records)/len(ids)
                        assert torch.isfinite(objective);objective.backward();active_rows+=1;batch_active=True
                        del values,coefficients,objective
                del x,c,q,p,h,scores,all_scores
            if batch_active:
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(gradient)
                optimizer.step();updates+=1
            if buffer_inputs and (base_index//BATCH+1)%16==0:
                print('EPOCH_BATCH',epoch,'/',EPOCHS,'ROWS',min(base_index+BATCH,len(order)),'/',len(order),flush=True)
        assert pair_count==pairs_per_epoch
        trace.append(dict(epoch=epoch,structured_hinge=total_loss,active_rows=active_rows,all_pairs_seen=pair_count,density_half_rows=half_rows,
            epoch_seconds=time.perf_counter()-ep,seconds=elapsed+time.perf_counter()-start,updates=updates))
        atomic_binary_write(dest/'latest_training_state.pt',lambda f:torch.save(dict(state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,trace=trace,epoch=epoch,norm=norm,updates=updates,
            supervision_run=supervision_run,input_bindings=bindings),f))
        write(dest/'training_progress.json',dict(trace=trace,total_epochs=EPOCHS))
        print('EPOCH',epoch,'/',EPOCHS,'SET_HINGE',round(total_loss,6),'ACTIVE_ROWS',active_rows,
            'PAIRS',pair_count,'SECONDS',round(trace[-1]['epoch_seconds'],2),flush=True)
        gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        training_resource_gate(dest,f'epoch_{epoch}',buffer_inputs)
    model.cpu().eval();controls(model,dest)
    payload.update(state_dict=model.state_dict(),input_run=input_run,epochs=EPOCHS,
        objective='LOSS_AUGMENTED_ORDERED_INTERVAL_SET_HINGE',initial_model_sha256=initial_receipt['model_sha256'],
        supervision_run=supervision_run)
    atomic_binary_write(dest/'pair_model.pt',lambda f:torch.save(payload,f))
    reload=base.PairModel(payload['classes']);reload.load_state_dict(torch.load(dest/'pair_model.pt')['state_dict'])
    assert all(torch.equal(v,reload.state_dict()[k]) for k,v in model.state_dict().items())
    write(dest/'training_receipt.json',dict(status='STRUCTURED_INTERVAL_SET_MODEL_TRAINED',TRAIN=train_keys,G=expected_g,
        epochs=EPOCHS,trace=trace,input_bindings=bindings,model_sha256=digest(dest/'pair_model.pt'),
        initial_model_sha256=initial_receipt['model_sha256'],fresh_optimizer_for_changed_objective=True,
        input_run=input_run,input_routes=input_routes,initial_run=initial_run,initialization_file=initialization_file,
        initialization_kind=initialization_kind,supervision_run=supervision_run,buffer_inputs=buffer_inputs,reload_exact=True,DEV_reads=0,STEP_reads=0,endpoint_gate_used=False,
        density_augmentation_run='run436',augmentation_rng_checkpointed=True,inference_original_features_only=True,
        explicit_material_role_labels_available=False,full_contract_pass=False,seconds=elapsed+time.perf_counter()-start))
    print('STRUCTURED_SET_TRAIN_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--stage',choices=['train','infer'],required=True)
    p.add_argument('--source');p.add_argument('--resume',action='store_true');a=p.parse_args();d=OUT/a.run
    if a.stage=='train' and not a.resume:
        d.mkdir(exist_ok=False)
        write(d/'registered_protocol.json',dict(source_sha256=digest(__file__),base_source_sha256=digest(base.__file__),
            input_run=INPUT_RUN,set_targets_run='run404',initial_model_run='run399',TRAIN=list(base.TRAIN),DEV=['64'],
            epochs=EPOCHS,batch=BATCH,optimizer='fresh Adam',learning_rate=.0003,seed=1031,
            all_pairs_each_epoch=True,final_epoch_only=True,preserve_best379=True,external_consultation_paused=True,
            whole_parent_assembly_still_required=True,full_contract_pass=False))
    try:
        train(d,a.resume) if a.stage=='train' else base.infer(d,a.source)
    except BaseException:
        write(d/f'{a.stage}_{a.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
