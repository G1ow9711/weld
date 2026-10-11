"""Restore a completed Adam update; never score or accept intermediate weights."""
import os
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_parent_set_scorer_1159 import ParentSetScorer,AdamState,SHAPES
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS


def restore(source,model,optimizer,mean,scale,bindings,allow_completed_stage=False):
    receipt=read(source/'selector_resume_checkpoint.json')
    path=source/'selector_resume_checkpoint.npz'
    if digest(path)!=receipt['checkpoint_sha256']:raise ValueError('Checkpoint SHA mismatch')
    count=receipt['updates']
    maximum=receipt.get('registered_updates',112)
    if not isinstance(count,int) or not 0<count<=maximum or (count==maximum and not allow_completed_stage):
        raise ValueError('Invalid completed-update checkpoint position')
    if receipt['bindings']!=bindings:raise ValueError('Candidate or supervision binding changed')
    expected_epoch=count//len(KEYS)+1 if count<maximum else None
    expected_source=KEYS[count%len(KEYS)] if count<maximum else None
    if receipt['next_epoch']!=expected_epoch or receipt['next_source']!=expected_source:
        raise ValueError('Wrong next update position')
    if receipt['random_draws_after_initialization']!=0 or receipt['initialization_rng_state']!=model.initialization_rng_state:
        raise ValueError('RNG state or deterministic protocol mismatch')
    # Restore the saved generator state itself; no draw is used by resumed updates.
    model.resume_rng=np.random.Generator(np.random.PCG64(0))
    model.resume_rng.bit_generator.state=receipt['initialization_rng_state']
    assert model.resume_rng.bit_generator.state==receipt['initialization_rng_state']
    old_definition=source/'source_snapshot/sjg_parent_set_scorer_1159.py'
    from weld_pipeline.sjg_local_runtime_904 import ROOT
    if digest(old_definition)!=digest(ROOT/'weld_pipeline/sjg_parent_set_scorer_1159.py'):
        raise ValueError('Model or Adam definition changed since checkpoint')
    trace=receipt['trace']
    if len(trace)!=count:raise ValueError('Incomplete update trace')
    for i,row in enumerate(trace):
        if row['step']!=i+1 or row['epoch']!=i//len(KEYS)+1 or row['source']!=KEYS[i%len(KEYS)]:
            raise ValueError('Update order changed')
    with np.load(path,allow_pickle=False) as data:
        expected={f'{prefix}_{key}' for prefix in ('parameter','first','second') for key in SHAPES}|{'updates','mean','scale'}
        if set(data.files)!=expected or int(data['updates'])!=count:raise ValueError('Checkpoint fields/count mismatch')
        if not np.array_equal(data['mean'],mean) or not np.array_equal(data['scale'],scale):
            raise ValueError('Training normalization changed')
        for prefix,target in (('parameter',model.parameters),('first',optimizer.first),('second',optimizer.second)):
            for key,shape in SHAPES.items():
                value=data[f'{prefix}_{key}']
                if value.dtype!=np.float64 or value.shape!=shape or not np.isfinite(value).all():
                    raise ValueError('Invalid checkpoint array')
                if prefix=='second' and np.any(value<0):raise ValueError('Negative Adam second moment')
                target[key]=value.copy()
    optimizer.step_count=count
    return receipt


def control(source,dest):
    """Fixed artificial input and independent next-Adam arithmetic; no real update."""
    receipt=read(source/'selector_resume_checkpoint.json')
    with np.load(source/'selector_resume_checkpoint.npz',allow_pickle=False) as data:
        mean=data['mean'].copy();scale=data['scale'].copy()
        raw={key:data[key].copy() for key in data.files}
    model=ParentSetScorer(1031);optimizer=AdamState(model)
    restored=restore(source,model,optimizer,mean,scale,receipt['bindings'])
    for prefix,target in (('parameter',model.parameters),('first',optimizer.first),('second',optimizer.second)):
        for key in SHAPES:assert np.array_equal(target[key],raw[f'{prefix}_{key}'])
    reference=ParentSetScorer(1031)
    for key in SHAPES:reference.parameters[key]=raw[f'parameter_{key}'].copy()
    unary=np.linspace(-.7,.8,3*160).reshape(3,160);relations=np.zeros((3,32))
    assert np.array_equal(model.forward(unary,relations,[])[0],reference.forward(unary,relations,[])[0])
    gradients={key:np.linspace(-.2,.3,int(np.prod(shape))).reshape(shape) for key,shape in SHAPES.items()}
    norm=float(np.sqrt(sum(np.sum(value*value) for value in gradients.values())))
    multiplier=min(1.,5./max(norm,np.finfo(float).tiny));next_step=receipt['updates']+1
    expected={};first={};second={}
    for key in SHAPES:
        param=raw[f'parameter_{key}'].copy();grad=gradients[key]*multiplier+1e-4*param
        first[key]=.9*raw[f'first_{key}']+.1*grad
        second[key]=.999*raw[f'second_{key}']+.001*grad*grad
        expected[key]=param-1e-3*(first[key]/(1.-.9**next_step))/(np.sqrt(second[key]/(1.-.999**next_step))+1e-8)
    optimizer.update(model,gradients)
    assert optimizer.step_count==next_step
    for key in SHAPES:
        assert np.array_equal(model.parameters[key],expected[key])
        assert np.array_equal(optimizer.first[key],first[key]) and np.array_equal(optimizer.second[key],second[key])
    assert model.resume_rng.bit_generator.state==receipt['initialization_rng_state']
    result=dict(status='CHECKPOINT_RESTORE_AND_NEXT_ADAM_BITWISE_PASSED',completed_updates=receipt['updates'],
        next_epoch=restored['next_epoch'],next_source=restored['next_source'],
        checkpoint_sha256=receipt['checkpoint_sha256'],parameters_moments_normalization_bitwise_equal=True,
        fixed_forward_bitwise_equal=True,synthetic_next_adam_bitwise_equal=True,
        comparison_scope='independent checkpoint reload; no old process forward digest was saved',
        PCG64_state_restored_and_unchanged=True,
        actual_training_updates=0,GT_reads=0,synthetic_model_discarded=True)
    write(dest/'resume_control.json',result)
    print('RESUME_CONTROL',result,flush=True)


def reuse_columns(source,cache):
    copied=skipped=0
    for key,binding in cache.bindings.items():
        old=source/key;new=cache.root/key
        if read(old/'binding.json')!=binding:raise ValueError('Cached relation inputs changed')
        for path in sorted(old.glob('*.npy')):
            receipt=path.with_suffix('.json')
            if not receipt.exists():skipped+=1;continue
            meta=read(receipt)
            if not path.stem.isdigit() or digest(path)!=meta['sha256']:raise ValueError('Invalid cached column')
            needed=path.stat().st_size+receipt.stat().st_size
            if cache.disk_bytes+needed>cache.disk_limit:raise ValueError('Existing cache exceeds registered bound')
            os.link(path,new/path.name);os.link(receipt,new/receipt.name)
            cache.disk_bytes+=needed;copied+=1
    return dict(complete_columns_reused=copied,incomplete_columns_ignored=skipped,
        disk_bytes=cache.disk_bytes,hardlinked_immutable_columns=True)
