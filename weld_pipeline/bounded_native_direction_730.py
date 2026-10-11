"""Preserve descending AdamW blocks; cap replacements for ascending blocks.

This constructs a proposal only. It never updates parameters, optimizer state,
or RNG. The caller must check actual displacement and objective decrease.
"""
import math
import torch

VERSION='NATIVE_DESCENT_UNCHANGED_CLIPPED_PRECONDITIONED_NORM_CAP_V1'


def flat(values):
    return torch.cat([v.detach().double().reshape(-1) for v in values])


def prepare_bounded_blocks(optimizer, parameters, raw_gradient):
    groups={id(p):group for group in optimizer.param_groups for p in group['params']}
    assert len(groups)==sum(len(g['params']) for g in optimizer.param_groups)
    assert raw_gradient.ndim==1 and raw_gradient.numel()==sum(p.numel() for p in parameters)
    assert bool(torch.isfinite(raw_gradient).all())
    blocks={};offset=0
    for parameter in parameters:
        group=groups[id(parameter)]
        assert not group.get('maximize',False)
        row=blocks.setdefault(group['phase_name'],dict(indices=[],preconditioned_clipped_gradient=[]))
        size=parameter.numel()
        if parameter.grad is None:
            pg=torch.zeros_like(parameter,dtype=torch.float64)
        else:
            clipped=parameter.grad.detach()
            assert bool(torch.isfinite(clipped).all())
            state=optimizer.state.get(parameter,{})
            beta2=group['betas'][1]
            step=float(state['step'])+1. if state else 1.
            variance=state['exp_avg_sq'] if state else torch.zeros_like(parameter)
            # Same clipped gradient is used in numerator and next variance.
            next_variance=variance.mul(beta2).addcmul(clipped,clipped,value=1.-beta2)
            if group.get('amsgrad',False):
                next_variance=torch.maximum(state.get('max_exp_avg_sq',torch.zeros_like(parameter)),next_variance)
            denominator=next_variance.double().sqrt()/math.sqrt(1.-beta2**step)+group['eps']
            pg=clipped.double()*group['lr']/denominator
            assert bool(torch.isfinite(pg).all())
        row['indices'].extend(range(offset,offset+size))
        row['preconditioned_clipped_gradient'].append(pg.reshape(-1))
        offset+=size
    for row in blocks.values():
        row['indices']=torch.tensor(row['indices'],dtype=torch.long,device=raw_gradient.device)
        row['preconditioned_clipped_gradient']=torch.cat(row['preconditioned_clipped_gradient'])
    return blocks


def bounded_direction(proposal, raw_gradient, blocks):
    assert proposal.shape==raw_gradient.shape and proposal.ndim==1
    assert bool(torch.isfinite(proposal).all()) and bool(torch.isfinite(raw_gradient).all())
    result=proposal.clone();facts={};covered=[]
    for name,row in blocks.items():
        index=row['indices'];covered.extend(index.cpu().tolist())
        gradient=raw_gradient[index];native=proposal[index]
        pg=row['preconditioned_clipped_gradient']
        assert pg.shape==native.shape and bool(torch.isfinite(pg).all())
        old_dot=float(gradient@native)
        native_norm=float(torch.linalg.vector_norm(native))
        pg_norm=float(torch.linalg.vector_norm(pg))
        if old_dot<0.:
            mode='NATIVE_DESCENDING_UNCHANGED';scale=None
            assert torch.equal(result[index],native)
        elif pg_norm==0. or native_norm==0.:
            mode='ZERO_DIRECTION_NO_MANUFACTURED_UPDATE';scale=0.
            result[index]=torch.zeros_like(native)
        else:
            mode='BOUNDED_PRECONDITIONED_CLIPPED_GRADIENT'
            assert float(gradient@pg)>0., 'Nonzero preconditioner must align with the raw gradient'
            scale=min(1.,native_norm/pg_norm)
            result[index]=-scale*pg
        final_dot=float(gradient@result[index])
        final_norm=float(torch.linalg.vector_norm(result[index]))
        assert final_dot<=0. and final_norm<=native_norm*(1.+1e-12)
        facts[name]=dict(mode=mode,g_dot_native=old_dot,g_dot_bounded=final_dot,
            native_norm=native_norm,bounded_norm=final_norm,preconditioned_clipped_norm=pg_norm,scale=scale)
    assert sorted(covered)==list(range(proposal.numel())), 'Blocks must partition the actual proposal once'
    return result,dict(version=VERSION,blocks=facts,g_dot_total=float(raw_gradient@result),
        native_descending_blocks_not_amplified=True,optimizer_state_updated=False)
