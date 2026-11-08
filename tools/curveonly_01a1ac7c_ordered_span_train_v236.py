"""TRAIN-only ordered span learning; original complete-instance labels are preserved."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_sequence_train_v172 as configuration
from weld_pipeline import curveonly_01a1ac7c_ordered_spans_v235 as m

SETTINGS=dict(seed=20261106,epochs=128,learning_rate=.003,final_learning_rate=.0003,
    weight_decay=.0001,gradient_norm_limit=5.,CPU_threads=2,GPU_memory_fraction=.35,
    batch_size=m.base.BATCH_SIZE,channels=m.CHANNELS,observations=111,
    continuation_loss_weight=1.,pointer_loss_weight=2.,offset_loss_weight=1.,full_cycle_loss_weight=1.,
    offset_smooth_beta=.1,normalizer='sameTRAIN_midrange_half_range_no_clipping',
    supervision='original169literal_alias_ranges_in_order_with_continue_and_stop',
    fixed_count_classes=False,minimum_weld_or_gap_mm=None,foreground_event_threshold=False)


def configure():
    configuration.configure(SETTINGS)


def assemble(cases):
    rows,details=[],[]
    masses=dict(classification=float(len(cases)),pointer=0.,full=0.)
    for case in sorted(cases,key=lambda q:q['source_id']):
        assert case['split']=='TRAIN'
        for key in ('features','geometry','local_targets'):
            assert e.sha(case[key]['path'])==case[key]['sha256']
        packet=e.read(case['geometry']['path'])
        current=m.candidate_rows(np.load(case['features']['path']),packet)
        representations=Counter(q['truth'] for q in case['chosen'])
        truths=len(representations)
        counts=Counter(bool(any(q['candidate']==row['candidate'] for q in case['chosen'])) for row in current)
        for row in current:
            chosen=[q for q in case['chosen'] if q['candidate']==row['candidate']]
            forward=m.sequence(row,chosen);reverse=m.sequence(row,chosen,True)
            row.update(source_id=case['source_id'],sequence=forward,reverse_sequence=reverse,
                stop_weight=1/(len(counts)*counts[bool(chosen)]),
                continue_factor=1/(1+bool(truths)),
                instance_weights={q['truth']:1/(truths*representations[q['truth']]) for q in chosen})
            for item in forward:
                weight=row['instance_weights'][item['truth']]
                if not item['full']:masses['pointer']+=weight
            if row['closed'] and forward:
                masses['full']+=sum(row['instance_weights'][q['truth']] for q in forward)
        positive_mass=sum(sum(q['instance_weights'].values()) for q in current)
        assert abs(positive_mass-(1. if truths else 0.))<1e-8
        assert abs(sum(q['stop_weight'] for q in current)-1)<1e-8
        rows.extend(current)
        details.append(dict(source_id=case['source_id'],candidates=len(current),
            complete_true_instances=truths,represented_intervals=len(case['chosen']),
            positive_instance_mass=positive_mass,maximum_sequence=max(map(lambda q:len(q['sequence']),current)),
            all_original_choices_retained=True,known_true_welds_not_merged_or_removed=True))
    mean,scale=m.codec.previous.fit_range_normalizer(rows)
    assert max(float(np.abs((q['grid']-mean)/scale).max()) for q in rows)<=1.00001
    return rows,mean,scale,masses,details


def loss_terms(output,batch):
    cont=(F.binary_cross_entropy_with_logits(output['continue_logit'],batch['continue_target'],reduction='none')*batch['continue_weight']).sum()
    weight=batch['span_weight']
    start=F.cross_entropy(output['start'].transpose(1,2),batch['starts'],reduction='none')
    end=F.cross_entropy(output['end'].transpose(1,2),batch['ends'],reduction='none')
    pointer=((start+end)*.5*weight).sum()
    offset=(F.smooth_l1_loss(output['offset'],batch['offsets'],reduction='none',beta=SETTINGS['offset_smooth_beta']).mean(2)*weight).sum()
    full=(F.binary_cross_entropy_with_logits(output['full_logit'],batch['full_target'],reduction='none')*batch['full_weight']).sum()
    return cont,pointer,offset,full


def objective(terms,masses):
    cont,pointer,offset,full=terms
    return (SETTINGS['continuation_loss_weight']*cont/masses['classification']+
        SETTINGS['pointer_loss_weight']*pointer/max(1.,masses['pointer'])+
        SETTINGS['offset_loss_weight']*offset/max(1.,masses['pointer'])+
        SETTINGS['full_cycle_loss_weight']*full/max(1.,masses['full']))


def assembly_control(cases,destination):
    configure();rows,mean,scale,masses,details=assemble(cases)
    ids=[]
    predicates=[lambda q:bool(q['sequence']),
        lambda q:q['closed'] and any(x['full'] for x in q['sequence']),
        lambda q:q['closed'] and len(q['sequence'])>1,
        lambda q:not q['sequence'],
        lambda q:not q['closed'] and len(q['sequence'])==1]
    for predicate in predicates:
        eligible=[i for i,row in enumerate(rows) if predicate(row)]
        if eligible:
            chosen=max(eligible,key=lambda i:(len(rows[i]['sequence']),len(rows[i]['grid'])))
            if chosen not in ids:ids.append(chosen)
    network=m.OrderedSpanNetwork().cuda()
    loss=0.
    for reverse in (False,True):
        batch=m.pack([rows[i] for i in ids],mean,scale,'cuda',reverse)
        output=network.teacher(batch);loss=loss+.5*objective(loss_terms(output,batch),masses)
        selected=batch['span_weight']>0
        if bool(selected.any()):
            assert bool((output['start'].gather(2,batch['starts'][:,:,None])[:,:,0][selected]>-1e8).all())
            assert bool((output['end'].gather(2,batch['ends'][:,:,None])[:,:,0][selected]>-1e8).all())
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(network.parameters(),5.,error_if_nonfinite=True)
    assert torch.isfinite(loss).item() and torch.isfinite(norm).item()
    e.write(Path(destination)/'ASSEMBLY_CONTROL.json',dict(passed=True,
        actual_GPU_finite_loss_and_backward=True,all_teacher_pointers_unmasked=True,
        parameter_count=sum(q.numel() for q in network.parameters()),masses=masses,details=details,
        no_optimizer_step_or_saved_random_weights=True))


def fit(cases,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    configure();start=time.perf_counter()
    rows,mean,scale,masses,details=assemble(cases)
    network=m.OrderedSpanNetwork().cuda().train()
    optimizer=torch.optim.AdamW(network.parameters(),lr=SETTINGS['learning_rate'],weight_decay=SETTINGS['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=SETTINGS['epochs'],eta_min=SETTINGS['final_learning_rate'])
    losses=[];first=0;latest=destination/'LATEST_CHECKPOINT.pt'
    if latest.exists():
        checkpoint=torch.load(latest,map_location='cuda',weights_only=False)
        assert checkpoint['settings']==SETTINGS and checkpoint['source_ids']==sorted(q['source_id'] for q in cases)
        assert np.array_equal(mean,checkpoint['mean']) and np.array_equal(scale,checkpoint['scale'])
        network.load_state_dict(checkpoint['network']);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        losses,first=checkpoint['losses'],checkpoint['epoch']
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first,SETTINGS['epochs']):
        begin=time.perf_counter();totals=np.zeros(4)
        groups=m.base.batches(rows,np.random.default_rng(SETTINGS['seed']+epoch))
        for ids in groups:
            selected=[rows[i] for i in ids];optimizer.zero_grad(set_to_none=True)
            value=0.
            for reverse in (False,True):
                batch=m.pack(selected,mean,scale,'cuda',reverse)
                output=network.teacher(batch);terms=loss_terms(output,batch)
                value=value+.5*len(groups)*objective(terms,masses)
                totals+=.5*np.array([float(q.detach()) for q in terms])
            assert torch.isfinite(value).item()
            value.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),SETTINGS['gradient_norm_limit'],error_if_nonfinite=True)
            optimizer.step()
        scheduler.step();torch.cuda.synchronize()
        record=dict(epoch=epoch+1,continuation_loss=float(totals[0]/masses['classification']),
            pointer_loss=float(totals[1]/max(1.,masses['pointer'])),offset_loss=float(totals[2]/max(1.,masses['pointer'])),
            full_cycle_loss=float(totals[3]/max(1.,masses['full'])),seconds=time.perf_counter()-begin,
            learning_rate=float(scheduler.get_last_lr()[0]))
        losses.append(record);e.write(destination/'TRAIN_PROGRESS.json',dict(settings=SETTINGS,epochs=losses,
            teacher_forced_losses_not_published_complete_curve_scores=True))
        print('ORDERED_SPAN_EPOCH',record,flush=True)
        if (epoch+1)%16==0 or epoch+1==SETTINGS['epochs']:
            path=destination/f'epoch_{epoch+1:04d}.pt'
            torch.save(dict(network=network.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                epoch=epoch+1,settings=SETTINGS,source_ids=sorted(q['source_id'] for q in cases),mean=mean,scale=scale,losses=losses),path)
            link=destination/'LATEST_CHECKPOINT.link'
            if link.is_symlink():link.unlink()
            link.symlink_to(path.name);os.replace(link,latest)
    peak=int(torch.cuda.max_memory_allocated());network.cpu().eval()
    torch.save(dict(network=network.state_dict(),mean=mean,scale=scale,settings=SETTINGS),destination/'weights.pt')
    restored=m.OrderedSpanNetwork()
    restored.load_state_dict(torch.load(destination/'weights.pt',map_location='cpu',weights_only=False)['network'])
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in network.state_dict().items())
    result=dict(weights=e.bind(destination/'weights.pt'),TRAIN_inputs=len(cases),cases=details,
        settings=SETTINGS,masses=masses,parameter_count=sum(q.numel() for q in network.parameters()),
        GPU_peak_allocated_bytes=peak,elapsed_seconds=time.perf_counter()-start,
        all_saved_parameters_CPU_reload_bit_exact=True,epochs=losses,
        all_original_ranges_and_unobserved_truths_retained=True,no_previous_weights_or_VALfit=True)
    e.write(destination/'FIT.json',result)
    print('ORDERED_SPAN_FIT_CLOSED',result['parameter_count'],result['elapsed_seconds'],flush=True)
    return network,mean,scale,result
