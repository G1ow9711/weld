"""Optimize full reference-set regret on fixed original and new parents."""
from pathlib import Path
import time
import numpy as np
import torch
from weld_pipeline.server_structured_parent_residual import StructuredParentResidual, predict_scores, score_difference
from weld_pipeline.server_offline_parent_set_comparator import reference_comparator
from weld_pipeline.server_offline_parent_set_reference import evaluate_reference_subset
from weld_pipeline.server_reference_parent_selection import greedy_nullable_set, set_model_utility
from weld_pipeline.sjg_local_runtime_904 import read, write, digest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1'
PREVIOUS=BASE/'reference_parent_selection_v1';DEST=BASE/'structured_parent_residual_v1'
STEPS=330


def moments(data, name):
    arrays=[row[name] for row in data if len(row[name])]
    mean=np.mean([np.mean(x.astype(np.float64),axis=0) for x in arrays],axis=0)
    second=np.mean([np.mean(x.astype(np.float64)**2,axis=0) for x in arrays],axis=0)
    return mean.astype(np.float32), np.maximum(np.sqrt(np.maximum(second-mean**2,0.)),1e-3).astype(np.float32)


def main():
    started=time.perf_counter();torch.manual_seed(1031);torch.set_num_threads(4);rng=np.random.default_rng(1031)
    dataset=read(PREVIOUS/'train_dataset/receipt.json');assert dataset['status']=='READY'
    assert read(PREVIOUS/'train_outputs/summary.json')['status']=='EVALUATED_NOT_PROMOTED'
    DEST.mkdir(exist_ok=False)
    modules=['server_structured_parent_residual','server_offline_parent_set_comparator','server_reference_parent_selection','server_offline_parent_set_reference']
    protocol=dict(steps=STEPS,seed=1031,learning_rate=3e-4,final_learning_rate=3e-5,weight_decay=1e-4,
          regularization=.01,case_sampling='30 shuffled complete passes over eleven training families',
          input_receipt_sha256=digest(PREVIOUS/'train_dataset/receipt.json'),base_model_sha256=digest(PREVIOUS/'model_v1/model.pkl'),
          base_training_outputs_sha256=digest(PREVIOUS/'train_outputs/summary.json'),
          loss='relu(exact_D3_U(teacher)-exact_D3_U(predicted)+model_score(predicted)-model_score(teacher)) / reference_child_count',
          teacher='Feasible matching proposal, verified using original D3; use only when exact reference utility improves',
          reference_utility='TP-FP under recorded annotation policy; not physical absence',
          candidate_pool_unchanged=True,graph_unchanged=True,baseline_zero_residual_required=True,
          full_parent_preserved=True,conditional_upstream=True,DEV_reads=0,independent_acceptance=False,
          modules={n:digest(ROOT/'weld_pipeline'/f'{n}.py') for n in modules},source_sha256=digest(__file__))
    write(DEST/'protocol.json',protocol)
    data=[];bindings={}
    for row in dataset['cases']:
        for item in list(row['files'].values())+[row['reference_table']]:
            assert digest(ROOT/item['path'])==item['sha256']
        key=row['key'];frozen=PREVIOUS/'train_outputs'/key;fr=read(frozen/'prediction_freeze.json')
        assert digest(frozen/'model_scores.npz')==fr['scores_sha256']
        with np.load(frozen/'model_scores.npz') as z:baseu=z['unary'];basep=z['penalty']
        with np.load(ROOT/row['reference_table']['path']) as z:table={n:z[n] for n in z.files}
        x=np.load(ROOT/row['files']['features.npy']['path']);px=np.load(ROOT/row['files']['pair_features.npy']['path'])
        pairs=np.load(ROOT/row['files']['pairs.npy']['path']);counts=table['child_counts']
        pc=read(ROOT/row['files']['parent_children.json']['path'])
        evaluation=read(frozen/'evaluation.json');g=evaluation['scores']['soft_set']['categories']['overall']['G']
        data.append(dict(key=key,x=x,px=px,pairs=pairs,counts=counts,pc=pc,table=table,base_unary=baseu,base_penalty=basep,
                         pair_scale=2*np.minimum(counts[pairs[:,0]],counts[pairs[:,1]]),G=g,
                         baseline_indices=fr['arms']['soft_set']['selected_parent_indices']))
        bindings[key]=dict(dataset_case_sha256=digest(PREVIOUS/'train_dataset'/key/'receipt.json'),base_freeze_sha256=digest(frozen/'prediction_freeze.json'))
    pm,ps=moments(data,'x');em,es=moments(data,'px')
    model=StructuredParentResidual(pm,ps,em,es).cuda()
    # Real full candidate pools must recover the already evaluated base branch.
    for row in data:
        u,p=predict_scores(model,row['x'],row['px'],row['counts'],row['pairs'],row['base_unary'],row['base_penalty'])
        assert np.array_equal(u,row['base_unary']) and np.array_equal(p,row['base_penalty'])
        chosen=greedy_nullable_set(u,row['pairs'],p,row['pc'])
        assert chosen==row['baseline_indices']
        teacher,score=reference_comparator(row['table'],row['pc'],u)
        difference=score_difference(model,row,chosen,teacher)
        expected=set_model_utility(u,row['pairs'],p,chosen)-set_model_utility(u,row['pairs'],p,teacher)
        assert abs(difference.item()-expected)<1e-8*max(1.,abs(expected))
    write(DEST/'zero_control.json',dict(status='PASS',cases=len(data),score_arrays_bitwise_equal=True,
          selected_parents_identical=True,differentiable_score_matches_numpy=True,bindings=bindings))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,STEPS,eta_min=3e-5)
    history=[];order=[];updates=0
    for step in range(STEPS):
        if step%len(data)==0:order=rng.permutation(len(data)).tolist()
        row=data[order[step%len(data)]]
        model.eval();u,p=predict_scores(model,row['x'],row['px'],row['counts'],row['pairs'],row['base_unary'],row['base_penalty'])
        predicted=greedy_nullable_set(u,row['pairs'],p,row['pc'])
        actual=evaluate_reference_subset(row['table'],predicted)
        teacher,target=reference_comparator(row['table'],row['pc'],u)
        advantage=target['reference_utility']-actual['reference_utility']
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=torch.zeros((),device='cuda',dtype=torch.float64)
        if advantage>0:
            loss=torch.relu(score_difference(model,row,predicted,teacher)+advantage)/max(1,row['G'])
            updates+=1
        nodes=rng.choice(len(row['x']),min(128,len(row['x'])),replace=False)
        edges=rng.choice(len(row['px']),min(128,len(row['px'])),replace=False)
        regularizer=model.parent_residual(torch.as_tensor(row['x'][nodes],device='cuda')).square().mean()
        if len(edges):regularizer=regularizer+model.pair_residual(torch.as_tensor(row['px'][edges],device='cuda')).square().mean()
        total=loss+.01*regularizer;assert torch.isfinite(total)
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();scheduler.step()
        if step%11==0 or step==STEPS-1:
            record=dict(step=step,key=row['key'],loss=float(total.detach().cpu()),prediction=actual,comparator=target,seconds=time.perf_counter()-started)
            history.append(record);print(record,flush=True)
        if (step+1)%110==0 or step==STEPS-1:
            temporary=DEST/'last.pt.tmp';torch.save(dict(state_dict={name:value.detach().cpu() for name,value in model.state_dict().items()},step=step+1,protocol_sha256=digest(DEST/'protocol.json')),temporary)
            temporary.replace(DEST/'last.pt')
    write(DEST/'receipt.json',dict(status='TRAINED_NOT_PROMOTED',steps=STEPS,structured_comparisons=updates,
          checkpoint_sha256=digest(DEST/'last.pt'),protocol_sha256=digest(DEST/'protocol.json'),
          zero_control_sha256=digest(DEST/'zero_control.json'),history=history,bindings=bindings,
          seconds=time.perf_counter()-started,DEV_reads=0,original_run446_unchanged=True))
    print('STRUCTURED_PARENT_RESIDUAL_READY',flush=True)


if __name__=='__main__':main()
