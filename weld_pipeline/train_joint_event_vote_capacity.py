"""Fixed train-source event-voting pilot, gated before any new held evaluation."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.labels import sha256
from weld_pipeline.contract_child_evaluation import aggregate,score_children
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.joint_event_votes import JointEventVotes,dense_targets,infer_votes,VERSION
from weld_pipeline.joint_relation_heatmaps import LOCAL_COUNT
from weld_pipeline.train_joint_relation_crossfit import DATA,load_data,seed_all,atomic_save

OUT=Path('weld_pipeline/runs/joint_event_vote_capacity_seed151')
TARGETS=Path('weld_pipeline/data/joint_event_votes_pilot151')


def prepare_targets(data,names):
    if TARGETS.exists():raise ValueError('Preserve vote target arrays')
    TARGETS.mkdir(parents=True);manifests={}
    for name in names:
        piece=data[name];target=piece['targets'];votes=np.zeros((len(target),7),np.float32)
        known=np.zeros((len(target),7),np.float32);cuts=[]
        for entry in piece['entries']:
            a,b=entry['start'],entry['stop']
            if entry['type']!='horizontal':continue
            original=target[a:b]
            # All prepared straight candidates use the original fixed 2 mm grid.
            dense=dense_targets(original,2.)
            valid=np.c_[original>=0.,(original[:,1:4]>=0.)&(dense[:,1:4]>.5)]
            votes[a:b]=np.where(valid,dense,0.)
            known[a:b]=valid
            ids=np.flatnonzero(original[:,3]>.25)
            if len(ids):cuts.append(dict(entry,event_indices=ids.tolist()))
        np.save(TARGETS/(name+'_targets.npy'),votes);np.save(TARGETS/(name+'_known.npy'),known)
        piece['vote_targets']=np.load(TARGETS/(name+'_targets.npy'),mmap_mode='r')
        piece['vote_known']=np.load(TARGETS/(name+'_known.npy'),mmap_mode='r');piece['cut_entries']=cuts
        manifests[name]=dict(original_source_manifest_sha256=sha256(DATA/name/'manifest.json'),
            targets_sha256=sha256(TARGETS/(name+'_targets.npy')),known_sha256=sha256(TARGETS/(name+'_known.npy')),
            offset_validity_is_explicit_not_sign_test=True,rows=len(target))
        print('EVENT_VOTE_TRAIN_TARGETS',name,'SUPPORT',votes[:,1:4].sum(0).astype(int).tolist(),flush=True)
    (TARGETS/'manifest.json').write_text(json.dumps(manifests,indent=2,ensure_ascii=False),encoding='utf8')
    return manifests


def sample(piece,rng):
    batch=8;points=1024;halo=64
    x=np.zeros((batch,LOCAL_COUNT,points+2*halo),np.float32)
    y=np.zeros((batch,7,points),np.float32);known=np.zeros_like(y);nodes=[]
    cut_focus=bool(piece['cut_entries'] and rng.random()<.5)
    for bi in range(batch):
        if cut_focus:pool=piece['cut_entries']
        else:pool=piece['positive'] if piece['positive'] and rng.random()<.75 else piece['horizontal']
        entry=pool[int(rng.integers(len(pool)))];length=entry['stop']-entry['start']
        if length>points:
            focus=entry['event_indices'] if rng.random()<.65 else entry['foreground_indices']
            if focus and rng.random()<.8:
                centre=focus[int(rng.integers(len(focus)))];start=int(np.clip(centre-rng.integers(points//4,3*points//4),0,length-points))
            else:start=int(rng.integers(length-points+1))
        else:start=0
        stop=min(length,start+points);lo=start-halo;left,right=max(0,lo),min(length,stop+halo)
        x[bi,:,left-lo:right-lo]=piece['features'][entry['start']+left:entry['start']+right].T
        a,b=entry['start']+start,entry['start']+stop
        y[bi,:,:stop-start]=piece['vote_targets'][a:b].T
        known[bi,:,:stop-start]=piece['vote_known'][a:b].T;nodes.append(entry['node'])
    return torch.from_numpy(x).cuda(),torch.tensor(nodes,device='cuda'),torch.from_numpy(y).cuda(),torch.from_numpy(known).cuda()


def loss_fn(logits,target,known):
    bce=F.binary_cross_entropy_with_logits(logits[:,:4],target[:,:4],reduction='none')
    fg=(bce[:,:1]*known[:,:1]).sum()/known[:,:1].sum().clamp_min(1.)
    positive=target[:,1:4]*known[:,1:4];negative=(1.-target[:,1:4])*known[:,1:4]
    pos=(bce[:,1:4]*positive).sum((0,2))/positive.sum((0,2)).clamp_min(1.)
    neg=(bce[:,1:4]*negative).sum((0,2))/negative.sum((0,2)).clamp_min(1.)
    offset=F.smooth_l1_loss(logits[:,4:7],target[:,4:7],beta=.05,reduction='none')
    offset=(offset*known[:,4:7]).sum((0,2))/known[:,4:7].sum((0,2)).clamp_min(1.)
    return fg+.5*(pos+neg).mean()+offset.mean()


def evaluate(model,data,names,role):
    scores={}
    for name in names:
        piece=load_piece(data[name]['manifest']['source_cache_root'],verify=True)
        probabilities=infer_votes(model,data[name],piece['sequences'])
        predictions=decode_intervals(piece['sequences'],probabilities,threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.)
        truth=[w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        scores[name]=score_children(predictions,truth)
        (OUT/(role+'_'+name+'.json')).write_text(json.dumps(dict(score=scores[name],predictions=predictions),indent=2),encoding='utf8')
        np.savez_compressed(OUT/(role+'_'+name+'_heatmaps.npz'),probabilities=np.concatenate(probabilities),
                            lengths=np.array([len(p) for p in probabilities],np.int64))
        print('EVENT_VOTE_CAPACITY_SCORE',role,name,scores[name]['valid_children'],'/',scores[name]['truth_count'],
              'FP',scores[name]['false_predictions'],flush=True)
    result=aggregate(scores)
    (OUT/(role+'_summary.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def main():
    if OUT.exists():raise ValueError('Preserve fixed voting training pilot')
    proof_path=Path('weld_pipeline/reports/joint_event_vote_targets_151_20261001/complete.json')
    proof=json.loads(proof_path.read_text('utf8'))
    if not proof['training_allowed']:raise ValueError('Full train-source representation proof required')
    for source,digest in proof['source_sha256'].items():
        if sha256(source)!=digest:raise ValueError('Vote representation proof source changed')
    data_config=json.loads((DATA/'config.json').read_text('utf8'));fold=data_config['folds'][0]
    names=fold['train_real']
    if names!=proof['train_sources'] or set(names)&set(fold['held_out_real']):raise ValueError('Pilot sources changed')
    OUT.mkdir(parents=True);seed_all(151)
    sources=[__file__,'weld_pipeline/joint_event_votes.py','weld_pipeline/joint_relation_heatmaps.py',
             'weld_pipeline/train_joint_relation_crossfit.py','weld_pipeline/event_gap_decoder.py',
             'weld_pipeline/contract_child_evaluation.py']
    config=dict(version=VERSION,seed=151,fold=0,train=names,held=fold['held_out_real'],epochs=120,
        steps_per_epoch=64,batch_size=8,points=1024,halo=64,width=128,messages=False,
        learning_rate=5e-4,weight_decay=1e-4,loss='foreground_BCE_balanced_event_support_BCE_positive_offset_SmoothL1',
        offset_beta=.05,offset_weight=1.,radius_mm=64.,vote_sigma_mm=1.5,vote_tail_mm=9.,vote_scale=4.,
        source_uniform=True,positive_pool_probability=.75,event_focus_probability=.65,cut_batch_probability=.5,
        data_config_sha256=sha256(DATA/'config.json'),representation_proof_sha256=sha256(proof_path),
        sources={p:sha256(p) for p in sources},no_pretrained_weights=True,
        selection='Fixed last epoch. Strict FIT geometry gates before any new held-source evaluation.',
        geometry_scope='original_fold0_horizontal_straight',other_contract_fields_not_tested=True,
        observed_check_read=False,full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    snap=OUT/'source_snapshot';snap.mkdir()
    for source in sources:shutil.copy2(source,snap/Path(source).name)
    train_config=dict(data_config,pieces=names);data=load_data(train_config)
    manifests=prepare_targets(data,names)
    model=JointEventVotes().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    rng=np.random.default_rng(151);source_steps={n:0 for n in names};started=time.perf_counter()
    # Verify actual negative offsets receive gradients; signed targets must not be treated as unknown.
    x,node,y,known=sample(data[names[0]],rng)
    if not bool(((y[:,4:7]<0)&(known[:,4:7]>0)).any()):raise ValueError('Negative known event offsets absent from pilot batch')
    optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(x,node,*data[names[0]]['torch_graph'])[:,:,64:1088],y,known)
    loss.backward()
    if not torch.isfinite(loss) or sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)<=0:
        raise ValueError('Actual vote training gradient invalid')
    optimizer.zero_grad(set_to_none=True)
    print('EVENT_VOTE_CAPACITY_START','SOURCES',len(names),'EPOCHS',120,flush=True)
    for epoch in range(1,121):
        model.train();losses=[]
        for _ in range(64):
            name=names[int(rng.integers(len(names)))];piece=data[name]
            x,node,y,known=sample(piece,rng)
            logits=model(x,node,*piece['torch_graph'])[:,:,64:1088]
            loss=loss_fn(logits,y,known)
            if not torch.isfinite(loss):raise ValueError('Nonfinite event-vote training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();losses.append(float(loss.detach()));source_steps[name]+=1
        with (OUT/'training.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(epoch=epoch,loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-started))+'\n')
        if epoch%16==0 or epoch==120:
            atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                sampling_rng=rng.bit_generator.state,source_steps=source_steps,config_sha256=sha256(OUT/'config.json')),OUT/'latest.pt')
            print('EVENT_VOTE_CAPACITY_EPOCH',epoch,'LOSS',round(float(np.mean(losses)),6),flush=True)
    atomic_save(dict(model=model.state_dict(),epoch=120,config=config,target_manifest_sha256=sha256(TARGETS/'manifest.json')),OUT/'final.pt')
    frozen=sha256(OUT/'final.pt');fit=evaluate(model,data,names,'FIT')
    eligible=fit['geometry_core_gates_pass']
    (OUT/'fit_adoption_decision.json').write_text(json.dumps(dict(fit=fit,eligible_on_fit=eligible,
        checkpoint_frozen_before_held=True,observed_check_read=False),indent=2),encoding='utf8')
    held=None
    if eligible:
        held_data=load_data(dict(data_config,pieces=fold['held_out_real']))
        held=evaluate(model,held_data,fold['held_out_real'],'POST_FIT_HELD')
    if sha256(OUT/'final.pt')!=frozen:raise ValueError('Frozen event-vote weights changed during evaluation')
    complete=dict(config_sha256=sha256(OUT/'config.json'),checkpoint_sha256=frozen,final_epoch=120,
        fit=fit,held=held,fit_gate_passed=eligible,target_manifests=manifests,source_steps=source_steps,
        held_status='FIXED_POST_FIT_SOURCE_HELD' if eligible else 'NOT_RUN_FIT_CAPACITY_GATE_FAILED',
        elapsed_seconds=time.perf_counter()-started,observed_check_read=False,
        three_seed_source_adoption_not_validated=True,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete,indent=2,ensure_ascii=False),encoding='utf8')
    print('EVENT_VOTE_CAPACITY_COMPLETE',fit['valid_children'],'/',fit['truth_count'],'FP',fit['false_predictions'],
          'HELD',complete['held_status'],flush=True)


if __name__=='__main__':
    main()
