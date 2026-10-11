"""Train-source event-capacity pilot; keep the failed cross-fit weights frozen."""
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
from weld_pipeline.joint_relation_heatmaps import JointRelationHeatmaps,infer_cached
from weld_pipeline.train_joint_relation_crossfit import DATA,OUT as PARENT,atomic_save,load_data,sample_batch,seed_all

OUT=Path('weld_pipeline/runs/joint_event_capacity_seed147')


def balanced_loss(logits,target,known):
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    foreground=(bce[:,:1]*known[:,:1]).sum()/known[:,:1].sum().clamp_min(1.)
    positive=(target[:,1:]>.001).float()*known[:,1:]
    negative=(target[:,1:]<=.001).float()*known[:,1:]
    pos=(bce[:,1:]*positive).sum((0,2))/positive.sum((0,2)).clamp_min(1.)
    neg=(bce[:,1:]*negative).sum((0,2))/negative.sum((0,2)).clamp_min(1.)
    return foreground+.5*(pos+neg).mean()


def evaluate(model,data,names,run,role):
    scores={}
    for name in names:
        piece=load_piece(data[name]['manifest']['source_cache_root'],verify=True)
        probabilities=infer_cached(model,data[name],piece['sequences'])
        predictions=decode_intervals(piece['sequences'],probabilities,threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.)
        truth=[w for w in piece['truth'] if w['type']=='horizontal' and all(s['kind']=='line' for s in w['segments'])]
        scores[name]=score_children(predictions,truth)
        (run/(role+'_'+name+'.json')).write_text(json.dumps(dict(score=scores[name],predictions=predictions),indent=2),encoding='utf8')
        print('EVENT_CAPACITY_SCORE',run.name,role,name,scores[name]['valid_children'],'/',scores[name]['truth_count'],'FP',scores[name]['false_predictions'],flush=True)
    result=aggregate(scores)
    (run/(role+'_summary.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def cut_pools(data):
    for piece in data.values():
        cuts=[]
        for entry in piece['horizontal']:
            target=piece['targets'][entry['start']:entry['stop'],3]
            ids=np.flatnonzero(target>.25)
            if len(ids):
                cut=dict(entry);cut['event_indices']=ids.tolist();cuts.append(cut)
        piece['cut_entries']=cuts


def train_one(data,names,messages,protocol):
    kind='messages' if messages else 'no_messages';run=OUT/kind
    if (run/'complete.json').exists():
        return json.loads((run/'complete.json').read_text('utf8'))
    run.mkdir(exist_ok=True)
    parent=PARENT/('fold0_seed143_'+kind)/'latest.pt';parent_hash=sha256(parent)
    seed_all(143);model=JointRelationHeatmaps(messages).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    state=torch.load(parent,map_location='cpu')
    if state['epoch']!=24 or state['config']['train']!=names:
        raise ValueError('Unexpected train-source parent checkpoint')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    rng=np.random.default_rng(147);epoch_start=24;source_steps={name:0 for name in names}
    if (run/'latest.pt').exists():
        resumed=torch.load(run/'latest.pt',map_location='cpu')
        if resumed['protocol_sha256']!=sha256(OUT/'config.json'):
            raise ValueError('Event-capacity resume protocol changed')
        model.load_state_dict(resumed['model']);optimizer.load_state_dict(resumed['optimizer'])
        rng.bit_generator.state=resumed['sampling_rng'];epoch_start=resumed['epoch'];source_steps=resumed['source_steps']
    started=time.perf_counter();print('EVENT_CAPACITY_START',kind,epoch_start,'TO',120,flush=True)
    for epoch in range(epoch_start+1,121):
        model.train();losses=[]
        for _ in range(64):
            name=names[int(rng.integers(len(names)))];piece=data[name];sample_piece=piece
            if piece['cut_entries'] and rng.random()<.5:
                sample_piece=dict(piece);sample_piece['horizontal']=piece['cut_entries'];sample_piece['positive']=piece['cut_entries']
            x,nodes,y,known=sample_batch(sample_piece,rng)
            logits=model(x,nodes,*piece['torch_graph'])[:,:,64:64+1024]
            loss=balanced_loss(logits,y,known)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite balanced Gaussian event loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();losses.append(float(loss.detach()));source_steps[name]+=1
        with (run/'training.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(epoch=epoch,loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-started))+'\n')
        if epoch%16==0 or epoch==120:
            atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                sampling_rng=rng.bit_generator.state,source_steps=source_steps,protocol_sha256=sha256(OUT/'config.json')),run/'latest.pt')
            print('EVENT_CAPACITY_EPOCH',kind,epoch,'LOSS',round(float(np.mean(losses)),6),flush=True)
    atomic_save(dict(model=model.state_dict(),epoch=120,config=protocol,messages=messages,parent_sha256=parent_hash),run/'final.pt')
    frozen=sha256(run/'final.pt');fit=evaluate(model,data,names,run,'FIT')
    if sha256(parent)!=parent_hash or sha256(run/'final.pt')!=frozen:
        raise ValueError('Frozen weights changed during FIT evaluation')
    result=dict(messages=messages,fit=fit,parent_sha256=parent_hash,final_sha256=frozen,source_steps=source_steps,
        final_epoch=120,fit_gate_passed=fit['geometry_core_gates_pass'],elapsed_seconds=time.perf_counter()-started,
        source_FIT_only=True,full_contract_pass=False)
    (run/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    del model,optimizer;torch.cuda.empty_cache()
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((PARENT/'config.json').read_text('utf8'));fold=old['folds'][0];names=fold['train_real']
    capacity=Path('weld_pipeline/reports/joint_relation_fit_capacity_20261001_v2/summary.json')
    sources=[__file__,'weld_pipeline/joint_relation_heatmaps.py','weld_pipeline/train_joint_relation_crossfit.py',
        'weld_pipeline/event_gap_decoder.py','weld_pipeline/contract_child_evaluation.py']
    protocol=dict(role='TRAIN_SOURCE_EVENT_CAPACITY_PILOT',fold=0,parent_seed=143,sampling_seed=147,
        train_real=names,held_real=fold['held_out_real'],first_epoch=25,final_epoch=120,steps_per_epoch=64,
        batch_size=8,points=1024,halo=64,learning_rate=5e-4,weight_decay=1e-4,
        loss='foreground_BCE_plus_equal_weight_positive_support_and_background_soft_Gaussian_BCE',
        cut_focus_batch_probability=.5,prior_capacity_audit_sha256=sha256(capacity),
        data_config_sha256=sha256(DATA/'config.json'),sources={path:sha256(path) for path in sources},
        parent_sha256={kind:sha256(PARENT/('fold0_seed143_'+kind)/'latest.pt') for kind in ('messages','no_messages')},
        decoder=dict(threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.),
        selection='Fixed last epoch. Both models must pass full train-source H-line geometry before any new held evaluation.',
        full_contract_pass=False)
    path=OUT/'config.json'
    if path.exists() and json.loads(path.read_text('utf8'))!=protocol:
        raise ValueError('Frozen event-capacity protocol changed')
    path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf8')
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for source in sources:
        shutil.copy2(source,snapshot/Path(source).name)
    config=json.loads((DATA/'config.json').read_text('utf8'));config['pieces']=names
    data=load_data(config);cut_pools(data)
    results=[train_one(data,names,messages,protocol) for messages in (False,True)]
    eligible=all(result['fit_gate_passed'] for result in results);held={}
    if eligible:
        config['pieces']=fold['held_out_real'];held_data=load_data(config)
        for messages in (False,True):
            kind='messages' if messages else 'no_messages';run=OUT/kind
            model=JointRelationHeatmaps(messages).cuda();state=torch.load(run/'final.pt',map_location='cpu');model.load_state_dict(state['model'])
            held[kind]=evaluate(model,held_data,fold['held_out_real'],run,'POST_FIT_HELD')
            del model;torch.cuda.empty_cache()
    final=dict(results=results,both_FIT_gates_passed=eligible,held_summary=held,
        held_status='POST_FIT_DEVELOPMENT_HELD' if eligible else 'NOT_RUN_FIT_CAPACITY_GATE_FAILED',
        observed_check_read=False,three_seed_adoption_not_validated=True,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(final,indent=2),encoding='utf8')
    print('EVENT_CAPACITY_COMPLETE',json.dumps(final),flush=True)


if __name__=='__main__':
    main()
