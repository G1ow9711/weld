"""Source-group cross fitting: same initialization/sampling, message ablation."""
import json
import random
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
from weld_pipeline.joint_relation_heatmaps import VERSION,JointRelationHeatmaps,LOCAL_COUNT,infer_cached

DATA=Path('weld_pipeline/data/joint_relation_v1')
OUT=Path('weld_pipeline/runs/joint_relation_crossfit_seed143')


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False


def load_data(config):
    complete=json.loads((DATA/'complete.json').read_text('utf8'))
    data={}
    for name in config['pieces']:
        root=DATA/name;manifest=json.loads((root/'manifest.json').read_text('utf8'))
        if sha256(root/'manifest.json')!=complete['manifests'][name]:
            raise ValueError('Prepared source manifest changed')
        for file,digest in manifest['files'].items():
            if sha256(root/file)!=digest:
                raise ValueError('Prepared source file changed')
        with np.load(root/'graph.npz',allow_pickle=False) as graph:
            arrays={key:graph[key].copy() for key in ['nodes','edge_index','edge_type','edge_features']}
        entries=manifest['entries']
        horizontal=[entry for entry in entries if entry['type']=='horizontal']
        positive=[entry for entry in horizontal if entry['foreground_indices']]
        data[name]=dict(features=np.load(root/'features.npy',mmap_mode='r'),targets=np.load(root/'targets.npy',mmap_mode='r'),
            entries=entries,horizontal=horizontal,positive=positive,manifest=manifest,
            torch_graph=tuple(torch.from_numpy(arrays[key]).cuda() for key in ['nodes','edge_index','edge_type','edge_features']))
    return data


def sample_batch(piece,rng,batch_size=8,points=1024,halo=64):
    length=points+2*halo
    x=np.zeros((batch_size,LOCAL_COUNT,length),np.float32)
    y=np.zeros((batch_size,4,points),np.float32)
    valid=np.zeros((batch_size,4,points),np.float32);nodes=[]
    for bi in range(batch_size):
        pool=piece['positive'] if piece['positive'] and rng.random()<.75 else piece['horizontal']
        entry=pool[int(rng.integers(len(pool)))];size=entry['stop']-entry['start']
        if size>points:
            focus=entry['event_indices'] if rng.random()<.65 else entry['foreground_indices']
            if focus and rng.random()<.8:
                centre=focus[int(rng.integers(len(focus)))]
                start=int(np.clip(centre-rng.integers(points//4,3*points//4),0,size-points))
            else:
                start=int(rng.integers(size-points+1))
        else:
            start=0
        stop=min(size,start+points);lo=start-halo
        a,b=max(0,lo),min(size,stop+halo)
        x[bi,:,a-lo:b-lo]=piece['features'][entry['start']+a:entry['start']+b].T
        target=piece['targets'][entry['start']+start:entry['start']+stop].T
        known=target>=0
        y[bi,:,:stop-start]=np.where(known,target,0.)
        valid[bi,:,:stop-start]=known
        nodes.append(entry['node'])
    return (torch.from_numpy(x).cuda(),torch.tensor(nodes,device='cuda'),torch.from_numpy(y).cuda(),torch.from_numpy(valid).cuda())


def loss_fn(logits,target,known):
    foreground=F.binary_cross_entropy_with_logits(logits[:,:1],target[:,:1],reduction='none')
    foreground=((foreground*known[:,:1]).sum((1,2))/known[:,:1].sum((1,2)).clamp_min(1.)).mean()
    h=logits[:,1:].float();p=h.sigmoid();y=target[:,1:];mask=known[:,1:]
    positive=(y>.65).float()*mask;negative=(y<=.65).float()*mask
    pos_loss=(1-p).pow(2)*F.softplus(-h)*positive
    neg_loss=p.pow(2)*F.softplus(h)*(1-y).pow(4)*negative
    heat=((pos_loss+neg_loss).sum((0,2))/positive.sum((0,2)).clamp_min(1.)).mean()
    return foreground+heat


def atomic_save(state,path):
    temporary=path.with_suffix('.pending.pt');torch.save(state,temporary);temporary.replace(path)


def evaluate(model,data,names,run):
    scores={}
    for name in names:
        piece=load_piece(data[name]['manifest']['source_cache_root'],verify=True)
        probabilities=infer_cached(model,data[name],piece['sequences'])
        predictions=decode_intervals(piece['sequences'],probabilities,threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.)
        truth=[weld for weld in piece['truth'] if weld['type']=='horizontal' and all(segment['kind']=='line' for segment in weld['segments'])]
        score=score_children(predictions,truth);scores[name]=score
        (run/('held_'+name+'.json')).write_text(json.dumps(dict(predictions=predictions,score=score),indent=2),encoding='utf8')
        print('RELATION_HELD',run.name,name,score['valid_children'],'/',score['truth_count'],'FP',score['false_predictions'],flush=True)
    result=aggregate(scores);result.update(scope='H_straight_source_held_out_model_not_blind_acceptance',full_contract_pass=False)
    (run/'held_summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result,scores


def train_one(protocol,data,fold,seed,messages):
    kind='messages' if messages else 'no_messages'
    run=OUT/('fold'+str(fold['fold'])+'_seed'+str(seed)+'_'+kind)
    if (run/'complete.json').exists():
        return json.loads((run/'complete.json').read_text('utf8'))
    run.mkdir(parents=True,exist_ok=True)
    seed_all(seed);model=JointRelationHeatmaps(messages).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    rng=np.random.default_rng(seed+10000*fold['fold']);epoch_start=0
    train_names=fold['train_real'];held_names=fold['held_out_real']
    if set(train_names)&set(held_names):
        raise ValueError('Source fold leakage')
    run_config=dict(protocol_sha256=sha256(OUT/'config.json'),fold=fold['fold'],seed=seed,messages=messages,
                    train=train_names,held_out=held_names,architecture=VERSION,width=128,final_epoch=24)
    config_path=run/'config.json'
    if config_path.exists() and json.loads(config_path.read_text('utf8'))!=run_config:
        raise ValueError('Fixed run configuration differs')
    config_path.write_text(json.dumps(run_config,indent=2,ensure_ascii=False),encoding='utf8')
    source_steps={name:0 for name in train_names};started=time.perf_counter()
    if (run/'latest.pt').exists():
        state=torch.load(run/'latest.pt',map_location='cpu')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['sampling_rng'];epoch_start=state['epoch'];source_steps=state['source_steps']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    print('RELATION_START',run.name,'TRAIN',len(train_names),'HELD',len(held_names),'EPOCH',epoch_start,flush=True)
    for epoch in range(epoch_start+1,25):
        model.train();losses=[]
        for step in range(64):
            name=train_names[int(rng.integers(len(train_names)))];piece=data[name]
            x,nodes,y,known=sample_batch(piece,rng)
            logits=model(x,nodes,*piece['torch_graph'])[:,:,64:64+1024]
            loss=loss_fn(logits,y,known)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite cross-fit loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()));source_steps[name]+=1
        with (run/'training.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(dict(epoch=epoch,loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-started))+'\n')
        if epoch%8==0 or epoch==24:
            atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                sampling_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                source_steps=source_steps,config=run_config),run/'latest.pt')
            print('RELATION_EPOCH',run.name,epoch,'LOSS',round(float(np.mean(losses)),6),flush=True)
    atomic_save(dict(model=model.state_dict(),config=run_config,epoch=24,source_sha256=protocol['sources']),run/'final.pt')
    frozen=sha256(run/'final.pt')
    summary,scores=evaluate(model,data,held_names,run)
    if sha256(run/'final.pt')!=frozen:
        raise ValueError('Weights changed during held-out evaluation')
    result=dict(config=run_config,held_summary=summary,per_source=scores,final_sha256=frozen,
                source_steps=source_steps,elapsed_seconds=time.perf_counter()-started,
                final_epoch_fixed_before_held_labels=True,full_contract_pass=False)
    (run/'complete.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    del model,optimizer;torch.cuda.empty_cache()
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data_config=json.loads((DATA/'config.json').read_text('utf8'))
    check=json.loads(Path('weld_pipeline/reports/joint_relation_geometry_20261001.json').read_text('utf8'))
    if not check['passed'] or check['source_sha256']!=sha256('weld_pipeline/joint_relation_heatmaps.py'):
        raise ValueError('Current graph physical check required')
    sources=[__file__,'weld_pipeline/joint_relation_heatmaps.py','weld_pipeline/event_gap_decoder.py',
             'weld_pipeline/contract_child_evaluation.py','weld_pipeline/canonical_horizontal_patch_heatmaps.py']
    protocol=dict(version=VERSION,seeds=[143,144,145],folds=data_config['folds'],epochs=24,steps_per_epoch=64,
        batch_size=8,points=1024,halo=64,learning_rate=5e-4,weight_decay=1e-4,width=128,
        source_balance='uniform real source per batch',foreground_sequence_probability=.75,event_focus_probability=.65,
        loss='foreground_BCE_plus_sparse_event_focal',decoder=dict(threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.),
        sources={path:sha256(path) for path in sources},data_config_sha256=sha256(DATA/'config.json'),
        selection='Final epoch fixed. No held-source gradients, no pretraining, no observed-check architecture tuning.',
        adoption='Each seed >=20% FN+FP reduction; majority of sources improved; aggregate TP and FP not worse.',
        applicable_scope='horizontal_straight_only',curves_and_V_not_validated=True,full_contract_pass=False)
    path=OUT/'config.json'
    if path.exists() and json.loads(path.read_text('utf8'))!=protocol:
        raise ValueError('Frozen cross-fit protocol changed')
    path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf8')
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for source in sources:
        shutil.copy2(source,snapshot/Path(source).name)
    data=load_data(data_config)
    results={}
    for seed in protocol['seeds']:
        for messages in (False,True):
            for fold in protocol['folds']:
                result=train_one(protocol,data,fold,seed,messages)
                results[(seed,messages,fold['fold'])]=result
    comparisons=[]
    for seed in protocol['seeds']:
        merged={}
        for messages in (False,True):
            scores={name:score for fold in protocol['folds'] for name,score in results[(seed,messages,fold['fold'])]['per_source'].items()}
            if set(scores)!=set(data_config['pieces']):
                raise ValueError('Source OOF coverage incomplete')
            merged[messages]=dict(summary=aggregate(scores),scores=scores)
        base,graph=merged[False],merged[True]
        eb=base['summary']['missed_children']+base['summary']['false_predictions']
        eg=graph['summary']['missed_children']+graph['summary']['false_predictions']
        improved=sum(graph['scores'][name]['missed_children']+graph['scores'][name]['false_predictions']<
                     base['scores'][name]['missed_children']+base['scores'][name]['false_predictions'] for name in data_config['pieces'])
        reduction=(eb-eg)/eb if eb else 0.
        passed=(reduction>=.2 and improved>len(data_config['pieces'])/2 and
                graph['summary']['valid_children']>=base['summary']['valid_children'] and
                graph['summary']['false_predictions']<=base['summary']['false_predictions'])
        comparisons.append(dict(seed=seed,no_messages=base['summary'],messages=graph['summary'],
                                error_reduction=reduction,improved_sources=improved,development_gate_passed=passed))
    final=dict(comparisons=comparisons,message_adoption_eligible=all(row['development_gate_passed'] for row in comparisons),
               models_completed=18,held_source_results_not_blind_acceptance=True,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(final,indent=2,ensure_ascii=False),encoding='utf8')
    print('RELATION_CROSSFIT_COMPLETE',json.dumps(final),flush=True)


if __name__=='__main__':
    main()
