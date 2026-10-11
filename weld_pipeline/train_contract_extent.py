"""TRAIN-only vertical refit; new child geometry scoring; frozen horizontal."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from training.pilot import seed_all
from training.scripts.train_pilot import atomic_save
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.candidate_extent_model import MODEL_NAME, from_checkpoint
from weld_pipeline.vertical_training_views import vertical_views
from weld_pipeline.parent_balanced_sampling import SourceBalancedPool
from weld_pipeline.partial_sequence_training_parent import batch_sample,mine_training_structure
from weld_pipeline.train_parent_balanced_branch import loss_fn
from weld_pipeline.structured_sampling import hard_heat_loss
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.contract_child_evaluation import score_children,aggregate


def load(path):
    with Path(path).open('rb') as stream:return torch.load(stream,map_location='cpu')


def evaluate(model,data,out,epoch):
    folder=out/f'validation_epoch_{epoch:03d}';folder.mkdir(exist_ok=True)
    scores={}
    for name,piece in data.items():
        pred=decode_intervals(piece['sequences'],infer_probabilities(model,piece['sequences']),
            threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.)
        score=score_children(pred,piece['truth']);scores[name]=score
        (folder/(name+'_predictions.json')).write_text(json.dumps(pred,indent=2),encoding='utf8')
        (folder/(name+'_score.json')).write_text(json.dumps(score,indent=2),encoding='utf8')
    summary=aggregate(scores)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    return summary


def quality(summary):
    c=summary['categories']
    f1=[2*v['valid']/max(1,v['truth']+v['predictions']) for k,v in c.items() if k in ['horizontal','vertical'] and v['truth']]
    return (summary['geometry_core_gates_pass'],min(f1),
        2*summary['valid_children']/max(1,summary['truth_count']+summary['prediction_count']),
        summary['valid_extraction_rate'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='weld_pipeline/data/contract_child_v1')
    ap.add_argument('--init',default='weld_pipeline/runs/typed_branch_surface_v3_seed115/best.pt')
    ap.add_argument('--outdir',default='weld_pipeline/runs/contract_extent_v1_seed121')
    ap.add_argument('--epochs',type=int,default=120)
    ap.add_argument('--steps-per-epoch',type=int,default=96)
    ap.add_argument('--points',type=int,default=512)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--seed',type=int,default=121)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    root,out=Path(args.data),Path(args.outdir)
    split=json.loads((root/'split.json').read_text('utf8'))
    initial=load(args.init);verify_cached_inference_sources(initial,args.init)
    if initial['features']!=84 or initial['model_name']!='typed_branch_surface':raise ValueError('Wrong initializer')
    def read(name):
        return load_piece(Path(split['piece_roots'][name]),verify=True)
    data_hashes={n:sha256(Path(split['piece_roots'][n])/'manifest.json') for n in split['train']+split['validation']}
    for n in [n for n in split['train'] if '__' not in n]+split['validation']:
        if initial['data_hashes'][n]!=data_hashes[n]:raise ValueError('Original cache changed: '+n)
    train,view_report=vertical_views([read(n) for n in split['train']])
    validation={n:read(n) for n in split['validation']}
    for piece in train:
        piece['positive_sequences']=[i for i,t in enumerate(piece['targets']) if (t[:,0]>.5).any()]
        piece['known_sequences']=[i for i,t in enumerate(piece['targets']) if (t[:,0]>=0).any()]
        piece['arc_sequences']=[]
    prior=Path('weld_pipeline/data/partial_augmented_surface_v3')
    reference_split=json.loads((prior/'split.json').read_text('utf8'))
    active={p['record']['name'] for p in train}
    records=[read(n)['record'] for n in reference_split['train'] if n in active]
    originals=[n for n in split['train'] if '__' not in n]
    pool=SourceBalancedPool(train,records,originals,partial_repeat=4)
    seed_all(args.seed);rng=np.random.default_rng(args.seed)
    model=from_checkpoint(initial,initialize_extent=True).cuda()
    frozen={n:v.detach().clone() for n,v in model.horizontal.state_dict().items()}
    optimizer=torch.optim.AdamW(model.vertical.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler()
    sources={**initial['source_hashes'],**{str(p):sha256(p) for p in [Path(__file__),
        Path('weld_pipeline/candidate_extent_model.py'),Path('weld_pipeline/contract_child_evaluation.py')]}}
    config={'arguments':{k:v for k,v in vars(args).items() if k!='resume'},'source_hashes':sources,
        'data_hashes':data_hashes,'split_sha256':sha256(root/'split.json'),'model_name':MODEL_NAME,
        'initializer_sha256':sha256(args.init),'sampling':pool.report,'vertical_views':view_report,
        'input_policy':'Point-derived 84-channel features only; restore candidate extents 55/56; freeze horizontal.',
        'label_policy':'Original seven-channel targets including all internal cuts; partial unknown masks unchanged.',
        'checkpoint_policy':'Geometry gates first, then minimum H/V F1, total F1 and child rate; retain earlier equal checkpoint.',
        'decoder':{'threshold':.5,'weld_penalty':2.,'mask_weight':.25,'event_gap_mm':0.},
        'attributes':'NOT_RUN_MISSING_TRUTH_OR_IMPLEMENTATION','timing':'DEFERRED_BY_USER',
        'scope':'Development refit; original 21 TRAIN and four DEV fixed, augmented entries separate.'}
    out.mkdir(parents=True,exist_ok=True)
    first,best,previous=1,None,0.
    if args.resume:
        if json.loads((out/'config.json').read_text('utf8'))!=config:raise ValueError('Resume config changed')
        state=load(out/'last.pt');model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']);rng.bit_generator.state=state['numpy_rng']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng'])
        first,best,previous=state['epoch']+1,state['best_quality'],state['elapsed_seconds']
        for p in train:p['hard_samples']=state['hard_samples'][p['record']['name']]
    else:
        if (out/'last.pt').exists():raise ValueError('Existing run requires --resume')
        (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
        archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
        for name,wanted in sources.items():
            path=Path(name)
            if not path.exists() or sha256(path)!=wanted:path=Path(args.init).parent/'source_snapshot'/path.name
            if sha256(path)!=wanted:raise ValueError('Source snapshot missing: '+name)
            shutil.copy2(path,archive/path.name)
    def checkpoint(epoch,elapsed):
        return {'model_name':MODEL_NAME,'features':84,'feature_version':split['feature_version'],
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch,'elapsed_seconds':elapsed,'best_quality':best,'numpy_rng':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),
            'hard_samples':{p['record']['name']:p.get('hard_samples',[]) for p in train},
            'source_hashes':sources,'data_hashes':data_hashes,'split_sha256':config['split_sha256']}
    start=time.perf_counter()
    if not args.resume:
        summary=evaluate(model,validation,out,0);best=quality(summary)
        atomic_save(checkpoint(0,time.perf_counter()-start),out/'best.pt')
        (out/'best_validation.json').write_text(json.dumps({'epoch':0,'summary':summary},indent=2),encoding='utf8')
        print('BASELINE',json.dumps(summary),flush=True)
    for epoch in range(first,args.epochs+1):
        if epoch==first or (epoch-1)%20==0:
            mining=mine_training_structure(model,train)
            (out/f'mining_{epoch:03d}.json').write_text(json.dumps(mining,indent=2),encoding='utf8')
        lr=args.lr*(.15+.85*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
        for g in optimizer.param_groups:g['lr']=lr
        model.train();losses=[]
        for step in range(args.steps_per_epoch):
            x,y,valid=batch_sample(pool,rng,8,args.points,True,.35,0.)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits=model(x);loss=loss_fn(logits,y,valid)+.1*hard_heat_loss(logits,y,valid)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.vertical.parameters(),5.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        for n,v in model.horizontal.state_dict().items():
            if not torch.equal(v,frozen[n]):raise ValueError('Frozen horizontal changed: '+n)
        record={'epoch':epoch,'loss':float(np.mean(losses)),'horizontal_unchanged':True}
        improved=False
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            summary=evaluate(model,validation,out,epoch);current=quality(summary);record['summary']=summary
            if current>tuple(best):best,improved=current,True
        elapsed=previous+time.perf_counter()-start;record['elapsed_seconds']=elapsed
        state=checkpoint(epoch,elapsed);atomic_save(state,out/'last.pt')
        if improved:
            atomic_save(state,out/'best.pt');(out/'best_validation.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        with (out/'epochs.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n')
        print('EPOCH',json.dumps(record),flush=True)
    (out/'complete.json').write_text(json.dumps({'epochs':args.epochs,'best_quality':best,
        'elapsed_seconds':previous+time.perf_counter()-start,'full_contract_pass':False},indent=2),encoding='utf8')


if __name__=='__main__':main()
