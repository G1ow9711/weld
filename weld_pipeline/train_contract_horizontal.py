"""Sparse endpoint/internal-cut focal training with child-contract validation."""
import argparse
import json
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
from weld_pipeline.contract_horizontal_model import MODEL_NAME,from_checkpoint
from weld_pipeline.horizontal_training_views import horizontal_views
from weld_pipeline.partial_sequence_training_parent import batch_sample,mine_training_structure
from weld_pipeline.train_parent_balanced_branch import focal_loss_fn
from weld_pipeline.train_contract_extent import load,evaluate,quality


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='weld_pipeline/data/contract_child_v1')
    ap.add_argument('--init',default='weld_pipeline/runs/contract_extent_v1_seed121/best.pt')
    ap.add_argument('--outdir',default='weld_pipeline/runs/contract_horizontal_v1_seed122')
    ap.add_argument('--epochs',type=int,default=120)
    ap.add_argument('--steps-per-epoch',type=int,default=96)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--seed',type=int,default=122)
    args=ap.parse_args();root,out=Path(args.data),Path(args.outdir)
    split=json.loads((root/'split.json').read_text('utf8'))
    initial=load(args.init);verify_cached_inference_sources(initial,args.init)
    originals=[n for n in split['train'] if '__' not in n]
    train,views=horizontal_views([load_piece(split['piece_roots'][n],verify=True) for n in originals])
    validation={n:load_piece(split['piece_roots'][n],verify=True) for n in split['validation']}
    hashes={n:sha256(Path(split['piece_roots'][n])/'manifest.json') for n in originals+split['validation']}
    if any(hashes[n]!=initial['data_hashes'][n] for n in hashes):raise ValueError('Original data changed')
    for p in train:
        p['positive_sequences']=[i for i,t in enumerate(p['targets']) if (t[:,0]>.5).any()]
        p['known_sequences']=list(range(len(p['sequences'])))
        p['arc_sequences']=[i for i,s in enumerate(p['sequences']) if s.get('curve_kind')=='arc']
    seed_all(args.seed);rng=np.random.default_rng(args.seed)
    model=from_checkpoint(initial,initialize_extent=True).cuda()
    frozen={n:v.detach().clone() for n,v in model.vertical.state_dict().items()}
    optimizer=torch.optim.AdamW(model.horizontal.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler()
    sources={**initial['source_hashes'],**{str(p):sha256(p) for p in [Path(__file__),
        Path('weld_pipeline/contract_horizontal_model.py'),Path('weld_pipeline/horizontal_training_views.py')]}}
    config={'arguments':vars(args),'source_hashes':sources,'data_hashes':hashes,'views':views,
        'model_name':MODEL_NAME,'split_sha256':sha256(root/'split.json'),'initializer_sha256':sha256(args.init),
        'sampling':'21 original TRAIN horizontal/arc views only; no augmentation, no DEV labels in updates.',
        'loss':'Sparse heatmap focal normalized by positive endpoints/internal cuts; no teacher distillation.',
        'decoder':{'threshold':.5,'weld_penalty':2.,'mask_weight':.25,'event_gap_mm':0.},
        'checkpoint_policy':'Same contract child geometry gates/minimum category F1/global F1; include epoch zero.',
        'scope':'Development only; no independent curve or complete attribute acceptance.'}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise ValueError('Use a fresh run directory')
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name,wanted in sources.items():
        path=Path(name)
        if not path.exists() or sha256(path)!=wanted:path=Path(args.init).parent/'source_snapshot'/path.name
        if sha256(path)!=wanted:raise ValueError('Missing source: '+name)
        shutil.copy2(path,snapshot/path.name)
    best=None;start=time.perf_counter()
    def checkpoint(epoch):
        return {'model_name':MODEL_NAME,'features':84,'feature_version':split['feature_version'],
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch,'elapsed_seconds':time.perf_counter()-start,'source_hashes':sources,
            'data_hashes':hashes,'best_quality':best,'split_sha256':config['split_sha256']}
    summary=evaluate(model,validation,out,0);best=quality(summary)
    atomic_save(checkpoint(0),out/'best.pt')
    (out/'best_validation.json').write_text(json.dumps({'epoch':0,'summary':summary},indent=2),encoding='utf8')
    print('BASELINE',json.dumps(summary),flush=True)
    for epoch in range(1,args.epochs+1):
        if epoch==1 or (epoch-1)%20==0:
            mining=mine_training_structure(model,train)
            (out/f'mining_{epoch:03d}.json').write_text(json.dumps(mining,indent=2),encoding='utf8')
        lr=args.lr*(.15+.85*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
        for g in optimizer.param_groups:g['lr']=lr
        model.train();losses=[]
        for step in range(args.steps_per_epoch):
            x,y,valid=batch_sample(train,rng,8,1024,True,.35,.25)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():loss=focal_loss_fn(model(x),y,valid)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.horizontal.parameters(),5.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        for n,v in model.vertical.state_dict().items():
            if not torch.equal(v,frozen[n]):raise ValueError('Frozen vertical changed: '+n)
        record={'epoch':epoch,'loss':float(np.mean(losses)),'vertical_unchanged':True};improved=False
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            summary=evaluate(model,validation,out,epoch);record['summary']=summary;current=quality(summary)
            if current>tuple(best):best,improved=current,True
        record['elapsed_seconds']=time.perf_counter()-start;state=checkpoint(epoch)
        atomic_save(state,out/'last.pt')
        if improved:
            atomic_save(state,out/'best.pt');(out/'best_validation.json').write_text(json.dumps(record,indent=2),encoding='utf8')
        with (out/'epochs.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record)+'\n')
        print('EPOCH',json.dumps(record),flush=True)
    (out/'complete.json').write_text(json.dumps({'epochs':args.epochs,'best_quality':best,
        'elapsed_seconds':time.perf_counter()-start,'full_contract_pass':False},indent=2),encoding='utf8')


if __name__=='__main__':main()
