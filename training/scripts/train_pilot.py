"""Supervised single-crop fit and multi-workpiece SpUNet pilot training."""
import os, sys, json, time, math, argparse, random, traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from training.labels import sha256
from training.pilot import (DATA, seed_all, training_crop, segmentation_loss, scores,
                           evaluate, create_model, geometry, supervision)


def atomic_save(value,path):
    temp=path.with_suffix('.tmp')
    # PyTorch 1.13's native Windows writer mishandles non-ASCII path strings.
    # A Python file object keeps Unicode path handling in Python.
    with temp.open('wb') as stream:
        torch.save(value,stream)
    os.replace(temp,path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['overfit','train','evaluate'],required=True)
    ap.add_argument('--run',default='pilot_v1_seed42')
    ap.add_argument('--epochs',type=int,default=100)
    ap.add_argument('--steps',type=int,default=300)
    ap.add_argument('--crops-per-piece',type=int,default=4)
    ap.add_argument('--points',type=int,default=32768)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--stop-after-epoch',type=int,default=0,help='Controlled checkpoint handoff for resume verification')
    ap.add_argument('--subset',choices=['validation','internal_test'],default='validation')
    args=ap.parse_args()
    if args.stop_after_epoch and args.stop_after_epoch!=1 and args.stop_after_epoch%5:
        raise ValueError('Controlled handoff must be at a saved checkpoint epoch')
    run=ROOT/'training/runs'/args.run; run.mkdir(parents=True,exist_ok=True)
    split_path=ROOT/'training/configs/pilot_split.frozen.json'
    split=json.loads(split_path.read_text(encoding='utf-8'))
    assert split['status']=='FROZEN_FOR_PILOT'
    assert sha256(DATA/'manifest.json')==split['manifest_sha256']
    split_hash=sha256(split_path)
    seed_all(args.seed)
    model=create_model()
    if args.mode=='evaluate':
        ckpt=torch.load(run/'best.pt',map_location='cpu')
        assert ckpt['split_sha256']==split_hash
        model.load_state_dict(ckpt['model']); del ckpt
        result=evaluate(model,split[args.subset],args.points,run/f'predictions_{args.subset}')
        result['checkpoint_sha256']=sha256(run/'best.pt')
        result['split_sha256']=split_hash
        (run/f'{args.subset}_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return
    if (run/'last.pt').exists() and not args.resume:
        raise RuntimeError('Run already contains a checkpoint; use --resume or a new run name')
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    count={i:sum(g['voxel_label_counts'][str(i)] for g in manifest['groups'] if g['name'] in split['train']) for i in [0,1]}
    foreground_weight=min(10.,math.sqrt(count[0]/max(count[1],1)))
    config={**vars(args),'split_sha256':split_hash,'manifest_sha256':split['manifest_sha256'],
        'foreground_weight':foreground_weight,'model':'SpUNet-v1m1 default 39,144,098 parameters',
        'features':'crop-centered world XYZ / 1000; no labels or weld identifiers as input',
        'augmentation':'none for the first pilot','loss':'weighted CE + foreground Dice',
        'precision':'AMP training, FP32 inference for native Windows spconv compatibility',
        'batchnorm_momentum':.1,
        'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'training/pilot.py',Path(__file__),ROOT/'training/models/spunet.py']},
        'threshold':.5,'optimizer':'AdamW lr=.001 weight_decay=.01',
        'accumulation_steps':1 if args.mode=='overfit' else 4,
        'validation_interval_epochs':5,'early_stop_patience_epochs':20,
        'epoch_definition':'four 32768-point spatial crops per training workpiece; sampled epoch, not a full pass',
        'best_selection':'validation macro point F1; weld reconstruction is a later stage',
        'determinism':'seeds saved; sparse CUDA kernels may have nondeterministic atomic reductions'}
    (run/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    scaler=torch.cuda.amp.GradScaler(init_scale=1024)
    start_epoch=1; best=-1.; best_epoch=0
    rng=np.random.default_rng(args.seed)
    if args.resume:
        ck=torch.load(run/'last.pt',map_location='cpu')
        assert ck['split_sha256']==split_hash
        assert ck['config']['code_sha256']==config['code_sha256'], 'Training code changed across resume'
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start_epoch=ck['epoch']+1; best=ck['best']; best_epoch=ck['best_epoch']
        torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        np.random.set_state(ck['numpy_rng']); random.setstate(ck['python_rng']); rng.bit_generator.state=ck['sampler_rng']
        assert torch.equal(torch.get_rng_state(),ck['torch_rng'])
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),ck['cuda_rng']))
        receipt={'resumed_from_epoch':ck['epoch'],'next_epoch':start_epoch,
            'optimizer_state_entries':len(optimizer.state),'scaler_scale':scaler.get_scale(),
            'model_optimizer_scaler_rng_restored':True,'source_code_hashes_match':True}
        (run/'resume_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        del ck
        print(f'RESUMED epoch={start_epoch} best={best:.4f}',flush=True)
    writer=SummaryWriter(str(run/'tensorboard'))
    begin=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    if args.mode=='overfit':
        data,target,ix=training_crop(split['train'][0],np.random.default_rng(3),args.points)
        np.save(run/'fixed_crop_indices.npy',ix)
        logs=[]; first_loss=None
        for step in range(1,args.steps+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(): logits=model(data)
            loss=segmentation_loss(logits,target,foreground_weight)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Loss is not finite')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            if first_loss is None: first_loss=float(loss)
            if step==1 or step%25==0 or step==args.steps:
                model.eval()
                with torch.no_grad(): p=model(data).float().softmax(1)[:,1]
                metric=scores(target.cpu().numpy(),p.cpu().numpy())
                r={'step':step,'loss':float(loss),'f1_eval':metric['f1'],'precision_eval':metric['precision'],'recall_eval':metric['recall']}
                logs.append(r); print('OVERFIT',json.dumps(r),flush=True)
                writer.add_scalar('train/loss',float(loss),step); writer.add_scalar('fixed_crop/f1_eval',metric['f1'],step)
        result={'mode':'fixed audited training crop; no generalization claim','workpiece':split['train'][0],
            'positive_points':int((target==1).sum()),'points':len(target),'steps':args.steps,
            'first_loss':first_loss,'last_loss':float(loss),'loss_drop_fraction':1-float(loss)/first_loss,
            'metrics':metric,'history':logs,'seconds':time.perf_counter()-begin,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'gate_pass':metric['f1']>=.95 and float(loss)<=first_loss*.2}
        atomic_save({'model':model.state_dict(),'split_sha256':split_hash,'config':config},run/'overfit.pt')
        (run/'overfit_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        writer.close(); print('OVERFIT_COMPLETE gate_pass=',result['gate_pass'],flush=True)
        return
    # Train from independent initialization; never reuse the overfit checkpoint.
    log=(run/'epochs.jsonl').open('a',encoding='utf-8')
    completed=start_epoch-1
    for epoch in range(start_epoch,args.epochs+1):
        epoch_start=time.perf_counter()
        lr=.001*(epoch/5 if epoch<=5 else .5*(1+math.cos(math.pi*(epoch-5)/max(args.epochs-5,1))))
        for pg in optimizer.param_groups: pg['lr']=max(lr,1e-6)
        names=list(split['train']); rng.shuffle(names)
        schedule=[n for n in names for _ in range(args.crops_per_piece)]
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses=[]; updates=0
        for i,name in enumerate(schedule):
            data,target,_=training_crop(name,rng,args.points)
            with torch.cuda.amp.autocast(): logits=model(data)
            loss=segmentation_loss(logits,target,foreground_weight)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Loss is not finite')
            denom=min(4,len(schedule)-(i//4)*4)
            scaler.scale(loss/denom).backward()
            if (i+1)%4==0 or i+1==len(schedule):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); updates+=1
            losses.append(float(loss.detach()))
        row={'epoch':epoch,'loss':float(np.mean(losses)),'lr':optimizer.param_groups[0]['lr'],
            'crops':len(schedule),'optimizer_updates':updates,'train_seconds':time.perf_counter()-epoch_start}
        if epoch%5==0 or epoch==1 or epoch==args.epochs:
            val=evaluate(model,split['validation'],args.points)
            row['validation']=val
            score=val['macro_f1']
            if score>best+1e-5:
                best=score; best_epoch=epoch
                atomic_save({'model':model.state_dict(),'epoch':epoch,'validation':val,
                    'config':config,'split_sha256':split_hash},run/'best.pt')
            (run/f'validation_epoch_{epoch:03d}.json').write_text(json.dumps(val,ensure_ascii=False,indent=2),encoding='utf-8')
            writer.add_scalar('validation/macro_f1',score,epoch)
        row['best_epoch']=best_epoch; row['best_validation_macro_f1']=best
        row['elapsed_seconds']=time.perf_counter()-begin
        log.write(json.dumps(row,ensure_ascii=False)+'\n'); log.flush()
        writer.add_scalar('train/loss',row['loss'],epoch); writer.add_scalar('train/lr',row['lr'],epoch)
        print(f"EPOCH {epoch}/{args.epochs} loss={row['loss']:.4f} train={row['train_seconds']:.1f}s best_val_macro_f1={best:.4f}@{best_epoch}",flush=True)
        if epoch%5==0 or epoch==1 or epoch==args.epochs:
            atomic_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'epoch':epoch,'best':best,'best_epoch':best_epoch,'split_sha256':split_hash,'config':config,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'sampler_rng':rng.bit_generator.state},run/'last.pt')
        completed=epoch
        if args.stop_after_epoch and epoch>=args.stop_after_epoch:
            print('CHECKPOINT_HANDOFF for explicit resume verification',flush=True); break
        if epoch%5==0 and epoch-best_epoch>=20:
            print('EARLY_STOP validation macro F1 unchanged for 20 epochs',flush=True); break
    writer.close(); log.close()
    handoff=bool(args.stop_after_epoch and completed>=args.stop_after_epoch and completed<args.epochs)
    result={'status':'CHECKPOINT_HANDOFF' if handoff else 'TRAINING_COMPLETED','completed_epochs':completed,'best_epoch':best_epoch,
        'best_validation_macro_f1':best,'elapsed_seconds':time.perf_counter()-begin,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'best_checkpoint_sha256':sha256(run/'best.pt'),'internal_test_used_during_training':False,
        'scope':'semantic weld-neighborhood model only; no seam-instance or 3 mm endpoint acceptance'}
    (run/('checkpoint_handoff.json' if handoff else 'training_result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc(); raise
