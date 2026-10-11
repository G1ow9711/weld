"""Train a common straight residual while keeping the learned curve model fixed."""
import argparse,json,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from training.pilot import seed_all
from training.scripts.train_pilot import atomic_save
from weld_pipeline.straight_residual_model import StraightResidualContactNet,MODEL_NAME
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.train_mixed_distillation import loss_fn,load,compact
from weld_pipeline.structured_sampling import hard_heat_loss
from weld_pipeline.consistent_distillation import consistent_distillation
from weld_pipeline.residual_sequence_windows import batch_sample,mine_training_structure,infer_probabilities
from weld_pipeline.interval_decoder import decode_intervals
from weld_pipeline.interval_config import load_interval_config
from weld_pipeline.evaluation import score_welds
from weld_pipeline.curve_geometry_metrics import check_matched_curves
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.runtime_metrics import process_memory_mb


def evaluate(model,data,out,label,parameters):
    dest=out/label;dest.mkdir(exist_ok=True);scores={}
    for name,piece in data.items():
        pred=decode_intervals(piece['sequences'],infer_probabilities(model,piece['sequences']),**parameters)
        score=score_welds(pred,piece['truth'])
        score['curve_geometry']=check_matched_curves(pred,piece['truth'],score)
        (dest/(name+'_score.json')).write_text(json.dumps(score,indent=2),encoding='utf-8')
        (dest/(name+'_predictions.json')).write_text(json.dumps(pred,indent=2),encoding='utf-8')
        scores[name]=compact(score)
    return scores


def quality(scores):
    return (all(s['all_gates_pass'] for s in scores.values()),
            float(np.mean([2*s['valid_welds']/max(1,s['truth_count']+s['prediction_count']) for s in scores.values()])),
            float(np.mean([s['valid_extraction_rate'] for s in scores.values()])))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='weld_pipeline/data/mixed_surface_v1')
    ap.add_argument('--init',default='weld_pipeline/runs/arc_adapter_surface_v1_seed105/final.pt')
    ap.add_argument('--decoder-config',default='weld_pipeline/runs/arc_adapter_surface_v1_seed105/interval_calibration/decoder_config.json')
    ap.add_argument('--outdir',default='weld_pipeline/runs/straight_residual_surface_v2_seed108')
    ap.add_argument('--epochs',type=int,default=120);ap.add_argument('--steps-per-epoch',type=int,default=48)
    ap.add_argument('--lr',type=float,default=.0002);ap.add_argument('--seed',type=int,default=108)
    ap.add_argument('--stop-after-epoch',type=int,default=0);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();root=Path(a.data);out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    if json.loads((root/'preprocessing_complete.json').read_text())['status']!='READY':raise ValueError('Dataset not ready')
    seed_all(a.seed);torch.set_num_threads(4);rng=np.random.default_rng(a.seed)
    prior=load(a.init);historical=verify_cached_inference_sources(prior,a.init)
    if prior['model_name']!='arc_adapter_surface_process':raise ValueError('Expected shared curve adapter checkpoint')
    parameters,_=load_interval_config(a.decoder_config,a.init)
    planes=[{'axis':int(ax),'position_mm':float(p)} for ax,p in zip(prior['model']['plane_axes'],prior['model']['cut_positions_mm'])]
    model=StraightResidualContactNet(84,planes).cuda();model.load_prior(prior['model'])
    frozen={k:v.clone() for k,v in prior['model'].items()}
    data_hashes={n:sha256(root/n/'manifest.json') for n in split['train']+split['validation']}
    if data_hashes!=prior['data_hashes']:raise ValueError('Prior data manifests differ')
    train=[];added={}
    for name in split['train']:
        piece=load_piece(root/name)
        indices=[i for i,s in enumerate(piece['sequences']) if s.get('curve_kind')!='arc']
        view={**piece,'sequences':[piece['sequences'][i] for i in indices],'targets':[piece['targets'][i] for i in indices]}
        view['positive_sequences']=[i for i,y in enumerate(view['targets']) if y[:,0].any()]
        train.append(view)
        if any(s.get('curve_kind')=='arc' for s in piece['sequences']):added[name]=piece
    # The added geometry's straight sequences receive 6/26 of batches; no new pieces are fabricated.
    pool=train+[p for p in train if p['record']['name'] in added for _ in range(5)]
    validation={n:load_piece(root/n) for n in split['validation']}
    sources={str(Path(p)):sha256(p) for p in prior['source_hashes']}
    for name in ['straight_residual_model.py','model_registry_residual.py','train_mixed_distillation.py',
                 'structured_sampling.py','consistent_distillation.py','interval_decoder.py','constrained_decoder.py',
                 'interval_config.py','checkpoint_sources.py','residual_sequence_windows.py']:
        p=Path('weld_pipeline')/name;sources[str(p)]=sha256(p)
    sources[str(Path(__file__))]=sha256(Path(__file__))
    config={'arguments':{k:v for k,v in vars(a).items() if k not in ['resume','stop_after_epoch']},
            'source_hashes':sources,'data_hashes':data_hashes,'split_sha256':sha256(root/'split.json'),
            'initial_checkpoint_sha256':sha256(a.init),'decoder_config_sha256':sha256(a.decoder_config),
            'fixed_decoder_parameters':parameters,'historical_entrypoints':historical,
            'sampling':{'unique_training_groups':len(train),'added_geometry_groups':list(added),'added_group_multiplicity':6},
            'selection':'Initial checkpoint and every fifth epoch by development macro F1; no per-piece selection.',
            'policy':'Frozen prior trunk/curve branch; one straight residual; original labels and geometric breaks retained.',
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=.0001)
    scaler=torch.cuda.amp.GradScaler();first=1;elapsed_previous=0.;best=None
    if a.resume:
        if json.loads((out/'config.json').read_text(encoding='utf-8'))!=config:raise ValueError('Resume provenance differs')
        old=load(out/'last.pt');model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scaler.load_state_dict(old['scaler'])
        rng.bit_generator.state=old['numpy_rng'];torch.set_rng_state(old['torch_rng']);torch.cuda.set_rng_state_all(old['cuda_rng']);random.setstate(old['python_rng'])
        first=old['epoch']+1;elapsed_previous=old['elapsed_seconds'];best=tuple(old['best_quality'])
        for p in train:p['hard_samples']=old['hard_samples'][p['record']['name']]
        (out/'resume_receipt.json').write_text(json.dumps({'resumed_from_epoch':first-1,'next_epoch':first,
            'restored_optimizer_states':len(optimizer.state),'checkpoint_sha256':sha256(out/'last.pt')},indent=2))
    else:
        if (out/'last.pt').exists() or (out/'config.json').exists():raise ValueError('Use resume or a fresh output directory')
        (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        dest=out/'source_snapshot';dest.mkdir(exist_ok=True)
        for p in sources:shutil.copy2(p,dest/Path(p).name)
    def checkpoint(epoch,elapsed):
        return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'epoch':epoch,'features':84,'model_name':MODEL_NAME,'feature_version':split['feature_version'],
            'bounded_neighbors':True,'inference_halo':512,'source_hashes':sources,'data_hashes':data_hashes,
                'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'python_rng':random.getstate(),'elapsed_seconds':elapsed,'best_quality':best,
                'hard_samples':{p['record']['name']:p.get('hard_samples',[]) for p in train}}
    if not a.resume:
        scores=evaluate(model,validation,out,'validation_epoch_000',parameters);best=quality(scores)
        atomic_save(checkpoint(0,0.),out/'best.pt')
        (out/'best_validation.json').write_text(json.dumps({'epoch':0,'validation':scores,'strict_weld_macro_f1':best[1]},indent=2))
        print('INITIAL',json.dumps({n:[s['valid_welds'],s['prediction_count']] for n,s in scores.items()}),flush=True)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for epoch in range(first,a.epochs+1):
        if epoch==1 or (epoch-1)%20==0:
            mining=mine_training_structure(model,train)
            (out/f'mining_epoch_{epoch:03d}.json').write_text(json.dumps(mining,indent=2))
        lr=a.lr*(.15+.85*.5*(1+np.cos(np.pi*(epoch-1)/a.epochs)))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();losses=[]
        for step in range(a.steps_per_epoch):
            x,y,valid=batch_sample(pool,rng,8,512,augment_frames=True,hard_fraction=.35)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits,base=model.forward_with_prior(x)
                loss=loss_fn(logits,y,valid)+.1*hard_heat_loss(logits,y,valid)+2*consistent_distillation(logits,base,y,valid,x)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.straight_adapter.parameters(),5.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        row={'epoch':epoch,'loss':float(np.mean(losses)),'lr':lr};improved=False
        if epoch==1 or epoch%5==0 or epoch==a.epochs:
            row['validation']=evaluate(model,validation,out,f'validation_epoch_{epoch:03d}',parameters)
            q=quality(row['validation']);row['strict_weld_macro_f1']=q[1]
            if q>best:best=q;improved=True
        if epoch%20==0 or epoch==a.epochs:
            row['training_fit']=evaluate(model,added,out,f'training_fit_epoch_{epoch:03d}',parameters)
        row['elapsed_seconds']=elapsed_previous+time.perf_counter()-start;row['memory']=process_memory_mb()
        state=checkpoint(epoch,row['elapsed_seconds']);atomic_save(state,out/'last.pt')
        if improved:
            atomic_save(state,out/'best.pt');(out/'best_validation.json').write_text(json.dumps(row,indent=2))
        with (out/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        brief={k:v for k,v in row.items() if k not in ['validation','training_fit']}
        for key in ['validation','training_fit']:
            if key in row:brief[key]={n:[s['valid_welds'],s['truth_count'],s['prediction_count']] for n,s in row[key].items()}
        print('EPOCH',json.dumps(brief),flush=True)
        if a.stop_after_epoch and epoch>=a.stop_after_epoch:return
    for k,v in frozen.items():
        if not torch.equal(model.state_dict()[k].cpu(),v):raise ValueError('Frozen prior changed: '+k)
    (out/'complete.json').write_text(json.dumps({'epochs':a.epochs,'best_quality':best,'frozen_prior_exact':'PASS',
        'elapsed_seconds':elapsed_previous+time.perf_counter()-start,'formal_acceptance':False},indent=2))


if __name__=='__main__':main()
