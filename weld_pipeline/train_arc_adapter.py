"""Fixed-duration curve-head experiment; preserve the frozen straight model."""
import argparse,json,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from training.labels import sha256
from training.pilot import seed_all
from training.scripts.train_pilot import atomic_save
from weld_pipeline.arc_adapter_model import ArcAdapterContactNet,MODEL_NAME
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.curve_geometry_metrics import check_matched_curves
from weld_pipeline.evaluation import score_welds
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.train_typed_pilot import batch_sample,loss_fn,load,compact
from weld_pipeline.runtime_metrics import process_memory_mb


def evaluate(model,data,out,label):
    result={};dest=out/label;dest.mkdir(exist_ok=True)
    for name,piece in data.items():
        pred=decode_curved(piece['sequences'],infer_probabilities(model,piece['sequences']))
        score=score_welds(pred,piece['truth']);score['curve_geometry']=check_matched_curves(pred,piece['truth'],score)
        arcs=[w for w in piece['truth'] if all(s['kind']=='arc' for s in w['segments'])]
        if arcs:score['arc_score']=score_welds([w for w in pred if all(s['kind']=='arc' for s in w['segments'])],arcs)
        (dest/(name+'_score.json')).write_text(json.dumps(score,indent=2),encoding='utf-8')
        (dest/(name+'_predictions.json')).write_text(json.dumps(pred,indent=2),encoding='utf-8')
        result[name]={'valid':score['valid_welds'],'truth':len(piece['truth']),'predictions':len(pred),
                      'arc_valid':score.get('arc_score',{}).get('valid_welds'),
                      'curve_checked_valid':score['curve_geometry']['valid_welds_after_curve_check']}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='weld_pipeline/data/mixed_surface_v1')
    ap.add_argument('--init',default='weld_pipeline/runs/structured_surface_v1_seed102/best.pt')
    ap.add_argument('--outdir',default='weld_pipeline/runs/arc_adapter_surface_v1_seed105')
    ap.add_argument('--epochs',type=int,default=120);ap.add_argument('--steps-per-epoch',type=int,default=24)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--seed',type=int,default=105)
    ap.add_argument('--stop-after-epoch',type=int,default=0);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    root=Path(args.data);out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    if json.loads((root/'preprocessing_complete.json').read_text())['status']!='READY':raise ValueError('Dataset not ready')
    seed_all(args.seed);torch.set_num_threads(4);rng=np.random.default_rng(args.seed)
    base=load(args.init)
    if base['model_name']!='invariant_surface_process' or base['features']!=84:raise ValueError('Base architecture mismatch')
    planes=[{'axis':int(a),'position_mm':float(p)} for a,p in zip(base['model']['plane_axes'],base['model']['cut_positions_mm'])]
    model=ArcAdapterContactNet(84,planes).cuda();model.load_base(base['model'])
    original={k:v.clone() for k,v in base['model'].items()};del base
    validation={n:load_piece(root/n) for n in split['validation']}
    curve_data=[];curve_full={}
    # Inspect metadata for every training piece, never the validation targets.
    for name in split['train']:
        metadata=json.loads((root/name/'curve_metadata.json').read_text(encoding='utf-8'))
        indices=[i for i,s in enumerate(metadata['sequences']) if s.get('curve_kind')=='arc']
        if not indices:continue
        piece=load_piece(root/name);curve_full[name]=piece
        view={**piece,'sequences':[piece['sequences'][i] for i in indices],'targets':[piece['targets'][i] for i in indices]}
        view['positive_sequences']=[i for i,y in enumerate(view['targets']) if y[:,0].any()]
        curve_data.append(view)
    if not curve_data:raise ValueError('No train curve sequences')
    prior=json.loads(Path('weld_pipeline/runs/mixed_surface_v1_seed104/config.json').read_text(encoding='utf-8'))
    for path,wanted in prior['source_hashes'].items():
        if sha256(path)!=wanted:raise ValueError('Shared source drift: '+path)
    sources=[*[Path(p) for p in prior['source_hashes']],Path(__file__),*[Path('weld_pipeline')/p for p in
        ['arc_adapter_model.py','model_registry_adapter.py','model_registry_context.py','predict_typed.py','curve_geometry_metrics.py']]]
    hashes={str(p):sha256(p) for p in sources};data_hashes={n:sha256(root/n/'manifest.json') for n in split['train']+split['validation']}
    config={'arguments':{k:v for k,v in vars(args).items() if k not in ['resume','stop_after_epoch']},'source_hashes':hashes,
            'data_hashes':data_hashes,'split_sha256':sha256(root/'split.json'),'initial_checkpoint_sha256':sha256(args.init),
            'curve_training_groups':list(curve_full),'selection':'Fixed final epoch specified before training; no independent arc validation available.',
            'model_policy':'One frozen shared trunk and straight head, plus curved-input residual head; no piece identifiers or reference inputs.',
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.0001)
    scaler=torch.cuda.amp.GradScaler();first=1;elapsed_previous=0.
    if args.resume:
        if json.loads((out/'config.json').read_text(encoding='utf-8'))!=config:raise ValueError('Resume provenance differs')
        old=load(out/'last.pt');model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);scaler.load_state_dict(old['scaler'])
        rng.bit_generator.state=old['numpy_rng'];torch.set_rng_state(old['torch_rng']);torch.cuda.set_rng_state_all(old['cuda_rng']);random.setstate(old['python_rng'])
        first=old['epoch']+1;elapsed_previous=old['elapsed_seconds']
        (out/'resume_receipt.json').write_text(json.dumps({'resumed_from_epoch':first-1,'next_epoch':first,'restored_optimizer_states':len(optimizer.state)},indent=2))
    else:
        if (out/'last.pt').exists():raise ValueError('Use resume for existing run')
        (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        snapshots=out/'source_snapshot';snapshots.mkdir(exist_ok=True)
        for p in sources:shutil.copy2(p,snapshots/p.name)
        baseline={'development':evaluate(model,validation,out,'initial_development'),
                  'training_fit':evaluate(model,curve_full,out,'initial_training_fit')}
        (out/'initial_scores.json').write_text(json.dumps(baseline,indent=2));print('INITIAL',json.dumps(baseline),flush=True)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for epoch in range(first,args.epochs+1):
        lr=args.lr*(.15+.85*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();losses=[]
        for _ in range(args.steps_per_epoch):
            x,y,valid=batch_sample(curve_data,rng,8,128,augment_frames=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():loss=loss_fn(model(x),y,valid)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.arc_adapter.parameters(),5.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        row={'epoch':epoch,'loss':float(np.mean(losses)),'lr':lr}
        if epoch%20==0 or epoch==args.epochs:
            row['training_fit']=evaluate(model,curve_full,out,f'training_fit_epoch_{epoch:03d}')
        if epoch==args.epochs:
            row['development']=evaluate(model,validation,out,'final_development')
            for k,v in original.items():
                if not torch.equal(model.state_dict()[k].cpu(),v):raise ValueError('Frozen backbone changed: '+k)
            row['frozen_backbone_exact']='PASS'
        row['elapsed_seconds']=elapsed_previous+time.perf_counter()-start;row['memory']=process_memory_mb()
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,
               'features':84,'model_name':MODEL_NAME,'feature_version':split['feature_version'],'bounded_neighbors':True,
               'source_hashes':hashes,'data_hashes':data_hashes,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'elapsed_seconds':row['elapsed_seconds']}
        atomic_save(state,out/'last.pt')
        with (out/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print('EPOCH',json.dumps(row),flush=True)
        if args.stop_after_epoch and epoch>=args.stop_after_epoch:return
    atomic_save(state,out/'final.pt');(out/'complete.json').write_text(json.dumps(row,indent=2),encoding='utf-8')


if __name__=='__main__':main()
