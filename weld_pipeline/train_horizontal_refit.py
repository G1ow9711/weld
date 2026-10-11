"""Refit horizontal/arc heads on original TRAIN assemblies; freeze verticals."""
import argparse
import copy
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
from weld_pipeline.curved_cache import load_piece, STORAGE_FORMAT
from weld_pipeline.mixed_sequences import FEATURE_VERSION
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.horizontal_refit_branch_model import HorizontalRefitBranchContactNet, MODEL_NAME
from weld_pipeline.horizontal_training_views import horizontal_views
from weld_pipeline.partial_sequence_training_parent import batch_sample, mine_training_structure
from weld_pipeline.train_parent_balanced_branch import loss_fn, evaluate
from weld_pipeline.structured_sampling import hard_heat_loss
from weld_pipeline.consistent_distillation import consistent_distillation
from weld_pipeline.runtime_metrics import process_memory_mb


def load(path):
    with Path(path).open('rb') as stream:
        return torch.load(stream, map_location='cpu')


def quality(scores):
    return (all(s['all_gates_pass'] for s in scores.values()),
            float(np.mean([2*s['valid_welds']/max(1,s['truth_count']+s['prediction_count']) for s in scores.values()])),
            float(np.mean([s['valid_extraction_rate'] for s in scores.values()])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='weld_pipeline/data/partial_augmented_surface_v4')
    ap.add_argument('--init', default='weld_pipeline/runs/typed_branch_surface_v3_seed115/best.pt')
    ap.add_argument('--outdir', default='weld_pipeline/runs/horizontal_refit_v1_seed120')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--steps-per-epoch', type=int, default=96)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--points', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=4e-5)
    ap.add_argument('--seed', type=int, default=120)
    ap.add_argument('--teacher-weight', type=float, default=2.)
    ap.add_argument('--hard-negative-fraction', type=float, default=.35)
    ap.add_argument('--hard-heat-weight', type=float, default=.1)
    ap.add_argument('--curve-sampling-fraction', type=float, default=.25)
    ap.add_argument('--mine-every', type=int, default=20)
    ap.add_argument('--stop-after-epoch', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.epochs < 1 or args.steps_per_epoch < 1 or not 0 <= args.teacher_weight <= 10:
        raise ValueError('Invalid schedule or teacher weight')
    if not 0 <= args.hard_negative_fraction <= .75 or not 0 <= args.curve_sampling_fraction <= .75:
        raise ValueError('Invalid sampling fraction')
    root, out = Path(args.data), Path(args.outdir)
    split = json.loads((root/'split.json').read_text(encoding='utf-8'))
    ready = json.loads((root/'preprocessing_complete.json').read_text(encoding='utf-8'))
    if split['storage_format'] != STORAGE_FORMAT or split['feature_version'] != FEATURE_VERSION or ready['status'] not in ['READY', 'READY_PARTIAL']:
        raise ValueError('Expected verified original 84-channel curved cache')
    for source, wanted in ready.get('sources', {}).items():
        if sha256(source) != wanted:
            raise ValueError('Preprocessing source changed: '+source)
    original_train = split['augmentation_policy']['original_train']
    if set(original_train) & set(split['validation']) or not set(original_train) <= set(split['train']):
        raise ValueError('Original TRAIN membership invalid')
    folders = {n: (root/split.get('piece_roots', {}).get(n, n)).resolve() for n in split['train']+split['validation']}
    if any(not p.is_relative_to(Path.cwd().resolve()) for p in folders.values()):
        raise ValueError('Cache outside project')
    hashes = {n: sha256(p/'manifest.json') for n,p in folders.items()}
    initial = load(args.init)
    verify_cached_inference_sources(initial, args.init)
    if initial.get('model_name') != 'typed_branch_surface' or initial['feature_version'] != FEATURE_VERSION or initial['features'] != 84:
        raise ValueError('Expected frozen 115 branch initializer')
    if any(initial['data_hashes'][n] != hashes[n] for n in original_train + split['validation']):
        raise ValueError('Original initializer cache has changed')
    train, view_report = horizontal_views([load_piece(folders[n]) for n in original_train])
    validation = {n: load_piece(folders[n]) for n in split['validation']}
    for piece in train:
        piece['positive_sequences'] = [i for i,t in enumerate(piece['targets']) if (t[:,0] > .5).any()]
        piece['known_sequences'] = [i for i,t in enumerate(piece['targets']) if (t[:,0] >= 0).any()]
        piece['arc_sequences'] = [i for i,s in enumerate(piece['sequences']) if s.get('curve_kind') == 'arc']
    seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    def planes(prefix):
        return [{'axis': int(a), 'position_mm': float(p)} for a,p in zip(initial['model'][prefix+'.plane_axes'], initial['model'][prefix+'.cut_positions_mm'])]
    model = HorizontalRefitBranchContactNet(84, planes('vertical'), horizontal_planes=planes('horizontal')).cuda()
    model.load_state_dict(initial['model'], strict=True)
    frozen_vertical = {n: p.detach().clone() for n,p in model.vertical.state_dict().items()}
    teacher = copy.deepcopy(model.horizontal).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in model.horizontal.named_parameters() if n != 'cut_positions_mm']},
        {'params': [model.horizontal.cut_positions_mm], 'weight_decay': 0.}], lr=args.lr, weight_decay=.0001)
    scaler = torch.cuda.amp.GradScaler()
    sources = {str(p): sha256(p) for p in [Path(__file__),
        *[Path('weld_pipeline')/n for n in ['horizontal_refit_branch_model.py', 'model_registry_horizontal_refit.py',
        'horizontal_training_views.py', 'train_parent_balanced_branch.py', 'partial_sequence_training_parent.py',
        'parent_balanced_sampling.py', 'vertical_frame_augmentation.py', 'checkpoint_sources.py']]]}
    # Include the verified initializer dependencies used by feature generation and inference.
    sources = {**initial['source_hashes'], **sources}
    arguments = {k:v for k,v in vars(args).items() if k not in ['resume','stop_after_epoch']}
    config = {'arguments': arguments, 'model_name': MODEL_NAME, 'source_hashes': sources, 'data_hashes': hashes,
        'split_sha256': sha256(root/'split.json'), 'original_train': original_train,
        'horizontal_training_views': view_report,
        'augmentation_entries_excluded_from_training': [n for n in split['train'] if n not in original_train],
        'initial_checkpoint_sha256': sha256(args.init), 'frozen_vertical_initial_sha256': sha256(args.init),
        'fixed_decoder_parameters': {'threshold': .5, 'weld_penalty': 2., 'mask_weight': .25},
        'teacher_policy': 'Frozen initial horizontal branch; TRAIN-aligned distillation; curved sequences excluded.',
        'sampling_policy': 'Uniform original TRAIN piece, horizontal/arc views only; known targets preserved.',
        'checkpoint_policy': 'Original four DEV strict macro F1; include epoch-zero baseline; first equal-quality checkpoint retained.',
        'scope': 'Development experiment, not independent or formal acceptance.'}
    out.mkdir(parents=True, exist_ok=True)
    first, elapsed_previous, best = 1, 0., None
    if args.resume:
        previous = json.loads((out/'config.json').read_text(encoding='utf-8'))
        if previous != config:
            raise ValueError('Resume config, data, or source changed')
        state = load(out/'last.pt')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        first, elapsed_previous, best = state['epoch']+1, state['elapsed_seconds'], state['best_quality']
        for piece in train:
            piece['hard_samples'] = state['hard_samples'][piece['record']['name']]
        (out/'resume_receipt.json').write_text(json.dumps({'resumed_from_epoch': first-1, 'next_epoch': first,
            'restored_optimizer_states': len(optimizer.state), 'checkpoint_sha256': sha256(out/'last.pt')},indent=2),encoding='utf-8')
    else:
        if (out/'last.pt').exists() or (out/'epochs.jsonl').exists():
            raise ValueError('Existing run; resume or use a new directory')
        (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        snapshot = out/'source_snapshot'; snapshot.mkdir(exist_ok=True)
        for name,wanted in sources.items():
            source = Path(name)
            if not source.exists() or sha256(source) != wanted:
                source = Path(args.init).parent/'source_snapshot'/source.name
            if sha256(source) != wanted:
                raise ValueError('Cannot archive initializer dependency')
            shutil.copy2(source,snapshot/source.name)
    def checkpoint(epoch, elapsed):
        return {'model_name': MODEL_NAME, 'features': 84, 'feature_version': FEATURE_VERSION,
            'bounded_neighbors': True, 'fast_context': False, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
            'best_quality': best, 'elapsed_seconds': elapsed, 'numpy_rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(), 'python_rng': random.getstate(),
            'hard_samples': {p['record']['name']: p.get('hard_samples',[]) for p in train},
            'source_hashes': sources, 'data_hashes': hashes, 'frozen_vertical_initial_sha256': sha256(args.init)}
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if not args.resume:
        scores = evaluate(model, validation, out, 0)
        best = quality(scores)
        elapsed = time.perf_counter()-start
        atomic_save(checkpoint(0,elapsed), out/'best.pt')
        (out/'best_validation.json').write_text(json.dumps({'epoch':0,'validation':scores,'best_quality':best},indent=2),encoding='utf-8')
        print('EPOCH_ZERO',best,flush=True)
    for epoch in range(first,args.epochs+1):
        if epoch == 1 or (epoch-1)%args.mine_every == 0:
            mining = mine_training_structure(model,train)
            (out/f'negative_mining_epoch_{epoch:03d}.json').write_text(json.dumps({'scope':'Original TRAIN horizontals and arcs only','groups':mining},indent=2),encoding='utf-8')
            print('NEGATIVE_MINING',epoch,sum(r['selected'] for r in mining.values()),flush=True)
        lr = args.lr*(.15+.85*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train(); losses=[]
        for step in range(args.steps_per_epoch):
            x,y,valid = batch_sample(train,rng,args.batch_size,args.points,True,args.hard_negative_fraction,args.curve_sampling_fraction)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits,y,valid)+args.hard_heat_weight*hard_heat_loss(logits,y,valid)
                with torch.no_grad():
                    teacher_logits = teacher(x)
                loss = loss+args.teacher_weight*consistent_distillation(logits,teacher_logits,y,valid,x)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.horizontal.parameters(),5.)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach()))
        record = {'epoch':epoch,'loss':float(np.mean(losses)),'lr':lr,'process_memory_mb':process_memory_mb()}
        improved = False
        if epoch == 1 or epoch%5 == 0 or epoch == args.epochs:
            scores = evaluate(model,validation,out,epoch); current = quality(scores)
            record.update(validation=scores,strict_weld_macro_f1=current[1])
            if current > tuple(best):
                best, improved = current, True
        for name,value in frozen_vertical.items():
            if not torch.equal(value, model.vertical.state_dict()[name]):
                raise ValueError('Frozen vertical state changed: '+name)
        record.update(elapsed_seconds=elapsed_previous+time.perf_counter()-start,
            cuda_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20,vertical_parameters_unchanged=True)
        saved = checkpoint(epoch,record['elapsed_seconds'])
        atomic_save(saved,out/'last.pt')
        if improved:
            atomic_save(saved,out/'best.pt')
            (out/'best_validation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        with (out/'epochs.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record)+'\n')
        brief = {k:v for k,v in record.items() if k!='validation'}
        if 'validation' in record:
            brief['valid_welds'] = {n:[s['valid_welds'],s['truth_count'],s['prediction_count']] for n,s in record['validation'].items()}
        print('EPOCH',json.dumps(brief),flush=True)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            print('CONTROLLED_CHECKPOINT_HANDOFF',epoch,flush=True)
            return
    (out/'complete.json').write_text(json.dumps({'completed_epochs':args.epochs,'best_quality':best,
        'elapsed_seconds':elapsed_previous+time.perf_counter()-start,'not_formal_acceptance':True},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
