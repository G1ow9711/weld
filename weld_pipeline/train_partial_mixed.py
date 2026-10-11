"""Multi-assembly development training; shared weights, no per-piece model."""
import argparse,json,time,shutil,random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from training.pilot import seed_all
from training.labels import sha256
from training.scripts.train_pilot import atomic_save
from weld_pipeline.typed_contact_model import TypedContactNet,infer_probabilities,decode_typed
from weld_pipeline.evaluation import score_welds
from weld_pipeline.process_plane_model import ProcessPlaneContactNet,estimate_training_planes
from weld_pipeline.sequence_augmentation import augment_batch
from weld_pipeline.surface_process_model import SurfaceProcessContactNet
from weld_pipeline.local_surface_model import LocalSurfaceProcessContactNet
from weld_pipeline.invariant_surface_model import InvariantSurfaceContactNet
from weld_pipeline.mapped_cache import load_piece as load_mapped_piece,STORAGE_FORMAT
from weld_pipeline.hard_negative_sampling import mine_training_negatives
from weld_pipeline.runtime_metrics import process_memory_mb
from weld_pipeline.structured_sampling import mine_training_structure,hard_heat_loss
from weld_pipeline.context_surface_model import ContextSurfaceContactNet
from weld_pipeline.context_augmentation import augment_context_batch
from weld_pipeline.frame_context_features import FEATURE_VERSION as CONTEXT_VERSION
from weld_pipeline.mixed_sequences import FEATURE_VERSION as MIXED_VERSION
from weld_pipeline.curved_sequences import decode_curved
from weld_pipeline.interval_decoder import decode_intervals
from weld_pipeline.curved_cache import load_piece as load_curved_piece,STORAGE_FORMAT as CURVED_STORAGE
from weld_pipeline.consistent_distillation import consistent_distillation
from weld_pipeline.model_registry_research import model_from_checkpoint as teacher_factory

SURFACE_MODELS=['surface_process','local_surface_process','invariant_surface_process','context_surface_process']
ROBUST_VERSIONS=['typed_contact_surface_v3',CONTEXT_VERSION,MIXED_VERSION]


def load(path):
    with Path(path).open('rb') as f:return torch.load(f,map_location='cpu')


def compact(score):
    return {k:v for k,v in score.items() if k not in ['per_truth','unmatched_prediction_indices']}


from weld_pipeline.partial_sequence_training import batch_sample,mine_training_structure


def loss_fn(logits,y,valid):
    counts=valid.sum((1,2)).clamp_min(1)
    mask=F.binary_cross_entropy_with_logits(logits[:,:1],y[:,:1],reduction='none')
    mask=((mask*valid).sum((1,2))/counts).mean()
    heat=F.binary_cross_entropy_with_logits(logits[:,1:5],y[:,1:5],reduction='none')
    heat=((heat*(1+25*y[:,1:5])*valid).sum((1,2))/(counts*4)).mean()
    aux=F.binary_cross_entropy_with_logits(logits[:,5:7],y[:,5:7],reduction='none')
    foreground=y[:,:1]*valid;num=foreground.sum((1,2)).clamp_min(1)
    aux=((aux*foreground).sum((1,2))/(num*2)).mean()
    return mask+3*heat+.3*aux


def focal_loss_fn(logits,y,valid):
    """Normalize sparse endpoint supervision by positives, not line length."""
    counts=valid.sum((1,2)).clamp_min(1)
    mask=F.binary_cross_entropy_with_logits(logits[:,:1],y[:,:1],reduction='none')
    mask=((mask*valid).sum((1,2))/counts).mean()
    h=logits[:,1:5].float();p=h.sigmoid();target=y[:,1:5]
    positive=(target>.65).float()*valid
    negative=(target<=.65).float()*valid
    pos_loss=(1-p).pow(2)*F.softplus(-h)*positive
    neg_loss=p.pow(2)*F.softplus(h)*(1-target).pow(4)*negative
    number=positive.sum((0,2)).clamp_min(1)
    heat=((pos_loss+neg_loss).sum((0,2))/number).mean()
    aux=F.binary_cross_entropy_with_logits(logits[:,5:7],y[:,5:7],reduction='none')
    foreground=y[:,:1]*valid;num=foreground.sum((1,2)).clamp_min(1)
    aux=((aux*foreground).sum((1,2))/(num*2)).mean()
    return mask+heat+.3*aux


def evaluate(model,data,out,epoch):
    result={}
    for name,piece in data.items():
        pred=decode_intervals(piece['sequences'],infer_probabilities(model,piece['sequences']),threshold=.5,weld_penalty=2.,mask_weight=1.)
        score=score_welds(pred,piece['truth']);result[name]=compact(score)
        dest=out/f'validation_epoch_{epoch:03d}';dest.mkdir(exist_ok=True)
        (dest/(name+'_predictions.json')).write_text(json.dumps(pred,indent=2),encoding='utf-8')
        (dest/(name+'_score.json')).write_text(json.dumps(score,indent=2),encoding='utf-8')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='weld_pipeline/data/typed_pilot_v2')
    ap.add_argument('--outdir',default='weld_pipeline/runs/typed_pilot_v2_seed93')
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--steps-per-epoch',type=int,default=48)
    ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--points',type=int,default=1024)
    ap.add_argument('--lr',type=float,default=.0007);ap.add_argument('--seed',type=int,default=93)
    ap.add_argument('--loss',choices=['bce','focal'],default='bce');ap.add_argument('--init')
    ap.add_argument('--model',choices=['typed','process_planes',*SURFACE_MODELS],default='typed')
    ap.add_argument('--augment-frames',action='store_true')
    ap.add_argument('--hard-negative-fraction',type=float,default=0.)
    ap.add_argument('--mine-every',type=int,default=20)
    ap.add_argument('--keep-init-planes',action='store_true')
    ap.add_argument('--bounded-neighbors',action='store_true')
    ap.add_argument('--mining-mode',choices=['mask','structured'],default='mask')
    ap.add_argument('--hard-heat-weight',type=float,default=0.)
    ap.add_argument('--fast-context',action='store_true')
    ap.add_argument('--curve-sampling-fraction',type=float,default=0.)
    ap.add_argument('--teacher',required=True);ap.add_argument('--teacher-weight',type=float,default=2.)
    ap.add_argument('--stop-after-epoch',type=int,default=0);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();root=Path(args.data);out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/'split.json').read_text(encoding='utf-8'))
    if split['feature_version'] in ROBUST_VERSIONS:
        readiness=json.loads((root/'preprocessing_complete.json').read_text(encoding='utf-8'))
        if readiness['status'] not in ['READY','READY_PARTIAL']:raise ValueError('Resolve representation gaps before training this dataset')
        if readiness['status']=='READY_PARTIAL' and readiness.get('required_trainer_capability')!='partial_supervision_v1':raise ValueError('Unknown partial supervision format')
        if readiness['status']=='READY_PARTIAL' and args.mining_mode!='structured':raise ValueError('Partial labels require masked structured mining')
    if args.bounded_neighbors and split['feature_version'] not in ROBUST_VERSIONS:
        raise ValueError('Bounded generator is validated for the robust surface feature version only')
    if not 0<=args.hard_negative_fraction<=.75 or args.mine_every<1:raise ValueError('Invalid mining configuration')
    if not 0<=args.hard_heat_weight<=1:raise ValueError('Invalid hard heat weight')
    if not 0<=args.curve_sampling_fraction<=.75:raise ValueError('Invalid curve sampling fraction')
    if args.curve_sampling_fraction and split['feature_version']!=MIXED_VERSION:raise ValueError('Curve sampling requires mixed features')
    if split['feature_version']==MIXED_VERSION and not args.bounded_neighbors:raise ValueError('Mixed generator requires bounded neighbors')
    if args.fast_context and args.model!='context_surface_process':raise ValueError('Fast context requires context model')
    context_equivalence=None
    if args.fast_context:
        receipt_path=Path('weld_pipeline/reports/context_feature_equivalence/receipt.json')
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['status']!='PASS':raise ValueError('Context equivalence missing')
        for name,wanted in receipt['sources'].items():
            if sha256(name)!=wanted:raise ValueError('Context equivalence source changed: '+name)
        context_equivalence=sha256(receipt_path)
    curved=split.get('storage_format')==CURVED_STORAGE
    mapped=split.get('storage_format') in [STORAGE_FORMAT,CURVED_STORAGE]
    def mapped_folder(name):
        folder=(root/split.get('piece_roots',{}).get(name,name)).resolve()
        if not folder.is_relative_to(Path.cwd().resolve()):raise ValueError('Mapped source is outside the project')
        return folder
    def read_piece(name):
        if curved:return load_curved_piece(mapped_folder(name),verify=True)
        return load_mapped_piece(mapped_folder(name),verify=True) if mapped else load(root/(name+'.pt'))
    train=[read_piece(n) for n in split['train']]
    validation={n:read_piece(n) for n in split['validation']}
    for p in train:
        p['positive_sequences']=[i for i,t in enumerate(p['targets']) if (t[:,0]>.5).any()]
        p['known_sequences']=[i for i,t in enumerate(p['targets']) if (t[:,0]>=0).any()]
        for t in p['targets']:
            if (t<0).any() and p['record'].get('target_policy')!='all_channel_minus_one_is_unknown':raise ValueError('Unknown negative target encoding')
        p['arc_sequences']=[i for i,s in enumerate(p['sequences']) if s.get('curve_kind')=='arc']
    sources=[Path(__file__),*[Path('weld_pipeline')/n for n in ['typed_contact_model.py','typed_sequences.py','surface_planes_local.py','surface_planes.py','geometry_candidates.py','evaluation.py']]]
    if split['feature_version'] in ROBUST_VERSIONS:
        sources.extend(Path('weld_pipeline')/n for n in ['robust_sequences.py','robust_surface_planes.py','point_frame.py'])
    if args.bounded_neighbors:sources.append(Path('weld_pipeline/bounded_sequences.py'))
    if args.model=='process_planes':sources.append(Path('weld_pipeline/process_plane_model.py'))
    if args.augment_frames:sources.append(Path('weld_pipeline/sequence_augmentation.py'))
    if args.model in SURFACE_MODELS:sources.extend([Path('weld_pipeline')/n for n in ['surface_process_model.py','surface_features.py','process_plane_model.py','model_registry.py']])
    if args.model in ['local_surface_process','invariant_surface_process']:sources.extend([Path('weld_pipeline')/n for n in ['local_surface_model.py','model_registry_extended.py']])
    if args.model=='invariant_surface_process':sources.extend([Path('weld_pipeline')/n for n in ['invariant_surface_model.py','model_registry_research.py']])
    if args.model=='context_surface_process':sources.extend([Path('weld_pipeline')/n for n in ['context_surface_model.py','context_augmentation.py','frame_context_features.py','indexed_surface_features.py','local_height_features.py','model_registry_context.py','model_registry_research.py']])
    if args.fast_context:sources.extend([Path('weld_pipeline/frame_context_features_fast.py'),Path('weld_pipeline/range_extrema.py')])
    if split['feature_version']=='typed_contact_surface_v2':sources.append(Path('weld_pipeline/bidirectional_surface_features.py'))
    sources.extend([Path('weld_pipeline/runtime_metrics.py')])
    if mapped:sources.append(Path('weld_pipeline/mapped_cache.py'))
    if split['feature_version']==MIXED_VERSION:
        sources.extend(Path('weld_pipeline')/n for n in ['mixed_sequences.py','curved_sequences.py','curved_cache.py',
            'arc_tangent_sequences.py','arc_surface_candidates.py','local_tangent_circles.py','tangent_circle_candidates.py','circle_candidates.py'])
    if args.hard_negative_fraction:sources.append(Path('weld_pipeline/hard_negative_sampling.py'))
    if args.mining_mode=='structured' or args.hard_heat_weight:sources.append(Path('weld_pipeline/structured_sampling.py'))
    sources.extend([Path('weld_pipeline/consistent_distillation.py'),Path('weld_pipeline/model_registry_research.py'),Path('weld_pipeline/interval_decoder.py'),Path('weld_pipeline/constrained_decoder.py'),Path('weld_pipeline/partial_supervision.py'),Path('weld_pipeline/partial_sequence_training.py')])
    hashes={str(p):sha256(p) for p in sources}
    data_hashes={n:sha256(mapped_folder(n)/'manifest.json') if mapped else sha256(root/(n+'.pt')) for n in split['train']+split['validation']}
    seed_all(args.seed);rng=np.random.default_rng(args.seed)
    features=train[0]['sequences'][0]['features'].shape[1]
    planes=estimate_training_planes(train) if args.model in ['process_planes',*SURFACE_MODELS] else []
    if args.keep_init_planes:
        if not args.init:raise ValueError('Keeping prior planes requires --init')
        initial_planes=load(args.init)
        planes=[{'axis':int(a),'position_mm':float(p),'origin':'prior training checkpoint'} for a,p in
            zip(initial_planes['model']['plane_axes'],initial_planes['model']['cut_positions_mm'])]
        del initial_planes
    if args.model=='context_surface_process':model=ContextSurfaceContactNet(features,planes).cuda()
    elif args.model=='invariant_surface_process':model=InvariantSurfaceContactNet(features,planes).cuda()
    elif args.model=='local_surface_process':model=LocalSurfaceProcessContactNet(features,planes).cuda()
    elif args.model=='surface_process':model=SurfaceProcessContactNet(features,planes).cuda()
    elif args.model=='process_planes':model=ProcessPlaneContactNet(features,planes).cuda()
    else:model=TypedContactNet(features).cuda()
    if args.model in SURFACE_MODELS:
        optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if n!='cut_positions_mm']},
            {'params':[model.cut_positions_mm],'weight_decay':0.}],lr=args.lr,weight_decay=.0001)
    else:optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    if args.init and not args.resume:
        initial=load(args.init)
        surface_extension=args.model=='surface_process' and initial.get('model_name')=='process_planes' and initial['features']==70 and initial['feature_version']=='typed_contact_sequence_v3'
        surface_geometry_extension=(args.model==initial.get('model_name') and args.model in ['surface_process','local_surface_process']
            and features==initial['features']==84 and initial['feature_version']=='typed_contact_surface_v1'
            and split['feature_version']=='typed_contact_surface_v3')
        context_extension=(args.model=='context_surface_process' and features==102 and initial['features']==84
            and initial.get('model_name')=='invariant_surface_process' and initial['feature_version']=='typed_contact_surface_v3')
        mixed_extension=(args.model==initial.get('model_name') and features==initial['features']==84
            and initial['feature_version']=='typed_contact_surface_v3' and split['feature_version']==MIXED_VERSION)
        if not (surface_extension or surface_geometry_extension or context_extension or mixed_extension) and (initial['features']!=features or initial['feature_version']!=split['feature_version']):raise ValueError('Initialization feature mismatch')
        if context_extension:model.load_base_state(initial['model'])
        elif surface_extension:model.load_process_state(initial['model'])
        elif args.model=='process_planes' and initial.get('model_name','typed')=='typed':model.load_base_state(initial['model'])
        else:model.load_state_dict(initial['model'])
        del initial
    teacher_state=load(args.teacher)
    if teacher_state['features']!=features or teacher_state.get('model_name')!='invariant_surface_process':raise ValueError('Teacher feature/model mismatch')
    teacher=teacher_factory(teacher_state).cuda().eval();del teacher_state
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    if not 0<=args.teacher_weight<=10:raise ValueError('Invalid distillation weight')
    scaler=torch.cuda.amp.GradScaler();first=1;best=None;elapsed_previous=0.
    config={'required_trainer_capability':'partial_supervision_v1','fixed_decoder_parameters':{'threshold':.5,'weld_penalty':2.,'mask_weight':1.},'decoder_policy':'Frozen from 109 TRAIN calibration before this partial-supervision experiment; no per-piece tuning.','augmentation_policy':split.get('augmentation_policy'),'arguments':{k:v for k,v in vars(args).items() if k not in ['resume','stop_after_epoch']},
        'source_hashes':hashes,'data_hashes':data_hashes,'split_sha256':sha256(root/'split.json'),
        'feature_channels':features,'parameters':sum(p.numel() for p in model.parameters()),
        'initial_checkpoint_sha256':sha256(args.init) if args.init else None,
        'model_name':args.model,'initial_shared_planes_from_training':planes,
        'teacher_checkpoint_sha256':sha256(args.teacher),'teacher_policy':'Training-label-consistent straight-input distillation only; teacher absent at inference.',
        'context_equivalence_receipt_sha256':context_equivalence,
        'plane_initialization_policy':'prior_training_checkpoint' if args.keep_init_planes else 'current_training_labels',
        'storage_format':split.get('storage_format','torch_pickle'),
        'scope':'Shared horizontal/vertical and arc development pilot; added arc piece is training data, not independent arc acceptance.' if split['feature_version']==MIXED_VERSION else 'Shared horizontal/vertical straight-line development pilot, not blind or arc acceptance.'}
    if args.resume:
        previous=json.loads((out/'config.json').read_text(encoding='utf-8'))
        if previous!=config:raise ValueError('Resume config/source/data differ')
        ckpt=load(out/'last.pt');model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler']);rng.bit_generator.state=ckpt['numpy_rng']
        torch.set_rng_state(ckpt['torch_rng']);torch.cuda.set_rng_state_all(ckpt['cuda_rng']);random.setstate(ckpt['python_rng'])
        first=ckpt['epoch']+1;best=ckpt['best_quality'];elapsed_previous=ckpt['elapsed_seconds']
        if args.hard_negative_fraction:
            for piece in train:piece['hard_samples']=ckpt['hard_samples'][piece['record']['name']]
        (out/'resume_receipt.json').write_text(json.dumps({'resumed_from_epoch':first-1,'next_epoch':first,
            'restored_optimizer_states':len(optimizer.state),'checkpoint_sha256':sha256(out/'last.pt')},indent=2),encoding='utf-8')
    else:
        if (out/'last.pt').exists() or (out/'epochs.jsonl').exists():raise ValueError('Existing run; resume or use a fresh directory')
        (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        snapshots=out/'source_snapshot';snapshots.mkdir(exist_ok=True)
        for p in sources:shutil.copy2(p,snapshots/p.name)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    for epoch in range(first,args.epochs+1):
        if args.hard_negative_fraction and (epoch==1 or (epoch-1)%args.mine_every==0):
            mining=(mine_training_structure if args.mining_mode=='structured' else mine_training_negatives)(model,train)
            (out/f'negative_mining_epoch_{epoch:03d}.json').write_text(json.dumps({'scope':'training groups only','groups':mining},indent=2),encoding='utf-8')
            print('NEGATIVE_MINING',epoch,sum(v['selected'] for v in mining.values()),flush=True)
        lr=args.lr*(.15+.85*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();losses=[]
        for step in range(args.steps_per_epoch):
            x,y,valid=batch_sample(train,rng,args.batch_size,args.points,args.augment_frames,args.hard_negative_fraction,args.curve_sampling_fraction);optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits=model(x)
                loss=(focal_loss_fn if args.loss=='focal' else loss_fn)(logits,y,valid)
                if args.hard_heat_weight:loss=loss+args.hard_heat_weight*hard_heat_loss(logits,y,valid)
                with torch.no_grad():teacher_logits=teacher(x)
                distillation=consistent_distillation(logits,teacher_logits,y,valid,x)
                loss=loss+args.teacher_weight*distillation
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);scaler.step(optimizer);scaler.update()
            losses.append(float(loss.detach()))
        rec={'epoch':epoch,'loss':float(np.mean(losses)),'lr':lr,
            'elapsed_seconds':elapsed_previous+time.perf_counter()-start,'cuda_peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,
            'process_memory_mb':process_memory_mb()}
        improved=False
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            scores=evaluate(model,validation,out,epoch);rec['validation']=scores
            macro_f1=float(np.mean([2*s['valid_welds']/max(1,s['truth_count']+s['prediction_count']) for s in scores.values()]))
            rec['strict_weld_macro_f1']=macro_f1
            quality=(all(s['all_gates_pass'] for s in scores.values()),
                macro_f1,
                float(np.mean([s['valid_extraction_rate'] for s in scores.values()])))
            if best is None or quality>tuple(best):best=quality;improved=True
        rec['elapsed_seconds']=elapsed_previous+time.perf_counter()-start
        rec['cuda_peak_allocated_mb']=torch.cuda.max_memory_allocated()/2**20
        checkpoint={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch,'features':features,'best_quality':best,'numpy_rng':rng.bit_generator.state,
            'model_name':args.model,
            'bounded_neighbors':args.bounded_neighbors,
            'fast_context':args.fast_context,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),
            'elapsed_seconds':rec['elapsed_seconds'],'source_hashes':hashes,'data_hashes':data_hashes,
            'feature_version':split['feature_version']}
        if args.hard_negative_fraction:checkpoint['hard_samples']={p['record']['name']:p.get('hard_samples',[]) for p in train}
        atomic_save(checkpoint,out/'last.pt')
        if improved:
            atomic_save(checkpoint,out/'best.pt');(out/'best_validation.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
        with (out/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
        brief={k:v for k,v in rec.items() if k!='validation'}
        if 'validation' in rec:brief['valid_welds']={n:[s['valid_welds'],s['truth_count'],s['prediction_count']] for n,s in rec['validation'].items()}
        print('EPOCH',json.dumps(brief),flush=True)
        if args.stop_after_epoch and epoch>=args.stop_after_epoch:
            print('CONTROLLED_CHECKPOINT_HANDOFF',epoch,flush=True);return
    (out/'complete.json').write_text(json.dumps({'completed_epochs':args.epochs,'best_quality':best,
        'elapsed_seconds':elapsed_previous+time.perf_counter()-start,'not_formal_acceptance':True},indent=2),encoding='utf-8')


if __name__=='__main__':main()
