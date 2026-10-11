"""Direct supervised point-frame adaptation; unseen pieces never select."""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from training.labels import sha256
from training.pilot import seed_all
from training.scripts.train_pilot import atomic_save
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.horizontal_training_views import horizontal_views
from weld_pipeline.invariant_horizontal_context import context_sequences, FEATURES
from weld_pipeline.invariant_horizontal_model import from_checkpoint, MODEL_NAME
from weld_pipeline.partial_sequence_training_parent import sample_arrays, mine_training_structure
from weld_pipeline.sequence_augmentation import change_frame
from weld_pipeline.structured_sampling import hard_heat_loss
from weld_pipeline.train_full_fit_horizontal import evaluation, quality
from weld_pipeline.train_parent_balanced_branch import loss_fn


def sample_context(data, rng):
    x,y,valid=sample_arrays(data,rng,8,1024,False,.35,.25)
    for i in range(len(x)):
        core=x[i,:84].copy()
        reverse=rng.random()<.5
        change_frame(core,y[i],valid[i],reverse=reverse,flip_normal=rng.random()<.5)
        x[i,:84]=core
        if reverse:
            x[i,84:]=x[i,84:,::-1].copy()
    return [torch.from_numpy(a).cuda() for a in [x,y,valid]]


def main():
    seed_all(133);torch.set_num_threads(4)
    root=Path('weld_pipeline/data/full_available_fit_v1')
    split=json.loads((root/'split.json').read_text('utf8'))
    stats_root=Path('weld_pipeline/data/invariant_horizontal_context_v1')
    receipt=json.loads((stats_root/'complete.json').read_text('utf8'))
    if receipt['parent_split_sha256']!=sha256(root/'split.json'):
        raise ValueError('Statistics split mismatch')
    names=[n for n in split['train'] if '__' not in n]
    datasets={};hashes={};extra_hashes={}
    expected={r['name']:r['statistics_sha256'] for r in receipt['rows']}
    for name in names+split['validation']:
        p=load_piece(split['piece_roots'][name],verify=True)
        stats_path=stats_root/(name+'.npz')
        if sha256(stats_path)!=expected[name]:
            raise ValueError('Point statistics changed')
        with np.load(stats_path) as stats:
            mean,covariance=stats['mean'].copy(),stats['covariance'].copy()
        structure_path=Path(split['structure_roots'][name])/(name+'.npz')
        with np.load(structure_path) as structures:
            structural=structures['structural'].copy()
        p['sequences']=context_sequences(p['sequences'],structural,mean,covariance)
        datasets[name]=p
        hashes[name]=sha256(Path(split['piece_roots'][name])/'manifest.json')
        extra_hashes[name]={'statistics':sha256(stats_path),'structures':sha256(structure_path)}
    data={n:datasets[n] for n in names}
    check={n:datasets[n] for n in split['validation']}
    train,views=horizontal_views(list(data.values()))
    for p in train:
        p['positive_sequences']=[i for i,t in enumerate(p['targets']) if (t[:,0]>.5).any()]
        p['known_sequences']=list(range(len(p['sequences'])))
        p['arc_sequences']=[i for i,s in enumerate(p['sequences']) if s.get('curve_kind')=='arc']
    initial_path=Path('weld_pipeline/runs/invariant_horizontal_seed131/best.pt')
    initial=torch.load(initial_path,map_location='cpu')
    verify_cached_inference_sources(initial,initial_path)
    model=from_checkpoint(initial).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler();rng=np.random.default_rng(133)
    out=Path('weld_pipeline/runs/invariant_horizontal_direct_seed133');out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists() or (out/'best.pt').exists():
        raise ValueError('Existing invariant training run')
    paths=[__file__,'weld_pipeline/invariant_horizontal_model.py',
           'weld_pipeline/invariant_horizontal_context.py','weld_pipeline/point_plate_context.py',
           'weld_pipeline/train_full_fit_horizontal.py']
    sources={**initial['source_hashes'],**{p:sha256(p) for p in paths}}
    config={'seed':133,'epochs':120,'steps_per_epoch':96,'points':1024,'lr':3e-4,
        'data_hashes':hashes,'point_context_hashes':extra_hashes,'sources':sources,
        'split_sha256':sha256(root/'split.json'),'fit_sources':names,'unseen_check':split['validation'],
        'teacher_weight':0.,'hard_heat_weight':.1,'views':views,
        'checkpoint_policy':'Best FIT H child quality including selected 131 initialization; unseen pieces never select.',
        'initialization_equal_to_selected_131':True,
        'absolute_process_plane_features_removed':True,'gravity_axis_retained':True,
        'source_policy':split['role_policy'],'full_contract_pass':False}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for path,wanted in sources.items():
        source=Path(path)
        if not source.exists() or sha256(source)!=wanted:
            source=initial_path.parent/'source_snapshot'/source.name
        if sha256(source)!=wanted:
            raise ValueError('Required source snapshot absent: '+path)
        shutil.copy2(source,archive/source.name)
    start=time.perf_counter();best=None
    def state(epoch):
        return {'model_name':MODEL_NAME,'features':FEATURES,'feature_version':'point_frame_context_v1',
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'source_hashes':sources,'data_hashes':hashes,'point_context_hashes':extra_hashes,
            'epoch':epoch,'best_quality':best,'split_sha256':config['split_sha256'],
            'evaluation_role':'FIT_NOT_INDEPENDENT','elapsed_seconds':time.perf_counter()-start}
    baseline=evaluation(model,data,out,0,'FIT');best=quality(baseline)
    atomic_save(state(0),out/'best.pt')
    (out/'best_fit.json').write_text(json.dumps({'epoch':0,'summary':baseline},indent=2),encoding='utf8')
    print('INVARIANT_FIT_INITIAL',baseline['valid_children'],'/',baseline['truth_count'],
          'FP',baseline['false_predictions'],flush=True)
    for epoch in range(1,121):
        if epoch==1 or (epoch-1)%20==0:
            mine_training_structure(model,train)
        lr=3e-4*(.15+.85*.5*(1+np.cos(np.pi*(epoch-1)/120.)))
        for group in optimizer.param_groups:
            group['lr']=lr
        model.train();losses=[]
        for step in range(96):
            x,y,valid=sample_context(train,rng)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits=model(x)
                loss=loss_fn(logits,y,valid)+.1*hard_heat_loss(logits,y,valid)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite invariant refit loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        row={'epoch':epoch,'loss':float(np.mean(losses)),'role':'FIT_NOT_INDEPENDENT',
             'elapsed_seconds':time.perf_counter()-start};improved=False
        if epoch==1 or epoch%5==0:
            summary=evaluation(model,data,out,epoch,'FIT')
            row.update(valid=summary['valid_children'],truth=summary['truth_count'],false=summary['false_predictions'],
                       quality=quality(summary))
            if quality(summary)>tuple(best):
                best=quality(summary);improved=True
        atomic_save(state(epoch),out/'last.pt')
        if improved:
            atomic_save(state(epoch),out/'best.pt')
            (out/'best_fit.json').write_text(json.dumps({'epoch':epoch,'summary':summary},indent=2),encoding='utf8')
        with (out/'epochs.jsonl').open('a',encoding='utf8') as stream:
            stream.write(json.dumps(row)+'\n')
        print('INVARIANT_FIT_EPOCH',json.dumps(row),flush=True)
    selected=torch.load(out/'best.pt',map_location='cpu')
    model=from_checkpoint(selected).cuda().eval()
    fit=evaluation(model,data,out,selected['epoch'],'FINAL_FIT')
    unseen=evaluation(model,check,out,selected['epoch'],'UNSEEN_CHECK')
    (out/'complete.json').write_text(json.dumps({'epochs':120,'selected_epoch':selected['epoch'],
        'elapsed_seconds':time.perf_counter()-start,'fit_summary':fit,'unseen_check_summary':unseen,
        'full_contract_pass':False,'checkpoint_sha256':sha256(out/'best.pt')},indent=2),encoding='utf8')


if __name__=='__main__':
    main()
