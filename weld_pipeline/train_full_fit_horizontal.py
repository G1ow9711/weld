"""Full available-label horizontal refit. FIT scores are never blind results."""
import copy
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
from weld_pipeline.consistent_distillation import consistent_distillation
from weld_pipeline.contract_child_evaluation import aggregate,score_children
from weld_pipeline.contract_horizontal_model import from_checkpoint,MODEL_NAME
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.horizontal_training_views import horizontal_views
from weld_pipeline.partial_sequence_training_parent import batch_sample,mine_training_structure
from weld_pipeline.train_parent_balanced_branch import loss_fn
from weld_pipeline.structured_sampling import hard_heat_loss
from weld_pipeline.typed_contact_model import infer_probabilities


def evaluation(model,data,out,epoch,role):
    folder=out/(role+f'_epoch_{epoch:03d}')
    folder.mkdir(exist_ok=True)
    scores={}
    for name,piece in data.items():
        pred=[p for p in decode_intervals(piece['sequences'],infer_probabilities(model,piece['sequences']),
            threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.) if p['type']=='horizontal']
        truth=[w for w in piece['truth'] if w['type']=='horizontal']
        score=score_children(pred,truth)
        scores[name]=score
        (folder/(name+'.json')).write_text(json.dumps({'predictions':pred,'score':score},indent=2),encoding='utf8')
    summary=aggregate(scores)
    summary.update(role=role,independent_acceptance=False,full_contract_pass=False)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    return summary


def quality(summary):
    rates=[2*c['valid']/max(1,c['truth']+c['predictions']) for k,c in summary['categories'].items()
           if k in ['line','curve'] and c['truth']]
    return (summary['geometry_core_gates_pass'],min(rates),
            2*summary['valid_children']/max(1,summary['truth_count']+summary['prediction_count']),
            summary['valid_extraction_rate'])


def main():
    seed_all(128)
    torch.set_num_threads(4)
    root=Path('weld_pipeline/data/full_available_fit_v1')
    split=json.loads((root/'split.json').read_text('utf8'))
    if json.loads((root/'complete.json').read_text('utf8'))['status']!='READY_FULL_FIT_WITH_UNSEEN_CHECK':
        raise ValueError('FIT cohort incomplete')
    names=[n for n in split['train'] if '__' not in n]
    data={n:load_piece(split['piece_roots'][n],verify=True) for n in names}
    check={n:load_piece(split['piece_roots'][n],verify=True) for n in split['validation']}
    train,views=horizontal_views(list(data.values()))
    for p in train:
        p['positive_sequences']=[i for i,t in enumerate(p['targets']) if (t[:,0]>.5).any()]
        p['known_sequences']=list(range(len(p['sequences'])))
        p['arc_sequences']=[i for i,s in enumerate(p['sequences']) if s.get('curve_kind')=='arc']
    initial_path=Path('weld_pipeline/runs/typed_branch_surface_v3_seed115/best.pt')
    initial=torch.load(initial_path,map_location='cpu')
    verify_cached_inference_sources(initial,initial_path)
    model=from_checkpoint(initial,initialize_extent=True).cuda()
    teacher=copy.deepcopy(model).eval().requires_grad_(False)
    frozen={n:v.clone() for n,v in model.vertical.state_dict().items()}
    optimizer=torch.optim.AdamW(model.horizontal.parameters(),lr=4e-5,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler()
    rng=np.random.default_rng(128)
    out=Path('weld_pipeline/runs/contract_full_fit_horizontal_seed128')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():
        raise ValueError('Existing training run')
    paths=[__file__,'weld_pipeline/contract_horizontal_model.py','weld_pipeline/horizontal_training_views.py',
           'weld_pipeline/consistent_distillation.py','weld_pipeline/contract_child_evaluation.py']
    sources={**initial['source_hashes'],**{p:sha256(p) for p in paths}}
    hashes={n:sha256(Path(split['piece_roots'][n])/'manifest.json') for n in names+split['validation']}
    config={'seed':128,'epochs':120,'steps_per_epoch':96,'points':1024,'lr':4e-5,
            'data_hashes':hashes,'sources':sources,'split_sha256':sha256(root/'split.json'),
            'fit_sources':names,'unseen_check':split['validation'],'teacher_weight':2.,'hard_heat_weight':.1,
            'source_policy':split['role_policy'],'checkpoint_policy':'Best FIT H child quality including epoch zero. Unseen check never used for selection.',
            'views':views,'full_contract_pass':False}
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    archive=out/'source_snapshot';archive.mkdir(exist_ok=True)
    for path,wanted in sources.items():
        source=Path(path)
        if not source.exists() or sha256(source)!=wanted:
            source=initial_path.parent/'source_snapshot'/source.name
        if sha256(source)!=wanted:
            raise ValueError('Source snapshot absent')
        shutil.copy2(source,archive/source.name)
    start=time.perf_counter()
    best=None
    def state(epoch):
        return {'model_name':MODEL_NAME,'features':84,'feature_version':split['feature_version'],
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
            'source_hashes':sources,'data_hashes':hashes,'epoch':epoch,'best_quality':best,
            'split_sha256':config['split_sha256'],'evaluation_role':'FIT_NOT_INDEPENDENT',
            'elapsed_seconds':time.perf_counter()-start}
    baseline=evaluation(model,data,out,0,'FIT')
    best=quality(baseline)
    atomic_save(state(0),out/'best.pt')
    (out/'best_fit.json').write_text(json.dumps({'epoch':0,'summary':baseline},indent=2),encoding='utf8')
    print('FIT_BASELINE',baseline['valid_children'],'/',baseline['truth_count'],'FP',baseline['false_predictions'],flush=True)
    for epoch in range(1,121):
        if epoch==1 or (epoch-1)%20==0:
            mine_training_structure(model,train)
        lr=4e-5*(.15+.85*.5*(1+np.cos(np.pi*(epoch-1)/120.)))
        for group in optimizer.param_groups:
            group['lr']=lr
        model.train();losses=[]
        for step in range(96):
            x,y,valid=batch_sample(train,rng,8,1024,True,.35,.25)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits=model(x)
                with torch.no_grad():
                    reference=teacher(x)
                loss=loss_fn(logits,y,valid)+.1*hard_heat_loss(logits,y,valid)+2.*consistent_distillation(logits,reference,y,valid,x)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite refit loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.horizontal.parameters(),5.)
            scaler.step(optimizer);scaler.update()
            losses.append(float(loss.detach()))
        if any(not torch.equal(v,frozen[n]) for n,v in model.vertical.state_dict().items()):
            raise ValueError('Unused frozen vertical branch changed')
        row={'epoch':epoch,'loss':float(np.mean(losses)),'role':'FIT_NOT_INDEPENDENT',
             'elapsed_seconds':time.perf_counter()-start}
        improved=False
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
        print('FIT_EPOCH',json.dumps(row),flush=True)
    selected=torch.load(out/'best.pt',map_location='cpu')
    model=from_checkpoint(selected).cuda().eval()
    final_fit=evaluation(model,data,out,selected['epoch'],'FINAL_FIT')
    unseen=evaluation(model,check,out,selected['epoch'],'UNSEEN_CHECK')
    (out/'complete.json').write_text(json.dumps({'epochs':120,'selected_epoch':selected['epoch'],
        'elapsed_seconds':time.perf_counter()-start,'fit_summary':final_fit,'unseen_check_summary':unseen,
        'full_contract_pass':False,'checkpoint_sha256':sha256(out/'best.pt')},indent=2),encoding='utf8')


if __name__=='__main__':
    main()
