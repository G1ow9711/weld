"""Fixed source-held-out material/support ablation; no observed-check reads."""
import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

from training.labels import sha256
from weld_pipeline.material_support_ablation import side_sorted
from weld_pipeline.signed_material_support import frame_views, predict_equivalent_views

DATA = Path('weld_pipeline/data/material_support_v1')
OUT = Path('weld_pipeline/runs/material_support_crossfit_seed148')
SEEDS = (148,149,150)


def balanced_weights(y, names, source_rows):
    w = np.zeros(len(y),float)
    for name in names:
        rows = np.arange(*source_rows[name])
        known = rows[y[rows]>=0]
        classes = np.unique(y[known])
        for c in classes:
            subset = known[y[known]==c]
            w[subset] = 1./(len(names)*len(classes)*len(subset))
    return w


def fit_head(x,y,weights,representation,seed):
    indices=np.flatnonzero(weights>0)
    if representation=='signed':
        augmented=np.concatenate(frame_views(x[indices]))
    else:
        # Match eight training views and leaf sample counts without changing the ablation values.
        augmented=np.tile(x[indices],(8,1))
    labels=np.tile(y[indices],8);w=np.tile(weights[indices]/8.,8)
    model=ExtraTreesClassifier(n_estimators=128,max_features=.5,min_samples_leaf=2,
                              max_depth=None,n_jobs=2,random_state=seed)
    model.fit(augmented,labels,sample_weight=w)
    del augmented,labels,w;gc.collect()
    return model


def score_probability(p,y,classes):
    known=y>=0
    predictions=np.asarray(classes)[np.argmax(p,axis=1)]
    cm=confusion_matrix(y[known],predictions[known],labels=classes)
    counts=cm.sum(1)
    recall=np.divide(np.diag(cm),counts,out=np.zeros(len(classes),float),where=counts>0)
    result=dict(rows=len(y),known_rows=int(known.sum()),unknown_rows=int((~known).sum()),
                confusion=cm.tolist(),class_counts=counts.tolist(),class_recall=recall.tolist(),
                macro_recall=float(recall[counts>0].mean()),correct=int(np.diag(cm).sum()),
                errors=int(cm.sum()-np.diag(cm).sum()),sample_classification_only=True,
                geometry_children_not_scored=True,full_contract_pass=False)
    if list(classes)==[0,1]:
        tn,fp,fn,tp=cm.ravel();g=tp+fn
        result.update(foreground=int(g),true_foreground=int(tp),false_foreground=int(fp),missed_foreground=int(fn),
                      tp_over_foreground=float(tp/g) if g else None,
                      fp_over_foreground=float(fp/g) if g else None,
                      foreground_sample_gate=bool(g and tp/g>.95 and fp/g<=.05))
    return result


def predict(model,x,representation):
    return predict_equivalent_views(model,x) if representation=='signed' else model.predict_proba(x)


def main():
    if OUT.exists():raise ValueError('Preserve fixed material classifier trial')
    started=time.perf_counter()
    data_config=json.loads((DATA/'config.json').read_text('utf8'))
    complete=json.loads((DATA/'complete.json').read_text('utf8'))
    if complete['config_sha256']!=sha256(DATA/'config.json'):
        raise ValueError('Material data config changed')
    for source,digest in data_config['sources'].items():
        if sha256(source)!=digest:raise ValueError('PCD preparation source changed')
    names=data_config['pieces'];signed=[];sorted_x=[];material=[];business=[];source_rows={};row=0
    for name in names:
        root=DATA/name
        if sha256(root/'manifest.json')!=complete['manifests'][name]:raise ValueError('Prepared source manifest changed')
        manifest=json.loads((root/'manifest.json').read_text('utf8'))
        for file,digest in manifest['files'].items():
            if sha256(root/file)!=digest:raise ValueError('Prepared material feature/label array changed')
        a=np.load(root/'signed.npy',allow_pickle=False);b=np.load(root/'side_sorted.npy',allow_pickle=False)
        with np.load(root/'labels.npz',allow_pickle=False) as labels:
            material.append(labels['material']);business.append(labels['business'])
        signed.append(a);sorted_x.append(b);source_rows[name]=(row,row+len(a));row+=len(a)
    x={'signed':np.concatenate(signed),'side_sorted':np.concatenate(sorted_x)}
    y={'material':np.concatenate(material),'business':np.concatenate(business)}
    # Verify that the control loses sign/order but still obeys the same equivalent-frame symmetry.
    for view in frame_views(x['signed'][::max(1,len(x['signed'])//64)]):
        expected=x['side_sorted'][::max(1,len(x['signed'])//64)]
        if not np.array_equal(side_sorted(view),expected):raise ValueError('Control is not frame invariant')
    OUT.mkdir(parents=True)
    sources=[__file__,'weld_pipeline/signed_material_support.py','weld_pipeline/material_support_ablation.py']
    config=dict(data_config_sha256=sha256(DATA/'config.json'),data_complete_sha256=sha256(DATA/'complete.json'),
        pieces=names,source_rows=source_rows,folds=data_config['folds'],seeds=list(SEEDS),
        source_sha256={p:sha256(p) for p in sources},n_estimators=128,max_features=.5,min_samples_leaf=2,
        max_depth=None,source_and_class_balance=True,equivalent_frames=8,
        targets_separate=['unreviewed_material_hypothesis','unchanged_original_welding_foreground'],
        fixed_business_threshold=.5,classification_not_geometry=True,
        adoption='Each seed business FN+FP reduction >=20%, TP not lower, FP not higher, majority sources improve; FIT foreground sample gates required.',
        observed_check_read=False,full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf8')
    snapshot=OUT/'source_snapshot';snapshot.mkdir()
    for source in sources:shutil.copy2(source,snapshot/Path(source).name)
    capacity={};checkpoints={};out_of_fold={};fold_scores={}
    for seed in SEEDS:
        capacity[str(seed)]={};out_of_fold[str(seed)]={r:{target:np.full((row,3 if target=='material' else 2),np.nan)
                                                     for target in y} for r in x}
        for representation in x:
            capacity[str(seed)][representation]={}
            for target,labels in y.items():
                weights=balanced_weights(labels,names,source_rows)
                model=fit_head(x[representation],labels,weights,representation,seed)
                path=OUT/(f'full_fit_seed{seed}_{representation}_{target}.joblib');joblib.dump(model,path)
                checkpoints[path.name]=sha256(path)
                p=predict(model,x[representation],representation)
                score=score_probability(p,labels,model.classes_)
                capacity[str(seed)][representation][target]=score
                np.save(OUT/(path.stem+'_fit_probabilities.npy'),p.astype(np.float32))
                print('MATERIAL_CAPACITY',seed,representation,target,'ERRORS',score['errors'],
                      'RECALL',score['class_recall'],flush=True)
                del model;gc.collect()
    (OUT/'capacity.json').write_text(json.dumps(capacity,indent=2),encoding='utf8')
    for fold in data_config['folds']:
        fi=fold['fold'];train=[n for n in fold['train_real'] if n in source_rows]
        held=[n for n in fold['held_out_real'] if n in source_rows]
        if set(train)&set(held) or set(train+held)!=set(names):raise ValueError('Source partition invalid')
        held_indices=np.concatenate([np.arange(*source_rows[n]) for n in held])
        for seed in SEEDS:
            for representation in x:
                for target,labels in y.items():
                    weights=balanced_weights(labels,train,source_rows)
                    model=fit_head(x[representation],labels,weights,representation,seed)
                    path=OUT/(f'fold{fi}_seed{seed}_{representation}_{target}.joblib');joblib.dump(model,path)
                    checkpoints[path.name]=sha256(path)
                    if len(model.classes_)!=(3 if target=='material' else 2):raise ValueError('Training fold lacks target class')
                    p=predict(model,x[representation][held_indices],representation)
                    out_of_fold[str(seed)][representation][target][held_indices]=p
                    score=score_probability(p,labels[held_indices],model.classes_)
                    fold_scores[path.stem]=dict(train=train,held=held,score=score,
                        train_rows=int((weights>0).sum()),held_rows=len(held_indices))
                    print('MATERIAL_SOURCE_HELD',fi,seed,representation,target,'ERRORS',score['errors'],
                          'RECALL',score['class_recall'],flush=True)
                    del model;gc.collect()
    aggregates={};decisions={}
    for seed in SEEDS:
        aggregates[str(seed)]={}
        by_source={r:{} for r in x}
        for representation in x:
            aggregates[str(seed)][representation]={}
            for target,labels in y.items():
                p=out_of_fold[str(seed)][representation][target]
                if not np.isfinite(p).all():raise ValueError('Incomplete source-held-out predictions')
                classes=np.arange(p.shape[1]);score=score_probability(p,labels,classes)
                aggregates[str(seed)][representation][target]=score
                np.save(OUT/(f'oof_seed{seed}_{representation}_{target}.npy'),p.astype(np.float32))
                for name in names:
                    a,b=source_rows[name]
                    by_source[representation].setdefault(name,{})[target]=score_probability(p[a:b],labels[a:b],classes)
        control=aggregates[str(seed)]['side_sorted']['business'];new=aggregates[str(seed)]['signed']['business']
        reduction=(control['errors']-new['errors'])/max(control['errors'],1)
        improvements=sum(by_source['signed'][n]['business']['errors']<by_source['side_sorted'][n]['business']['errors'] for n in names)
        fit_gate=capacity[str(seed)]['signed']['business']['foreground_sample_gate']
        passed=bool(fit_gate and reduction>=.2 and improvements>len(names)/2 and
                    new['true_foreground']>=control['true_foreground'] and new['false_foreground']<=control['false_foreground'])
        decisions[str(seed)]=dict(error_reduction=reduction,improved_sources=improvements,total_sources=len(names),
                                  signed_fit_foreground_sample_gate=fit_gate,source_gate_pass=passed)
        (OUT/(f'by_source_seed{seed}.json')).write_text(json.dumps(by_source,indent=2,ensure_ascii=False),encoding='utf8')
        print('MATERIAL_SOURCE_DECISION',seed,decisions[str(seed)],flush=True)
    eligible=all(d['source_gate_pass'] for d in decisions.values())
    complete=dict(config_sha256=sha256(OUT/'config.json'),capacity=capacity,source_held_aggregate=aggregates,
        source_decisions=decisions,eligible_for_later_geometry_trial=eligible,fold_scores=fold_scores,
        checkpoint_sha256=checkpoints,elapsed_seconds=time.perf_counter()-started,
        observed_check_read=False,material_truth_reviewed=False,geometry_not_scored=True,full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(complete,indent=2,ensure_ascii=False),encoding='utf8')
    print('MATERIAL_SOURCE_TRIAL_COMPLETE','ELIGIBLE',eligible,'NOT_CONTRACT_ACCEPTANCE',flush=True)


if __name__=='__main__':
    main()
