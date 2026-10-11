"""Frozen-pipeline diagnostics and grouped validity readout checks; never deploy."""
import argparse
import collections
import gc
import pickle
import shutil
import time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_paired_density_validity_1283 import TRAIN,KEYS,COLUMNS,original,verify

DEST=OUT/'run447'

def prepare():
    verify();check_resources();DEST.mkdir(exist_ok=False)
    bindings={str(p.relative_to(OUT)):digest(p) for p in [
        OUT/'run429/sequence_model.pt',OUT/'run432/pair_model.pt',
        OUT/'run445/models.pkl',OUT/'run446/models.pkl']}
    assert len(bindings)==4
    snapshot=DEST/'source_snapshot';snapshot.mkdir()
    for name in ('sjg_conditional_diagnostics_1284.py','sjg_same_interval_diagnostic_1285.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,snapshot/name)
    write(DEST/'registered_protocol.json',dict(TRAIN=list(TRAIN),DEV=['64'],G_TRAIN=269,G_DEV=26,
        frozen_models=bindings,source_sha256=digest(__file__),budget_seconds=900,
        fit_calls=5,trees=64,max_depth=16,min_samples_leaf=2,max_features=.75,seed=1031,n_jobs=1,
        threshold=.5,actual_runtime_acceptance='strictly greater than 0.5; equality reported separately',
        original_and_even_density_weight='w/2 each; four-source original weights uniformly renormalized',
        withheld_source_label_read_only_after_its_predictions_saved=True,
        upstream_has_seen_heldout_labels=True,not_end_to_end_independent_validation=True,
        execution_order='DEV64 same-pair factory parity, five conditional folds, five TRAIN same-pair tables',
        feature_tables_stored_as_lossless_source_references_plus_indices_and_margin=True,
        unknown_background_excluded_from_ranking=True,
        no_deployment_or_ensemble=True,full_contract_pass=False))
    print('DIAGNOSTIC_PROTOCOL_REGISTERED',flush=True)

def labels(key):
    folder=OUT/f'run434/source_{key}'
    receipt=read(OUT/'run434/supervision_receipt.json')
    expected=next(c['labels_sha256'] for c in receipt['cases'] if c['source']==key)
    assert digest(folder/'training_targets.npz')==expected
    with np.load(folder/'training_targets.npz') as z:return z['validity'].copy()

def positive_coverage(key,y,w,x):
    c=read(OUT/f'run434/source_{key}/child_compatibility.json')
    base=read(OUT/f'run434/source_{key}/origin.json')['base_children']
    raw=read(OUT/f'run432/proposal_{key}/decoded.json')['parents']
    report=[]
    for kind in ('Hline','Vline','HArc','VArc'):
        ids=[i for i,p in enumerate(raw) if y[i]==1 and
             ('V' if x[i,525]>.5 else 'H')+('line' if p['segments'][0]['kind']=='line' else 'Arc')==kind]
        owners={tuple(c['original_parent_child_owners'][j]) for i in ids for j in c['compatible_targets'][base+i]}
        report.append(dict(source=key,category=kind,positive_alias_rows=len(ids),
            effective_positive_weight=float(w[ids].sum()),distinct_original_parents=len({p for p,ch in owners}),
            distinct_original_children=len(owners),parent_ids=sorted({p for p,ch in owners})))
    return report

def maximum_matching(edges,selected):
    assigned={}
    def visit(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in assigned or visit(assigned[j],seen):assigned[j]=i;return True
        return False
    for i in selected:visit(i,set())
    return assigned

def fold(held):
    check_resources();start=time.perf_counter();dest=DEST/f'fold_{held}';dest.mkdir(exist_ok=False)
    fit_keys=[k for k in TRAIN if k!=held];parts=[];ys=[];ws=[];coverage=[];bindings={}
    for key in fit_keys:
        x,r=original(key);y=labels(key);known=y>=0;classes=np.unique(y[known]);w=np.zeros(len(y))
        for cat in classes:w[y==cat]=1/(len(fit_keys)*len(classes)*np.sum(y==cat))
        coverage.extend(positive_coverage(key,y,w,x))
        ap=OUT/f'run444/source_{key}/even_view_90.npy';ar=read(ap.parent/'receipt.json')
        assert digest(ap)==ar['alternate_sha256'];alt=x.copy();alt[:,COLUMNS]=np.load(ap)
        parts.extend((x[known],alt[known]));ys.extend((y[known],y[known]));ws.extend((w[known]/2,w[known]/2))
        bindings[key]=dict(features=r['feature_sha256'],alternate=ar['alternate_sha256'],
                           labels=digest(OUT/f'run434/source_{key}/training_targets.npz'))
    xx=np.concatenate(parts);yy=np.concatenate(ys);ww=np.concatenate(ws);del parts,ys,ws;gc.collect()
    assert abs(ww.sum()-1)<1e-12 and held not in bindings
    model=ExtraTreesClassifier(n_estimators=64,max_depth=16,min_samples_leaf=2,max_features=.75,
                              bootstrap=False,random_state=1031,n_jobs=1,warm_start=False)
    print('LEAVE_WORKPIECE_OUT',held,'FIT_START','TREES',64,'ROWS',len(xx),flush=True)
    model.fit(xx,yy,sample_weight=ww);assert len(model.estimators_)==64
    atomic_binary_write(dest/'validity.pkl',lambda f:pickle.dump(model,f,protocol=5))
    # No held-out validity/compatibility labels have been read above.
    hx,hr=original(held);ha=OUT/f'run444/source_{held}/even_view_90.npy'
    assert digest(ha)==read(ha.parent/'receipt.json')['alternate_sha256']
    alt=hx.copy();alt[:,COLUMNS]=np.load(ha)
    positive=np.flatnonzero(model.classes_==1)
    pp=model.predict_proba(hx)[:,positive[0]] if len(positive) else np.zeros(len(hx))
    pq=model.predict_proba(alt)[:,positive[0]] if len(positive) else np.zeros(len(hx))
    np.savez(dest/'heldout_predictions.npz',original=pp,even=pq)
    write(dest/'prediction_receipt.json',dict(heldout=held,fit_sources=fit_keys,fit_bindings=bindings,
        model_sha256=digest(dest/'validity.pkl'),prediction_sha256=digest(dest/'heldout_predictions.npz'),
        heldout_features_sha256=hr['feature_sha256'],heldout_labels_read=False))
    y=labels(held);c=read(OUT/f'run434/source_{held}/child_compatibility.json')
    base=read(OUT/f'run434/source_{held}/origin.json')['base_children'];edges=c['compatible_targets'][base:]
    owners=c['original_parent_child_owners'];G=len(owners);all_ids=list(range(len(hx)))
    stages={}
    for name,ids in [('raw',all_ids),('retained',[i for i in all_ids if pp[i]>.5])]:
        match=maximum_matching(edges,ids)
        parent_children=collections.defaultdict(set)
        for j in match:p,ch=owners[j];parent_children[p].add(ch)
        totals=collections.Counter(p for p,ch in owners)
        stages[name]=dict(TP=len(match),FP=len(ids)-len(match),FN=G-len(match),G=G,
            parents_with_all_children_individually_matched=sum(len(ch)==totals[p] for p,ch in parent_children.items()),
            NOT_complete_parent_order_gap_closure_acceptance=True)
    result=dict(heldout=held,fit_sources=fit_keys,upstream_pretrained_on_heldout=True,
        conditional_validity_readout_only=True,fit_positive_coverage=coverage,
        known_positive_rows=int((y==1).sum()),positive_retained=int(((y==1)&(pp>.5)).sum()),
        known_negative_rows=int((y==0).sum()),known_negative_accepted=int(((y==0)&(pp>.5)).sum()),
        unknown_rows=int((y<0).sum()),unknown_accepted=int(((y<0)&(pp>.5)).sum()),
        exact_half_probability_rows=int((pp==.5).sum()),child_geometry_stages=stages,
        seconds=time.perf_counter()-start,fit_calls=1,full_contract_pass=False)
    write(dest/'evaluation.json',result)
    print('LEAVE_WORKPIECE_OUT',held,'FIT_COMPLETE','POS',result['positive_retained'],'/',result['known_positive_rows'],
          'WRONG_ACCEPTED',result['known_negative_accepted'],'/',result['known_negative_rows'],flush=True)
    del model,xx,hx,alt;gc.collect();reclaim_own_idle_pages();check_resources()

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','fold'],required=True)
    p.add_argument('--heldout',choices=TRAIN);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:fold(a.heldout)

if __name__=='__main__':main()
