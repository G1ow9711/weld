"""Frozen-upstream validity-only density pairing. No DEV fitting or tuning."""
import argparse,gc,pickle,shutil,time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_density_feature_runtime_1277 import DensityView,ENDPOINT_COLUMNS,interpolate_endpoint_view

TRAIN=('21','BM79A_native_local','10','25','36')
KEYS=(*TRAIN,'64')
DEST=OUT/'run444'
ARMS={'C':'run445','D':'run446'}
COLUMNS=np.r_[ENDPOINT_COLUMNS,ENDPOINT_COLUMNS+180]
RETAINED=np.setdiff1d(np.arange(526),COLUMNS)


def original(key):
    folder=OUT/f'run434/source_{key}';r=read(folder/'receipt.json')
    assert digest(folder/'features.npy')==r['feature_sha256']
    x=np.load(folder/'features.npy');assert x.shape[1]==526
    return x,r


def verify():
    p=read(DEST/'registered_protocol.json')
    for name,h in p['module_sha256'].items():assert digest(ROOT/'weld_pipeline'/name)==h
    assert digest(OUT/'run435/models.pkl')==p['frozen_models_sha256']
    for path,h in p['frozen_inputs'].items():assert digest(OUT/path)==h
    return p


def prepare():
    check_resources();DEST.mkdir(exist_ok=False);snapshot=DEST/'source_snapshot';snapshot.mkdir()
    names=('sjg_paired_density_validity_1283.py','sjg_density_feature_runtime_1277.py',
        'sjg_child_parent_model_1256.py','sjg_child_parent_supervision_1255.py',
        'sjg_additive_parent_precision_1254.py','sjg_global_neighborhood_sequence_1239.py')
    bindings={}
    for name in names:
        path=ROOT/'weld_pipeline'/name;bindings[name]=digest(path);shutil.copy2(path,snapshot/name)
    frozen={}
    for key in KEYS:
        for path in (f'run434/source_{key}/features.npy',f'run432/proposal_{key}/selected_intervals.json',
                     f'run432/proposal_{key}/decoded.json'):
            frozen[path]=digest(OUT/path)
    write(DEST/'registered_protocol.json',dict(TRAIN=list(TRAIN),DEV=['64'],G=269,module_sha256=bindings,
        frozen_models_sha256=digest(OUT/'run435/models.pkl'),frozen_inputs=frozen,
        arms=ARMS,treatment_prespecified='D',validity_only=True,changed_columns=COLUMNS.tolist(),
        unchanged_columns=436,duplicate_and_link_frozen=True,threshold=.5,trees=64,depth=16,
        min_samples_leaf=2,max_features=.75,seed=1031,n_jobs=1,one_complete_fit_per_arm=True,
        pair_weight='original weight split exactly into two halves',UNKNOWN_excluded_unchanged=True,
        check_sampling_seed=4431031,check_view_never_fit=True,budget_seconds=900,
        full_contract_pass=False))
    assert len(COLUMNS)==90 and len(RETAINED)==436
    cases=[]
    for key in TRAIN:
        check_resources();view=DensityView(key);x,r=original(key)
        seq=np.load(OUT/f'run421/sequence56/source_{key}/features.npy',mmap_mode='r')
        assert digest(OUT/f'run421/sequence56/source_{key}/features.npy')==read(OUT/f'run421/sequence56/source_{key}/receipt.json')['features_sha256']
        intervals=read(OUT/f'run432/proposal_{key}/selected_intervals.json')['intervals']
        assert len(intervals)==len(x);alternate=np.empty((len(x),90),np.float32);maximum_original_error=0.
        for i,v in enumerate(intervals):
            j=view.lookup[v['row']//2];a,b=map(int,view.mapping['offsets'][j:j+2])
            length=float(view.mapping['q_mm'][b-1]);closed=bool(seq[a,27]>.5)
            z=view.endpoint(v['row'],np.array([v['lo_mm'],v['hi_mm']]),length,closed)
            alternate[i]=np.r_[z[0],z[3]]
            reference=interpolate_endpoint_view(view.mapping['q_mm'][a:b],seq[a:b,28:43],
                np.array([v['lo_mm'],v['hi_mm']]),length,closed)
            error=float(abs(np.r_[reference[0],reference[3]]-x[i,COLUMNS]).max())
            maximum_original_error=max(maximum_original_error,error);assert error<3e-5,(key,i,error)
        assert np.isfinite(alternate).all();changed=x.copy();changed[:,COLUMNS]=alternate
        assert np.array_equal(x[:,RETAINED],changed[:,RETAINED])
        folder=DEST/f'source_{key}';folder.mkdir();np.save(folder/'even_view_90.npy',alternate)
        c=dict(source=key,rows=len(x),original_features_sha256=r['feature_sha256'],
            alternate_sha256=digest(folder/'even_view_90.npy'),density_source_sha256=view.receipt['alternate_sha256'],
            all_other_436_columns_exact=True,original_90_column_reconstruction_max_abs=maximum_original_error)
        cases.append(c);write(folder/'receipt.json',c)
        seq._mmap.close();del x,seq,view,alternate,changed;gc.collect()
        print('PAIRED_DENSITY_FEATURES_READY',key,c['rows'],flush=True)
    write(DEST/'feature_receipt.json',dict(status='FIVE_TRAIN_PAIRED_FEATURES_READY',cases=cases,
        GT_reads=0,STEP_reads=0,DEV_reads=0,not_full_resampled_pipeline=True))
    prepare_check()


def prepare_check():
    # Same 163 candidates, with an independently drawn fixed random half-view.
    from scipy.spatial import cKDTree
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    from weld_pipeline.sjg_global_neighborhood_sequence_1239 import neighborhood
    old=read(OUT/'run435/sampling_sensitivity.json');cohort=read(OUT/'run98/local_cohort.json');reports=[]
    for case in old['cases']:
        check_resources();key=case['source'];assert key in TRAIN
        indices=np.array([v['candidate'] for v in case['rows']],np.int64);x,_=original(key)
        original_x=x[indices].copy();changed=original_x.copy()
        intervals=read(OUT/f'run432/proposal_{key}/selected_intervals.json')['intervals']
        binding=next(c for c in cohort['cases'] if c['source_key']==key)
        path=OUT/binding['input_file'];assert digest(path)==binding['input_sha256']
        xyz=np.load(path,mmap_mode='r');rng=np.random.default_rng(4431031)
        selected=np.sort(rng.choice(len(xyz),len(xyz)//2,replace=False));thin=np.asarray(xyz[selected])
        np.save(DEST/f'source_{key}/random_check_point_indices.npy',selected)
        tree=cKDTree(thin);loaded=load_case(key);g=loaded['graph'];loaded['x']._mmap.close()
        with np.load(OUT/f'run421/sequence56/source_{key}/sequence_map.npz') as z:m={k:z[k] for k in z.files}
        lookup={int(pid):j for j,pid in enumerate(m['profile_ids'])};cache={}
        for local,index in enumerate(indices):
            v=intervals[index];pid=v['row']//2;curve=g.curve(pid)
            if pid not in cache:
                j=lookup[pid];a,b=map(int,m['offsets'][j:j+2]);grid=m['q_mm'][a:b]
                value=neighborhood(tree,thin,curve.at(grid),curve.tangent(grid),grid,np.full(len(grid),curve.length))[:,:15]
                cache[pid]=(grid,value)
            grid,value=cache[pid];closed=curve.kind=='arc' and curve.arc.is_closed
            for endpoint,center,inward in ((0,v['lo_mm'],1),(180,v['hi_mm'],-1)):
                q=np.array([center,center+inward*50,center-inward*50])
                q=np.mod(q,curve.length) if closed else np.clip(q,0,curve.length)
                interpolated=np.column_stack([np.interp(q,grid,value[:,j]) for j in range(15)])
                for slot in range(3):changed[local,endpoint+slot*56+28:endpoint+slot*56+43]=interpolated[slot]
        assert np.array_equal(changed[:,RETAINED],original_x[:,RETAINED])
        folder=DEST/f'source_{key}';np.savez(folder/'random_check.npz',indices=indices,x=changed)
        reports.append(dict(source=key,candidates=len(indices),original_input_sha256=binding['input_sha256'],
            selected_indices_sha256=digest(folder/'random_check_point_indices.npy'),
            check_sha256=digest(folder/'random_check.npz'),seed=4431031,not_used_for_fitting=True))
        xyz._mmap.close();del xyz,thin,tree,loaded,g,x,cache,changed;gc.collect()
        print('UNSEEN_RANDOM_VIEW_CHECK_READY',key,len(indices),flush=True)
    assert sum(r['candidates'] for r in reports)==163
    write(DEST/'check_receipt.json',dict(status='RANDOM_CHECK_FROZEN',cases=reports,fit_calls=0,DEV_reads=0))


def fit(arm):
    p=verify();check_resources();dest=OUT/ARMS[arm];dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(shared_protocol_sha256=digest(DEST/'registered_protocol.json'),
        arm=arm,treatment_prespecified='D',no_selection_on_DEV=True,full_contract_pass=False))
    labels=read(OUT/'run434/supervision_receipt.json');old=read(OUT/'run435/training_receipt.json')
    assert labels['G']==269 and set(c['source'] for c in labels['cases'])==set(TRAIN)
    counts=[]
    for c in labels['cases']:
        path=OUT/f"run434/source_{c['source']}/training_targets.npz";assert digest(path)==c['labels_sha256']
        with np.load(path) as z:counts.append(int((z['validity']>=0).sum()))
    total=2*sum(counts);xf=dest/'expanded_training.npy'
    xall=np.lib.format.open_memmap(xf,mode='w+',dtype=np.float32,shape=(total,526))
    yall=np.empty(total,np.int8);wall=np.empty(total,np.float64);offset=0;weights=[]
    for c in labels['cases']:
        key=c['source'];x,fr=original(key)
        assert fr['feature_sha256']==old['bindings'][key]['features']
        with np.load(OUT/f'run434/source_{key}/training_targets.npz') as z:y=z['validity'].copy()
        mask=y>=0;x=x[mask];y=y[mask];classes=np.unique(y);w=np.zeros(len(y))
        for cat in classes:w[y==cat]=1/(len(TRAIN)*len(classes)*int((y==cat).sum()))
        n=len(y);half=w/2;alternate=x.copy()
        if arm=='D':
            path=DEST/f'source_{key}/even_view_90.npy';receipt=read(path.parent/'receipt.json')
            assert digest(path)==receipt['alternate_sha256']
            alternate[:,COLUMNS]=np.load(path)[mask]
        assert np.array_equal(x[:,RETAINED],alternate[:,RETAINED])
        xall[offset:offset+n]=x;xall[offset+n:offset+2*n]=alternate
        yall[offset:offset+2*n]=np.r_[y,y];wall[offset:offset+2*n]=np.r_[half,half]
        assert np.array_equal(half+half,w)
        weights.append(dict(source=key,known_rows=n,mass=float(w.sum()),duplicate_mass=float((half.sum())*2)))
        offset+=2*n;del x,alternate;gc.collect()
    assert offset==total;xall.flush();assert np.isfinite(xall).all()
    assert abs(wall.sum()-1)<1e-12
    model=ExtraTreesClassifier(n_estimators=64,max_depth=16,min_samples_leaf=2,max_features=.75,
        bootstrap=False,random_state=1031,n_jobs=1,warm_start=False)
    begin=time.perf_counter();print('FOREST_VALIDITY',arm,'FIT_START','ROWS',total,'TREES',64,flush=True)
    model.fit(xall,yall,sample_weight=wall)
    print('FOREST_VALIDITY',arm,'FIT_COMPLETE','TREES',len(model.estimators_),flush=True)
    assert len(model.estimators_)==64
    with (OUT/'run435/models.pkl').open('rb') as f:payload=pickle.load(f)
    frozen={k:pickle.dumps(payload['models'][k],protocol=5) for k in ('duplicate','link')}
    payload['models']['validity']=model
    assert payload['input_run']=='run434' and payload['threshold']==.5
    for k,v in frozen.items():assert pickle.dumps(payload['models'][k],protocol=5)==v
    atomic_binary_write(dest/'models.pkl',lambda f:pickle.dump(payload,f,protocol=5))
    with (dest/'models.pkl').open('rb') as f:reloaded=pickle.load(f)
    assert np.array_equal(model.predict_proba(xall[:32]),reloaded['models']['validity'].predict_proba(xall[:32]))
    write(dest/'training_receipt.json',dict(old,model_sha256=digest(dest/'models.pkl'),arm=arm,
        traces=dict(validity=[dict(fit_call=1,trees=64,seconds=time.perf_counter()-begin)],
                    duplicate='FROZEN_FROM_RUN435',link='FROZEN_FROM_RUN435'),
        validity_retrained_only=True,duplicate_and_link_byte_identical=True,one_complete_fit=True,
        source_model_sha256=p['frozen_models_sha256'],paired_weight_audit=weights,expanded_rows=total,
        input_run='run434',density_check_used_for_fitting=False,seconds=time.perf_counter()-begin,
        full_contract_pass=False))
    xall._mmap.close();del xall,model,payload,reloaded;gc.collect()


def check():
    verify();results=[]
    cr=read(DEST/'check_receipt.json')
    for arm,run in ARMS.items():
        receipt=read(OUT/run/'training_receipt.json');assert digest(OUT/run/'models.pkl')==receipt['model_sha256']
        with (OUT/run/'models.pkl').open('rb') as f:model=pickle.load(f)['models']['validity']
        for case in cr['cases']:
            key=case['source'];path=DEST/f'source_{key}/random_check.npz';assert digest(path)==case['check_sha256']
            with np.load(path) as z:indices=z['indices'];changed=z['x']
            x,_=original(key);original_x=x[indices]
            with np.load(OUT/f'run434/source_{key}/training_targets.npz') as z:y=z['validity'][indices]
            pp=model.predict_proba(original_x)[:,1];pq=model.predict_proba(changed)[:,1]
            for vertical in (False,True):
                for label in (0,1):
                    mask=(y==label)&((original_x[:,525]>.5)==vertical)
                    if not mask.any():continue
                    results.append(dict(arm=arm,source=key,vertical=vertical,label=label,count=int(mask.sum()),
                        original_accepted=int((mask&(pp>.5)).sum()),check_accepted=int((mask&(pq>.5)).sum()),
                        accepted_to_rejected=int((mask&(pp>.5)&(pq<=.5)).sum()),
                        rejected_to_accepted=int((mask&(pp<=.5)&(pq>.5)).sum()),
                        mean_abs_probability_change=float(abs(pp[mask]-pq[mask]).mean()),
                        max_abs_probability_change=float(abs(pp[mask]-pq[mask]).max())))
            write(DEST/f'source_{key}/check_probabilities_{arm}.json',dict(indices=indices.tolist(),
                labels=y.tolist(),original_probabilities=pp.tolist(),random_check_probabilities=pq.tolist()))
        del model;gc.collect()
    totals={}
    for arm in ARMS:
        rows=[r for r in results if r['arm']==arm]
        totals[arm]=dict(positive_accepted_to_rejected=sum(r['accepted_to_rejected'] for r in rows if r['label']==1),
            original_incompatible_accepted=sum(r['original_accepted'] for r in rows if r['label']==0),
            check_incompatible_accepted=sum(r['check_accepted'] for r in rows if r['label']==0),
            original_positive_accepted=sum(r['original_accepted'] for r in rows if r['label']==1),
            check_positive_accepted=sum(r['check_accepted'] for r in rows if r['label']==1))
    write(DEST/'unseen_sampling_check.json',dict(status='COMPLETE',cases=results,totals=totals,
        candidates_per_arm=163,TRAIN_perturbation_not_cross_workpiece_validation=True,full_contract_pass=False))
    print('UNSEEN_SAMPLING_CHECK',totals,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','train','check','infer'],required=True)
    ap.add_argument('--arm',choices=list(ARMS));ap.add_argument('--source',choices=list(KEYS));a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='train':fit(a.arm)
    elif a.stage=='check':check()
    else:
        verify()
        from weld_pipeline.sjg_child_parent_model_1256 import infer
        infer(OUT/ARMS[a.arm],a.source)


if __name__=='__main__':main()
