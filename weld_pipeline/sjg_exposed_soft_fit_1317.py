"""Original full-pool risk plus fixed full-domain exposed-error sums."""
import gc,json,pickle,time,traceback
import numpy as np
from scipy.optimize import minimize
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,digest,check_resources,reclaim_own_idle_pages
from weld_pipeline.sjg_owned_continuation_watchdog_1295 import atomic_json
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write

DEST=OUT/'run457';POOL=OUT/'run454'
TRAIN=('21','BM79A_native_local','10','25','36');BLOCK=4096;REG=1e-4

def event(name,**values):
    v=dict(event=name,monotonic=time.monotonic(),**values)
    with (DEST/'fit_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
    print('RUN457',name,values,flush=True)

def save_npz(path,**values):
    atomic_binary_write(path,lambda f:np.savez(f,**values))

def guard(deadline):
    if time.monotonic()>deadline-15:raise TimeoutError('Original preparation/control/fit deadline, 15s save reserve')

def chunks(p,deadline):
    offset=0
    for xf,bf in p.get('file_pairs',[('features.npy','base_margins.npy'),('extra_features.npy','extra_base.npy')]):
        x=np.load(p['folder']/xf,mmap_mode='r');b=np.load(p['folder']/bf,mmap_mode='r')
        try:
            assert x.shape==(len(b),605) and b.shape[1]==4
            for a in range(0,len(x),BLOCK):
                guard(deadline);z=a+min(BLOCK,len(x)-a)
                yield offset+a,offset+z,np.asarray(x[a:z],np.float64),np.asarray(b[a:z],np.float64)
            offset+=len(x)
        finally:x._mmap.close();b._mmap.close()
    assert offset==p['n']

def weights(n,records,positives,G):
    """Five equal pieces, half positive, half negative carrier-type group mean."""
    wp=np.zeros((n,4));wn=np.zeros((n,4));seen=np.zeros(n,bool)
    ng=sum(int(r['known_wrong'][:,c].any()) for r in records for c in range(4))
    assert ng>0
    negative_count=0
    for r in records:
        ids=np.asarray(r['indices']);assert not seen[ids].any();seen[ids]=True
        for c in range(4):
            mask=r['known_wrong'][:,c];k=int(mask.sum());negative_count+=k
            if k:wn[ids[mask],c]=.1/(ng*k)
        for (i,c),owners in r['compat'].items():
            if owners:assert not r['known_wrong'][i,c]
    assert seen.all() and len(positives)==G
    identities=set()
    for v in positives:
        i=v['pool_index'];c=v['native_identity'][-1]-1
        assert (i,c) not in identities and wn[i,c]==0;identities.add((i,c));wp[i,c]+=.1/G
    assert abs(wp.sum()-.1)<1e-12 and abs(wn.sum()-.1)<1e-12
    return wp,wn,dict(positive_count=G,negative_count=negative_count,negative_groups=ng,
                     positive_weight_sum=float(wp.sum()),negative_weight_sum=float(wn.sum()))

def block_loss(x,anchor_score,wp,wn,beta,d):
    scaled=x/d;s=anchor_score+scaled@beta.T
    hp=np.maximum(0,1-s);hn=np.maximum(0,1+s)
    loss=.5*float(np.sum(wp*hp*hp+wn*hn*hn))
    gradient=(-wp*hp+wn*hn).T@scaled
    return loss,gradient,s

def controls(samples,anchor,d):
    x=np.concatenate([v[0] for v in samples]);a=np.concatenate([v[1] for v in samples])
    wp=np.concatenate([v[2] for v in samples]);wn=np.concatenate([v[3] for v in samples])
    base=np.concatenate([v[4] for v in samples]);rng=np.random.default_rng(456)
    beta=rng.normal(0,.01,anchor.shape)
    loss,gradient,score=block_loss(x,a,wp,wn,beta,d)
    dense=base+x@(anchor+beta/d).T
    zero=block_loss(x,a,wp,wn,np.zeros_like(beta),d)[2]
    export_error=float(abs(dense-score).max());zero_error=float(abs(zero-(base+x@anchor.T)).max())
    assert export_error<1e-9 and zero_error<1e-9
    total=0.;grad=np.zeros_like(beta)
    for start in range(0,len(x),7):
        sl=slice(start,start+7);v,g,_=block_loss(x[sl],a[sl],wp[sl],wn[sl],beta,d);total+=v;grad+=g
    assert abs(total-loss)<1e-9 and abs(grad-gradient).max()<1e-9
    def independent(b):
        s=base+x@(anchor+b/d).T
        return .5*float((wp*np.maximum(0,1-s)**2+wn*np.maximum(0,1+s)**2).sum())+.5*REG*float((b*b).sum())
    analytic=gradient+REG*beta;errors=[]
    for i in rng.choice(beta.size,12,replace=False):
        plus=beta.copy();minus=beta.copy();plus.flat[i]+=1e-5;minus.flat[i]-=1e-5
        finite=(independent(plus)-independent(minus))/2e-5
        errors.append(abs(finite-analytic.flat[i]));assert errors[-1]<2e-6*max(1.,abs(finite))
    # Independent group-mean check, including an all-UNKNOWN carrier.
    records=[dict(indices=np.array([0,1]),known_wrong=np.array([[False,True,False,False],[True,True,False,False]]),compat={(0,0):[1]}),
             dict(indices=np.array([2,3]),known_wrong=np.zeros((2,4),bool),compat={})]
    pp,nn,receipt=weights(4,records,[dict(pool_index=0,native_identity=[0,0,1,1])],1)
    s=rng.normal(size=(4,4));lp=.5*np.maximum(0,1-s)**2;ln=.5*np.maximum(0,1+s)**2
    expected=.1*lp[0,0]+.1*(ln[1,0]+(ln[0,1]+ln[1,1])/2)/2
    assert abs(float((pp*lp+nn*ln).sum())-expected)<1e-12 and np.count_nonzero(nn[2:])==0
    return dict(status='PASS',sample_rows=len(x),zero_anchor_max_error=zero_error,export_max_error=export_error,
                streamed_dense_loss_error=abs(total-loss),streamed_dense_gradient_error=float(abs(grad-gradient).max()),
                finite_difference_max_error=max(errors),UNKNOWN_and_alias_mask_control=True,
                carrier_type_denominator_control=True,existing_runtime_projection_audit='run454/sampled_runtime_score_audit.json',
                existing_runtime_projection_audit_sha256=digest(POOL/'sampled_runtime_score_audit.json'))

def run():
    start=time.monotonic();allocation=read(ROOT/'.codex-local/run457_execution_allocation.json')
    deadline=min(start+300,allocation['deadline_monotonic']);pieces=[];count=0;iterations=0
    atomic_json(DEST/'fit_budget.json',dict(start_monotonic=start,deadline_monotonic=deadline,limit_seconds=300))
    assert read(DEST/'mining_completion.json')['all_five_complete']
    result=dict(status='NOT_COMPLETED',deployable=False,full_contract_pass=False)
    try:
        with np.load(OUT/'run453/unified_model.npz') as z:anchor=z['theta'].astype(np.float64)
        assert anchor.shape==(4,605) and np.isfinite(anchor).all()
        identity=read(POOL/'identity_control.json')['original_children'];assert len(identity)==269
        with np.load(OUT/'run456/feature_scale.npz') as z:d=z['d'].copy()
        with np.load(OUT/'run456/unified_model.npz') as z:initial_theta=z['theta'].copy()
        beta_initial=(initial_theta-anchor)*d
        bindings=[];samples=[];exposed_bindings=[]
        from weld_pipeline.sjg_exposed_risk_1316 import attach,controls as exposed_controls
        for key in TRAIN:
            guard(deadline);folder=POOL/f'workset_{key}';receipt=read(OUT/f'run450/workset_{key}/receipt.json');sr=read(folder/'structure_receipt.json')
            files=dict(sr['files'])
            files.update({'features.npy':receipt['features_sha256'],'base_margins.npy':receipt['base_sha256']})
            for f,h in files.items():guard(deadline);assert digest(folder/f)==h
            with (folder/'structure.pkl').open('rb') as f:data=pickle.load(f)
            pairs=np.load(folder/'pool_pairs.npy',mmap_mode='r');n=len(pairs)
            pos=[v for v in identity if v['source']==key]
            for v in pos:assert pairs[v['pool_index']].tolist()==v['native_identity'][:3]
            pairs._mmap.close()
            wp,wn,stats=weights(n,data['records'],pos,data['G']);del data
            p=dict(key=key,folder=folder,n=n,wp=wp,wn=wn,anchor_score=np.empty((n,4)))
            sample_ids=np.unique(np.r_[np.arange(min(n,32)),[v['pool_index'] for v in pos]])
            for a,b,x,base in chunks(p,deadline):
                assert np.isfinite(x).all() and np.isfinite(base).all()
                p['anchor_score'][a:b]=base+x@anchor.T
                selected=sample_ids[(sample_ids>=a)&(sample_ids<b)]-a
                if len(selected):samples.append((x[selected].copy(),p['anchor_score'][a:b][selected].copy(),wp[a:b][selected].copy(),wn[a:b][selected].copy(),base[selected].copy()))
            bindings.append(dict(source=key,rows=n,files=files,**stats))
            extra,er=attach(p,anchor,d);pieces.extend([p,extra]);exposed_bindings.append(er)
            event('POOL_READY',source=key,rows=n,**stats);gc.collect();check_resources()
        assert sum(v['positive_count'] for v in bindings)==269
        assert sum(v['negative_count'] for v in bindings)==1251058
        save_npz(DEST/'feature_scale.npz',d=d)
        atomic_json(DEST/'pool_binding.json',dict(pieces=bindings,anchor_sha256=digest(OUT/'run453/unified_model.npz'),
                    original_identity_sha256=digest(POOL/'identity_control.json'),exposed_bindings=exposed_bindings,DEV_reads=0,STEP_reads=0,fit_calls=0))
        exposed_controls(anchor,initial_theta,d,samples);del samples
        event('CONTROLS_PASS');last={};initial=None
        def objective(flat):
            nonlocal count,initial
            guard(deadline);beta=flat.reshape(4,605);loss=.5*REG*float((beta*beta).sum());gradient=REG*beta
            pn=nn=exposed_nn=0;max_violation=0.
            for p in pieces:
                for a,b,x,base in chunks(p,deadline):
                    l,g,s=block_loss(x,p['anchor_score'][a:b],p['wp'][a:b],p['wn'][a:b],beta,d)
                    loss+=l;gradient+=g
                    positive=p['wp'][a:b]>0;negative=p['original_wn'][a:b]>0
                    exposed_nn+=int(np.count_nonzero(p['exposed_mask'][a:b]&(s>=0)))
                    pn+=int(np.count_nonzero(positive&(s<=0)));nn+=int(np.count_nonzero(negative&(s>=0)))
                    max_violation=max(max_violation,float(np.max((1-s)[positive],initial=0)),float(np.max((1+s)[negative],initial=0)))
                gc.collect();check_resources()
            assert np.isfinite(loss) and np.isfinite(gradient).all()
            count+=1
            if initial is None:initial=loss
            last.update(x=flat.copy(),value=loss,gradient=gradient.ravel().copy(),positive_nonpositive=pn,negative_nonnegative=nn,maximum_margin_violation=max_violation,exposed_negative_nonnegative=exposed_nn)
            event('FULL_OBJECTIVE',evaluation=count,objective=loss,gradient_inf=float(abs(gradient).max()),positive_nonpositive=pn,negative_nonnegative=nn,exposed_negative_nonnegative=exposed_nn)
            return loss,gradient.ravel()
        def callback(flat):
            nonlocal iterations
            if not np.array_equal(flat,last['x']):objective(flat)
            iterations+=1;beta=flat.reshape(4,605)
            save_npz(DEST/'last_accepted_parameters.npz',beta=beta,theta=anchor+beta/d,d=d)
            atomic_json(DEST/'training_progress.json',dict(iteration=iterations,complete_objective_evaluations=count,objective=last['value'],
                gradient_inf=float(abs(last['gradient']).max()),seconds=time.monotonic()-start))
            event('ITERATION_ACCEPTED',iteration=iterations,maximum=60,objective=last['value'])
        solved=minimize(objective,beta_initial.ravel(),jac=True,method='L-BFGS-B',callback=callback,
                        options=dict(maxiter=60,maxfun=120,maxcor=10,maxls=10,ftol=1e-10,gtol=1e-6))
        atomic_binary_write(DEST/'solver_result.pkl',lambda f:pickle.dump(solved,f,protocol=5))
        normal=bool(solved.success or (solved.status==1 and ('ITERATIONS REACHED LIMIT' in str(solved.message).upper() or 'EVALUATIONS EXCEEDS LIMIT' in str(solved.message).upper())))
        result.update(solver_status=int(solved.status),solver_success=bool(solved.success),solver_message=str(solved.message),
                      solver_iterations=int(solved.nit),solver_nfev=int(solved.nfev),complete_objective_evaluations=count)
        if not normal:raise RuntimeError('Optimizer returned numerical/line-search failure rather than normal convergence or registered limit: '+str(solved.message))
        if not np.array_equal(solved.x,last['x']):objective(solved.x)
        beta=solved.x.reshape(4,605);theta=anchor+beta/d
        assert np.isfinite(theta).all()
        save_npz(DEST/'unified_model.npz',theta=theta)
        with np.load(DEST/'unified_model.npz') as z:assert np.array_equal(z['theta'],theta)
        result.update(status='NORMAL_NUMERICAL_CANDIDATE_READY_FOR_ACTUAL_EXTRACTION',deployable=True,
            initial_objective=initial,final_objective=last['value'],gradient_inf=float(abs(last['gradient']).max()),
            positive_nonpositive=last['positive_nonpositive'],negative_nonnegative=last['negative_nonnegative'],exposed_negative_nonnegative=last['exposed_negative_nonnegative'],
            maximum_margin_violation=last['maximum_margin_violation'],parameter_L1_change=float(abs(theta-anchor).sum()),
            parameter_L2_change=float(np.linalg.norm(theta-anchor)),beta_L2=float(np.linalg.norm(beta)),
            model_sha256=digest(DEST/'unified_model.npz'),feature_dim=605,original_G=269,complete_parents=162,
            DEV_reads=0,STEP_reads=0,zero_violation_not_required=True,reload_exact=True)
        atomic_json(DEST/'training_receipt.json',dict(**result,seconds=time.monotonic()-start))
    except Exception:
        result.update(status='FIT_INTERRUPTED_OR_NUMERICALLY_FAILED_NO_DEPLOYMENT',error=traceback.format_exc())
        print(result['error'],flush=True)
    result.update(seconds=time.monotonic()-start,accepted_iterations=iterations,complete_objective_evaluations=count)
    atomic_json(DEST/'fit_result.json',result);event('FIT_FINISH',status=result['status'],deployable=result['deployable'])
