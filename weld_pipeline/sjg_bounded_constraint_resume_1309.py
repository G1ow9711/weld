"""Fixed-witness linear feasibility and nearest residual, with finite-pool separation.

Only five TRAIN sources enter this module. No truth table is deployed. Solver
optimality and pool feasibility are separate from full-domain extraction gates.
"""
import collections,gc,time,traceback
import numpy as np
from scipy.optimize import linprog
from scipy import sparse
from weld_pipeline.sjg_local_runtime_904 import OUT,read,digest,check_resources
from weld_pipeline.sjg_owned_continuation_watchdog_1295 import atomic_json
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_constraint_pool_1306 import DEST,TRAIN,compile_piece,load_piece

TOL=1e-6

def remaining(deadline):
    seconds=deadline-time.monotonic()
    if seconds<=0:raise TimeoutError('Original 600-second compile/solve deadline reached')
    return seconds

def solver(c,A,b,bounds,deadline):
    check_resources()
    result=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs',options={
        'time_limit':remaining(deadline),'primal_feasibility_tolerance':1e-8,
        'dual_feasibility_tolerance':1e-8,'threads':1})
    report=dict(status=int(result.status),message=result.message,iterations=int(result.nit),
                objective=None if result.fun is None else float(result.fun),optimal=False)
    if result.status==0:
        v=result.x;mu=result.ineqlin.marginals;lm=result.lower.marginals;um=result.upper.marginals
        residual=np.asarray(A@v-b);stationarity=c-np.asarray(A.T@mu)-lm-um
        lower=np.asarray([0. if lo is None else lo for lo,hi in bounds]);upper=np.asarray([0. if hi is None else hi for lo,hi in bounds])
        dual=float(b@mu+lower@lm+upper@um)
        report.update(primal_max_violation=float(max(0.,residual.max(initial=0.))),
            dual_stationarity_max=float(abs(stationarity).max(initial=0.)),
            dual_sign_violation=float(max(0.,mu.max(initial=0.),-lm.min(initial=0.),um.max(initial=0.))),
            dual_objective=dual,primal_dual_gap=float(c@v-dual),finite=bool(np.isfinite(v).all()))
        report['optimal']=bool(report['finite'] and report['primal_max_violation']<=TOL and
            report['dual_stationarity_max']<=TOL and report['dual_sign_violation']<=TOL and
            abs(report['primal_dual_gap'])<=TOL*(1+abs(float(c@v))))
    return result,report

def stage_one(x,base,y,deadline):
    dim=x.shape[1];a=sparse.hstack([sparse.csr_matrix(-y[:,None]*x),-np.ones((len(y),1))],format='csc')
    b=y*base-1.;c=np.r_[np.zeros(dim),1.]
    result,report=solver(c,a,b,[(None,None)]*dim+[(0,None)],deadline)
    if not report['optimal']:return None,None,report
    theta=result.x[:dim];xi=float(max(0.,result.x[-1],np.max(1-y*(base+x@theta),initial=0.)))
    report.update(recomputed_feasible_slack=xi,slack_solver=float(result.x[-1]),
        numerical_lower_bound=float(report['dual_objective']),upper_bound=xi)
    return theta,xi,report

def stage_two(x,base,y,anchor,gamma,deadline):
    dim=x.shape[1];a=sparse.csr_matrix(-y[:,None]*x);eye=sparse.eye(dim,format='csr')
    matrix=sparse.vstack([sparse.hstack([a,sparse.csr_matrix(a.shape)]),
        sparse.hstack([eye,-eye]),sparse.hstack([-eye,-eye])],format='csc')
    rhs=np.r_[y*(base+x@anchor)-gamma,np.zeros(2*dim)]
    result,report=solver(np.r_[np.zeros(dim),np.ones(dim)],matrix,rhs,
        [(None,None)]*dim+[(0,None)]*dim,deadline)
    if not report['optimal']:return None,report
    theta=anchor+result.x[:dim]
    violation=float(max(0.,np.max(gamma-y*(base+x@theta),initial=0.)))
    report.update(original_formula_violation=violation,L1_change=float(abs(theta-anchor).sum()),
        L2_change=float(np.linalg.norm(theta-anchor)),maximum_parameter_change=float(abs(theta-anchor).max()))
    report['optimal'] &= violation<=TOL
    return theta if report['optimal'] else None,report

def controls(deadline):
    examples=[]
    # A decreasing classifier requires an actually negative unbounded weight.
    for name,x,y,expected in [
        ('separable_negative_weight',np.array([[-1.,1.],[1.,1.]]),np.array([1.,-1.]),0.),
        ('incompatible_affine_middle',np.array([[-1.,1.],[0.,1.],[1.,1.]]),np.array([1.,-1.,1.]),1.)]:
        t,xi,r=stage_one(x,np.zeros(len(y)),y,deadline)
        assert r['optimal'] and abs(xi-expected)<TOL,(name,r)
        if expected==0:
            assert t[0]<0
            z,s=stage_two(x,np.zeros(len(y)),y,np.zeros(2),1-xi,deadline)
            assert s['optimal'] and np.min(y*(x@z))>=1-TOL
            assert np.max(abs(x@z-np.einsum('ij,j->i',x,z)))<1e-12
        examples.append(dict(name=name,expected_slack=expected,stage_one=r))
    atomic_json(DEST/'solver_controls.json',dict(status='PASS',examples=examples,
        only_synthetic_no_scientific_fit=True,explicit_unrestricted_weights=True))

def positives_and_identity():
    positives=set();rows=[];raw=[];counts=[];parents_count=0
    for key in TRAIN:
        d=load_piece(key);folder=DEST/f'workset_{key}'
        parents=read(folder/'parents.json');w=read(folder/'complete_parent_witnesses.json')
        rm={r['row']:r for r in d['records']};positive_entries=[]
        for pi,p in enumerate(parents['parents']):
            view=p[0];wit=w['views'][2*pi]
            assert wit['parent']==pi and not wit['whole_parent_reverse'] and wit['max_mm']<=3
            assert len(view)==wit['children']
            for child,(i,c) in enumerate(view):
                pair=d['pairs'][i];identity=(*map(int,pair),int(c)+1)
                assert list(identity)==wit['input_native_pairs'][child]
                r=rm[int(pair[0])];local=int(np.searchsorted(r['indices'],i))
                assert not r['known_wrong'][local,c] and r['compat'].get((local,c))
                entry=(key,int(i),int(c),1);positives.add(entry);positive_entries.append(entry)
                rows.append(dict(source=key,parent=pi,child=child,pool_index=int(i),native_identity=list(identity),
                    full_parent_reverse=False,witness_max_mm=wit['max_mm']))
        assert len(positive_entries)==parents['G']==w['G']
        parents_count+=len(parents['parents'])
        for r in d['records']:
            for (i,c),owners in r['compat'].items():
                if owners:assert not r['known_wrong'][i,c]
            if r.get('mask_reason')=='UNKNOWN_background':assert not r['known_wrong'].any()
        if key in ('21','BM79A_native_local'):
            selected=read(OUT/f'run453/trained/proposal_{key}/selected_intervals.json')['intervals']
            with np.load(OUT/f'run430/source_{key}/role_map.npz') as z:m={k:z[k] for k in z.files}
            lookup={tuple(map(int,p)):i for i,p in enumerate(d['pairs'])};rmap={int(v):i for i,v in enumerate(m['rows'])}
            assert len(selected)==(50 if key=='21' else 44)
            for qi,v in enumerate(selected):
                row=v['row'];j=rmap[row];a,b=m['offsets'][j:j+2];q=m['q_mm'][a:b]
                i=int(np.searchsorted(q,v['lo_mm']));j=int(np.searchsorted(q,v['hi_mm']))
                assert q[i]==v['lo_mm'] and q[j]==v['hi_mm']
                ix=lookup[row,i,j];r=rm[row];loc=int(np.searchsorted(r['indices'],ix));c=v['category']-1
                state='known_wrong' if r['known_wrong'][loc,c] else ('native_compatible' if r['compat'].get((loc,c)) else 'UNKNOWN')
                raw.append(dict(source=key,query=qi,pool_index=ix,row=row,i=i,j=j,category=c+1,label=state))
        counts.append(dict(source=key,G=parents['G'],known_wrong_constraints=int(sum(r['known_wrong'].sum() for r in d['records'])),
            pool_rows=len(d['pairs']),structure_receipt_sha256=digest(folder/'structure_receipt.json')))
        del d;gc.collect()
    assert len(rows)==269 and parents_count==162
    atomic_json(DEST/'identity_control.json',dict(status='PASS',original_G=269,complete_parents=162,
        unique_positive_constraints=len(positives),original_children=rows,all_new_raw=raw,counts=counts,
        native_alias_and_UNKNOWN_not_negative=True,stable_first_complete_parent_view=True,DEV_reads=0,STEP_reads=0))
    return positives

def scan(theta,positive,gamma,deadline):
    worst=set();reports=[];minimum=np.inf;maximum_error=0.
    for key in TRAIN:
        remaining(deadline);d=load_piece(key);n=len(d['pairs']);scores=np.empty((n,4),np.float64)
        for start in range(0,n,4096):
            stop=min(n,start+4096);x=d['x'][start:stop].astype(np.float64)
            scores[start:stop]=d['base'][start:stop].astype(np.float64)+x@theta.T
            split=(x[:,:180]@theta[:,:180].T+x[:,180:360]@theta[:,180:360].T+
                x[:,360:600]@theta[:,360:600].T+x[:,600:604]@theta[:,600:604].T+x[:,604:605]*theta[:,604])
            maximum_error=max(maximum_error,float(abs((x@theta.T)-split).max(initial=0.)))
        nv=0;nc=0;pv=0;minpos=np.inf;minneg=np.inf;examples=[]
        for r in d['records']:
            ids=r['indices'];z=scores[ids]
            for c in range(4):
                local=np.flatnonzero(r['known_wrong'][:,c]);nc+=len(local)
                if not len(local):continue
                val=-z[local,c];minneg=min(minneg,float(val.min()));viol=gamma-val>TOL;nv+=int(viol.sum())
                if viol.any():
                    loc=local[int(np.argmin(val))];ix=int(ids[loc]);worst.add((key,ix,c,-1))
                    if len(examples)<12:examples.append(dict(native_pair=d['pairs'][ix].tolist(),category=c+1,score=float(scores[ix,c])))
        for k,i,c,y in sorted(positive):
            if k!=key:continue
            s=float(scores[i,c]);minpos=min(minpos,s);pv+=int(gamma-s>TOL)
        minimum=min(minimum,minpos,minneg)
        reports.append(dict(source=key,negative_count=nc,negative_violations=nv,positive_violations=pv,
            minimum_positive_signed_score=None if minpos==np.inf else minpos,
            minimum_negative_signed_score=None if minneg==np.inf else minneg,examples=examples))
        print('FULL_POOL_SCAN',key,'KNOWN_NEGATIVES',nc,'VIOLATIONS',nv,'POSITIVE_VIOLATIONS',pv,flush=True)
        del d,scores;gc.collect();check_resources()
    assert maximum_error<=TOL,'Training score disagrees with decomposed 605-dimensional runtime formula'
    return worst,dict(pieces=reports,minimum_signed_score=float(minimum),target_margin=gamma,
        violations=sum(r['negative_violations']+r['positive_violations'] for r in reports),
        maximum_decomposed_formula_error=maximum_error,full_frozen_pool_scanned=True)

def matrix(active,category,deadline):
    chunks=[];bases=[];labels=[];identities=[]
    for key in TRAIN:
        records=sorted(v for v in active if v[0]==key and v[2]==category)
        if not records:continue
        remaining(deadline);d=load_piece(key);ix=np.array([v[1] for v in records]);
        chunks.append(d['x'][ix].astype(np.float64));bases.append(d['base'][ix,category].astype(np.float64))
        labels.extend(v[3] for v in records);identities.extend(records);del d;gc.collect()
    if not chunks:return np.empty((0,605)),np.empty(0),np.empty(0),identities
    return np.concatenate(chunks),np.concatenate(bases),np.asarray(labels,float),identities

def run():
    start=read(DEST/'constraint_budget.json')['start_monotonic'];deadline=start+600.;rounds=[]
    atomic_json(DEST/'constraint_budget.json',dict(start_monotonic=start,deadline_monotonic=deadline,seconds=600,max_rounds=3))
    result=dict(deployable=False,status='STARTED',fit_kind='bounded_linear_program_not_Adam',Adam_updates=0,
        original_G=269,complete_parents=162,DEV_reads=0,STEP_reads=0,full_contract_pass=False)
    try:
        assert read(DEST/'solver_controls.json')['status']=='PASS'
        for key in TRAIN:
            remaining(deadline)
            folder=DEST/f'workset_{key}'; receipt=read(folder/'structure_receipt.json')
            for filename,sha in receipt['files'].items():assert digest(folder/filename)==sha
            print('VERIFIED_COMPILED_POOL_NO_RECOMPUTATION',key,flush=True)
        positive=positives_and_identity()
        with np.load(OUT/'run453/unified_model.npz') as z:anchor=z['theta'].copy()
        initial,initial_report=scan(anchor,positive,1.,deadline);active=positive|initial
        atomic_json(DEST/'initial_full_pool_scan.json',initial_report)
        for iteration in range(1,4):
            remaining(deadline);print('CONSTRAINT_ROUND',iteration,'/3 ACTIVE',len(active),flush=True)
            atomic_json(DEST/f'round_{iteration}_identities.json',dict(entries=sorted(active)))
            stage1=[];working=[];slacks=[]
            for category in range(4):
                x,b,y,ids=matrix(active,category,deadline)
                if not len(y):
                    t=anchor[category];xi=0.;rep=dict(optimal=True,empty=True,numerical_lower_bound=0.,upper_bound=0.)
                else:t,xi,rep=stage_one(x,b,y,deadline)
                stage1.append(dict(category=category+1,constraints=len(y),**rep));working.append((x,b,y));slacks.append(xi)
                atomic_json(DEST/f'round_{iteration}_stage1.json',dict(types=stage1))
                print('LP_STAGE1',iteration,'TYPE',category+1,rep,flush=True)
                if not rep['optimal']:raise RuntimeError('Stage-one solver did not provide numerically verified optimality')
            xi=max(slacks);gamma=1-xi;lower=max(v['numerical_lower_bound'] for v in stage1)
            # A boundary value near zero margin is explicitly undecided, not a
            # proof that all legal witnesses or point clouds are inseparable.
            if gamma<=10*TOL:
                result.update(status='NO_VERIFIED_STRICT_COMMON_MARGIN_FOR_FIXED_WITNESSES',
                    common_slack_upper=xi,numerical_slack_lower=lower,common_margin=gamma,
                    boundary_or_nonpositive=True);break
            theta=np.empty_like(anchor);stage2=[]
            for category,(x,b,y) in enumerate(working):
                if len(y):t,rep=stage_two(x,b,y,anchor[category],gamma,deadline)
                else:t=anchor[category];rep=dict(optimal=True,empty=True,L1_change=0.)
                stage2.append(dict(category=category+1,**rep));atomic_json(DEST/f'round_{iteration}_stage2.json',dict(types=stage2))
                print('LP_STAGE2',iteration,'TYPE',category+1,rep,flush=True)
                if not rep['optimal']:raise RuntimeError('Stage-two solver did not provide numerically verified optimality')
                theta[category]=t
            atomic_binary_write(DEST/f'round_{iteration}_candidate.npz',lambda f:np.savez(f,theta=theta))
            del working;gc.collect()
            added,audit=scan(theta,positive,gamma,deadline)
            audit.update(round=iteration,common_slack_upper=xi,numerical_slack_lower=lower,
                active_constraints=len(active),new_worst_constraints=len(added-active),
                parameter_L1_change=float(abs(theta-anchor).sum()),parameter_L2_change=float(np.linalg.norm(theta-anchor)))
            atomic_json(DEST/f'round_{iteration}_full_pool_scan.json',audit);rounds.append(audit)
            if audit['violations']==0 and audit['minimum_signed_score']>10*TOL:
                atomic_binary_write(DEST/'unified_model.npz',lambda f:np.savez(f,theta=theta))
                with np.load(DEST/'unified_model.npz') as z:assert np.array_equal(z['theta'],theta)
                result.update(deployable=True,status='FINITE_POOL_COMMON_MARGIN_VERIFIED',common_margin=gamma,
                    common_slack_upper=xi,numerical_slack_lower=lower,not_full_domain_certificate=True)
                atomic_json(DEST/'training_receipt.json',dict(**result,model_sha256=digest(DEST/'unified_model.npz'),
                    reload_exact=True,feature_dim=605,rounds=iteration,seconds=time.monotonic()-start,
                    prior_model_sha256=digest(OUT/'run453/unified_model.npz'),parameter_L1_change=audit['parameter_L1_change'],
                    parameter_L2_change=audit['parameter_L2_change'],all_269_positive_witnesses_preserved=True))
                break
            active|=added
        if result['status']=='STARTED':result['status']='THREE_ROUND_LIMIT_WITH_FULL_POOL_VIOLATIONS'
    except Exception:
        result.update(status='UNRESOLVED_EXECUTION_OR_NUMERICAL_FAILURE',error=traceback.format_exc())
        print(result['error'],flush=True)
    result.update(rounds=rounds,seconds=time.monotonic()-start)
    atomic_json(DEST/'constraint_result.json',result)
    print('CONSTRAINT_RESULT',result['status'],'DEPLOYABLE',result['deployable'],flush=True)
