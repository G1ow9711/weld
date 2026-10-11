"""Finish only run454 round 3, preserving its constraint identities and anchor."""
import gc,pickle,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,digest,check_resources
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_owned_continuation_watchdog_1295 import atomic_json
from weld_pipeline.sjg_constraint_pool_1306 import TRAIN,load_piece
from weld_pipeline.sjg_bounded_constraint_resume_1309 import matrix
from weld_pipeline.sjg_lp_certificate_1310 import first_problem,second_problem,solve,controls,TOL

DEST=OUT/'run455';PRIOR=OUT/'run454'

def event(name,**data):
    row=dict(event=name,monotonic=time.monotonic(),**data)
    with (DEST/'phase_events.jsonl').open('a',encoding='utf-8') as f:
        import json
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    print('ROUND3',name,data,flush=True)

def full_scan(theta,gamma,label,deadline):
    identity=read(PRIOR/'identity_control.json')['original_children'];pieces=[];overall=0;minimum=np.inf
    event('FULL_POOL_SCAN_BEGIN',label=label)
    for key in TRAIN:
        if time.monotonic()>=deadline-5:raise TimeoutError('Scan deadline reserve reached')
        d=load_piece(key);scores=np.empty((len(d['pairs']),4),np.float64)
        for a in range(0,len(scores),4096):scores[a:a+4096]=d['base'][a:a+4096].astype(np.float64)+d['x'][a:a+4096].astype(np.float64)@theta.T
        types=[dict(negative_count=0,negative_margin_violations=0,negative_nonnegative_scores=0,positive_count=0,
            positive_margin_violations=0,positive_nonpositive_scores=0,minimum_signed_score=None) for _ in range(4)]
        mins=np.full(4,np.inf)
        for r in d['records']:
            z=scores[r['indices']]
            for c in range(4):
                s=z[:,c][r['known_wrong'][:,c]];t=types[c]
                t['negative_count']+=len(s);t['negative_margin_violations']+=int((s+gamma>TOL).sum())
                t['negative_nonnegative_scores']+=int((s>=0).sum());mins[c]=min(mins[c],float((-s).min(initial=np.inf)))
        for v in identity:
            if v['source']!=key:continue
            c=v['native_identity'][-1]-1;s=float(scores[v['pool_index'],c]);t=types[c]
            t['positive_count']+=1;t['positive_margin_violations']+=int(gamma-s>TOL);t['positive_nonpositive_scores']+=int(s<=0)
            mins[c]=min(mins[c],s)
        for c in range(4):types[c]['minimum_signed_score']=None if mins[c]==np.inf else float(mins[c])
        row=dict(source=key,types=types,violations=sum(t['negative_margin_violations']+t['positive_margin_violations'] for t in types),
            positive_nonpositive=sum(t['positive_nonpositive_scores'] for t in types),negative_nonnegative=sum(t['negative_nonnegative_scores'] for t in types))
        pieces.append(row);overall+=row['violations'];minimum=min(minimum,float(mins.min()))
        event('SCAN_PIECE',label=label,source=key,violations=row['violations'],wrong_negative_sign=row['negative_nonnegative'])
        del d,scores;gc.collect();check_resources()
    assert sum(t['negative_count'] for p in pieces for t in p['types'])==1251058
    assert sum(t['positive_count'] for p in pieces for t in p['types'])==269
    out=dict(label=label,pieces=pieces,violations=overall,minimum_signed_score=minimum,target_margin=gamma,
        positive_nonpositive=sum(p['positive_nonpositive'] for p in pieces),negative_nonnegative=sum(p['negative_nonnegative'] for p in pieces),
        all_269_positive_and_1251058_negative_constraints_scanned=True,not_full_domain_certificate=True)
    atomic_json(DEST/f'{label}_full_pool_scan.json',out);event('FULL_POOL_SCAN_END',label=label,violations=overall)
    return out

def run():
    begin=time.monotonic();deadline=begin+480
    atomic_json(DEST/'finish_budget.json',dict(start_monotonic=begin,deadline_monotonic=deadline,seconds=480))
    result=dict(status='PARTIAL_LP_SOLVES_COMPLETE_NO_FINAL_MODEL',deployable=False,round=3,
        fourth_round_added=False,previous_completed_LP_solves=20,new_completed_LP_solves=0,Adam_updates=0,
        original_G=269,DEV_reads=0,STEP_reads=0,full_contract_pass=False)
    try:
        atomic_json(DEST/'certificate_control.json',controls(deadline))
        with np.load(OUT/'run453/unified_model.npz') as z:anchor=z['theta'].copy()
        active={tuple(v) for v in read(PRIOR/'round_3_identities.json')['entries']};assert len(active)==7909
        atomic_json(DEST/'problem_binding.json',dict(active_identities_sha256=digest(PRIOR/'round_3_identities.json'),
            anchor_sha256=digest(OUT/'run453/unified_model.npz'),identity_control_sha256=digest(PRIOR/'identity_control.json'),
            first_stage_arrays_missing_in_original=True,recomputed_missing_components_not_solver_state_resume=True,
            constraints=7909,features=605,solver='scipy.linprog(method=highs)',new_constraint_generation=False))
        working=[];first=np.empty_like(anchor);slacks=[]
        for c in range(4):
            x,b,y,ids=matrix(active,c,deadline);working.append((x,b,y));event('LP_BEGIN',stage=1,category=c+1,constraints=len(y))
            solved,rep=solve(*first_problem(x,b,y),deadline);event('LP_RETURN',stage=1,category=c+1,status=solved.status)
            atomic_binary_write(DEST/f'stage1_type{c+1}_solver.pkl',lambda f:pickle.dump(solved,f,protocol=5))
            atomic_json(DEST/f'stage1_type{c+1}_certificate.json',rep)
            if not rep['valid']:raise RuntimeError('Missing stage-one component did not return a verified numerical optimum')
            t=solved.x[:605];xi=float(max(0.,solved.x[-1],np.max(1-y*(b+x@t),initial=0.)))
            atomic_binary_write(DEST/f'stage1_type{c+1}_parameters.npz',lambda f:np.savez(f,theta=t,solver_variables=solved.x))
            first[c]=t;slacks.append(xi);result['new_completed_LP_solves']+=1
            atomic_json(DEST/'component_progress.json',dict(**result,last_saved=f'stage1_type{c+1}',slack_upper_bounds=slacks))
            event('COMPONENT_SAVED',stage=1,category=c+1,feasible_slack=xi,exact_mathematical_slack_lower_bound=0.)
        xi=max(slacks);gamma=1-xi
        atomic_binary_write(DEST/'stage1_all_types.npz',lambda f:np.savez(f,theta=first))
        result.update(common_slack_upper=xi,exact_mathematical_slack_lower_bound=0.,common_margin=gamma)
        stage1scan=full_scan(first,gamma,'stage1',deadline)
        result['stage1_finite_pool_witness']=bool(stage1scan['violations']==0 and stage1scan['minimum_signed_score']>1e-5)
        atomic_json(DEST/'stage1_result.json',result)
        if gamma<=1e-5:raise RuntimeError('Fixed active witness common margin lies in unresolved boundary band')
        final=np.empty_like(anchor)
        for c,(x,b,y) in enumerate(working):
            event('LP_BEGIN',stage=2,category=c+1,constraints=len(y))
            solved,rep=solve(*second_problem(x,b,y,anchor[c],gamma),deadline)
            event('LP_RETURN',stage=2,category=c+1,status=solved.status)
            atomic_binary_write(DEST/f'stage2_type{c+1}_solver.pkl',lambda f:pickle.dump(solved,f,protocol=5))
            atomic_json(DEST/f'stage2_type{c+1}_certificate.json',rep)
            if not rep['valid']:raise RuntimeError('Third-round stage-two component is not a numerically verified optimum')
            t=anchor[c]+solved.x[:605];assert np.isfinite(t).all()
            violation=float(np.max(gamma-y*(b+x@t),initial=0.));assert violation<=TOL
            final[c]=t
            atomic_binary_write(DEST/f'stage2_type{c+1}_parameters.npz',lambda f:np.savez(f,theta=t,solver_variables=solved.x))
            result['new_completed_LP_solves']+=1
            atomic_json(DEST/'component_progress.json',dict(**result,last_saved=f'stage2_type{c+1}'))
            event('COMPONENT_SAVED',stage=2,category=c+1,original_formula_violation=violation)
        atomic_binary_write(DEST/'stage2_all_types.npz',lambda f:np.savez(f,theta=final))
        del working;gc.collect();stage2scan=full_scan(final,gamma,'stage2',deadline)
        result.update(parameter_L1_change=float(abs(final-anchor).sum()),parameter_L2_change=float(np.linalg.norm(final-anchor)),
            status='ROUND3_FINISHED_FULL_POOL_VIOLATIONS_REMAIN')
        if stage2scan['violations']==0 and stage2scan['minimum_signed_score']>1e-5:
            atomic_binary_write(DEST/'unified_model.npz',lambda f:np.savez(f,theta=final))
            with np.load(DEST/'unified_model.npz') as z:assert np.array_equal(final,z['theta'])
            result.update(deployable=True,status='ROUND3_FINAL_MODEL_FINITE_POOL_VERIFIED')
            atomic_json(DEST/'training_receipt.json',dict(**result,model_sha256=digest(DEST/'unified_model.npz'),
                feature_dim=605,complete_parents=162,reload_exact=True,seconds=time.monotonic()-begin))
    except Exception:
        result.update(error=traceback.format_exc(),status='ROUND3_FINISH_PARTIAL_OR_NUMERICALLY_UNRESOLVED')
        print(result['error'],flush=True)
    result.update(seconds=time.monotonic()-begin)
    atomic_json(DEST/'constraint_result.json',result);event('FINISH',status=result['status'],deployable=result['deployable'])
