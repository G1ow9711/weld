"""Numerical LP checks. Approximate free-variable duals are not rigorous bounds."""
import time
import numpy as np
from scipy.optimize import linprog
from scipy import sparse

TOL=1e-6

def certificate(c,A,b,bounds,result):
    if result.status!=0:return dict(valid=False,status=int(result.status),message=str(result.message))
    v=np.asarray(result.x);mu=np.asarray(result.ineqlin.marginals);lm=np.asarray(result.lower.marginals);um=np.asarray(result.upper.marginals)
    lower=np.array([0. if lo is None else lo for lo,hi in bounds]);upper=np.array([0. if hi is None else hi for lo,hi in bounds])
    has_lower=np.array([lo is not None for lo,hi in bounds]);has_upper=np.array([hi is not None for lo,hi in bounds])
    raw=(c,b,A.data if sparse.issparse(A) else A,v,mu,lm,um,lower[has_lower],upper[has_upper],np.array([result.fun]))
    if not all(np.isfinite(a).all() for a in raw):return dict(valid=False,status=0,reason='NONFINITE_CERTIFICATE_OR_INPUT')
    stationarity=c-np.asarray(A.T@mu)-lm-um;residual=np.asarray(A@v-b)
    arrays=(c,b,A.data if sparse.issparse(A) else A,v,mu,lm,um,stationarity,residual,np.array([result.fun]))
    finite=all(np.isfinite(a).all() for a in arrays)
    if not finite:return dict(valid=False,status=0,reason='NONFINITE_CERTIFICATE_OR_INPUT')
    primal=float(max(0.,residual.max(initial=0.)))
    bound=float(max(0.,(lower[has_lower]-v[has_lower]).max(initial=0.),(v[has_upper]-upper[has_upper]).max(initial=0.)))
    missing=float(max(abs(lm[~has_lower]).max(initial=0.),abs(um[~has_upper]).max(initial=0.)))
    dual=float(b@mu+lower[has_lower]@lm[has_lower]+upper[has_upper]@um[has_upper])
    sign=float(max(0.,mu.max(initial=0.),-lm[has_lower].min(initial=0.),um[has_upper].max(initial=0.)))
    gap=float(c@v-dual);objective_error=float(abs(c@v-result.fun))
    station=float(abs(stationarity).max(initial=0.))
    valid=primal<=TOL and bound<=TOL and missing<=TOL and sign<=TOL and station<=TOL and objective_error<=TOL*(1+abs(result.fun)) and abs(gap)<=TOL*(1+abs(result.fun))
    return dict(valid=bool(valid),status=0,all_input_parameter_objective_multiplier_residual_finite=True,
        primal_max_violation=primal,actual_bound_violation=bound,nonexistent_bound_multiplier_abs_max=missing,
        dual_stationarity_max=station,dual_sign_violation=sign,dual_objective_diagnostic=dual,
        primal_dual_gap_diagnostic=gap,objective_error=objective_error,
        approximate_dual_not_rigorous_lower_bound_for_free_parameters=True)

def solve(c,A,b,bounds,deadline):
    remaining=deadline-time.monotonic()-20
    if remaining<=0:raise TimeoutError('Reserve final 20 seconds for checkpoint and exit')
    assert np.isfinite(c).all() and np.isfinite(b).all() and np.isfinite(A.data).all()
    result=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method='highs',options={
        'time_limit':remaining,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8,'threads':1})
    checked=certificate(c,A,b,bounds,result)
    return result,dict(**checked,solver_status=int(result.status),solver_message=str(result.message),iterations=int(result.nit),
        objective=None if result.fun is None else float(result.fun),assigned_solver_seconds=remaining)

def first_problem(x,base,y):
    dim=x.shape[1];A=sparse.hstack([sparse.csr_matrix(-y[:,None]*x),-np.ones((len(y),1))],format='csc')
    return np.r_[np.zeros(dim),1.],A,y*base-1.,[(None,None)]*dim+[(0,None)]

def second_problem(x,base,y,anchor,gamma):
    dim=x.shape[1];a=sparse.csr_matrix(-y[:,None]*x);eye=sparse.eye(dim,format='csr')
    A=sparse.vstack([sparse.hstack([a,sparse.csr_matrix(a.shape)]),sparse.hstack([eye,-eye]),sparse.hstack([-eye,-eye])],format='csc')
    return np.r_[np.zeros(dim),np.ones(dim)],A,np.r_[y*(base+x@anchor)-gamma,np.zeros(2*dim)],[(None,None)]*dim+[(0,None)]*dim

def controls(deadline):
    import copy
    x=np.array([[-1.,1.],[1.,1.]]);y=np.array([1.,-1.]);b=np.zeros(2)
    args=second_problem(x,b,y,np.zeros(2),1.)
    result,report=solve(*args,deadline)
    assert report['valid'],report
    rejected=[]
    for name in ('nonexistent_bound_multiplier','negative_absolute_auxiliary','nonfinite_multiplier','nonfinite_objective','nonfinite_input'):
        fake=copy.deepcopy(result);c,A,rhs,bounds=args
        if name=='nonexistent_bound_multiplier':fake.lower.marginals[0]=1.;fake.upper.marginals[0]=-1.
        elif name=='negative_absolute_auxiliary':fake.x[2]=-1.
        elif name=='nonfinite_multiplier':fake.ineqlin.marginals[0]=np.nan
        elif name=='nonfinite_objective':fake.fun=np.inf
        else:A=A.copy();A.data[0]=np.nan
        check=certificate(c,A,rhs,bounds,fake);assert not check['valid'],name
        if name=='nonexistent_bound_multiplier':assert check['nonexistent_bound_multiplier_abs_max']==1. and check['dual_stationarity_max']<=TOL
        if name=='negative_absolute_auxiliary':assert check['actual_bound_violation']>=1.
        rejected.append(name)
    # Independent check of the actual xi lower bound.
    args=first_problem(x,b,y);fake,rep=solve(*args,deadline);assert rep['valid']
    fake.x[-1]=-1.;assert not certificate(*args,fake)['valid']
    rejected.append('negative_slack_xi')
    return dict(status='PASS',legal_return_accepted=True,forged_certificates_rejected=rejected,
        no_old_capacity_or_geometry_controls_reexecuted=True,real_training_fit_calls=0)
