"""Finite numerical optimality certificate; does not change the path problem."""
import math


def numerical_certificate(status, solver_fun, recomputed_fun, dual_bound, reported_gap, tolerance, success=True):
    if status != 0 or not success:
        raise ValueError('Solver did not report a completed optimum')
    values=(solver_fun,recomputed_fun,dual_bound,reported_gap,tolerance)
    if any(v is None or not math.isfinite(float(v)) for v in values):
        raise ValueError('Missing or nonfinite objective certificate')
    solver_fun,recomputed_fun,dual_bound,reported_gap,tolerance=map(float,values)
    if tolerance <= 0 or reported_gap < 0:
        raise ValueError('Invalid certificate tolerance or relative gap')
    objective_difference=abs(solver_fun-recomputed_fun)
    bound_difference=abs(dual_bound-recomputed_fun)
    if objective_difference > tolerance:
        raise ValueError('Returned objective disagrees with the integer solution')
    if bound_difference > tolerance:
        raise ValueError('Objective bound does not close within original roundoff tolerance')
    solver_bound_difference=abs(solver_fun-dual_bound)
    if solver_bound_difference > tolerance:
        raise ValueError('Returned primal and dual bounds disagree')
    scaled_gap_difference=abs(reported_gap*abs(solver_fun)-max(0.,solver_fun-dual_bound))
    if abs(solver_fun)>tolerance and scaled_gap_difference>tolerance:
        raise ValueError('Reported relative gap contradicts the returned bounds')
    return dict(certificate='NUMERICAL_OPTIMAL_WITHIN_REGISTERED_TOL',
        reported_mip_gap=reported_gap,exact_zero_reported_gap=reported_gap==0.,
        objective= recomputed_fun,solver_objective=solver_fun,dual_bound=dual_bound,
        objective_difference=objective_difference,absolute_bound_difference=bound_difference,
        solver_bound_difference=solver_bound_difference,scaled_gap_difference=scaled_gap_difference,
        bound_roundoff_tolerance=tolerance)


def validate_integer_rows(x,lower,upper,matrix):
    import numpy as np
    if x is None or not np.isfinite(x).all():raise ValueError('Missing or nonfinite solution')
    rounded=np.rint(x)
    if np.max(np.abs(x-rounded),initial=0)>1e-6:raise ValueError('Noninteger solution')
    if not (np.all(rounded>=lower) and np.all(rounded<=1) and np.all(matrix@rounded<=upper)):
        raise ValueError('Rounded solution violates original variable or constraint bounds')
    return rounded


def regression_control(diagnosis):
    """Actual failed numerical tuple plus invalid-certificate rejection controls."""
    info=diagnosis['solver_info']
    base=dict(status=info['status'],solver_fun=diagnosis['solver_fun'],
        recomputed_fun=diagnosis['recomputed_objective'],dual_bound=info['mip_dual_bound'],
        reported_gap=info['mip_gap'],tolerance=diagnosis['roundoff_tolerance'])
    result=numerical_certificate(**base)
    assert result['exact_zero_reported_gap'] is False
    checks=[]
    cases=[dict(status=1),dict(status=2),dict(success=False),dict(dual_bound=None),dict(reported_gap=-1.),dict(reported_gap=0.01),
        dict(dual_bound=base['recomputed_fun']-1e-5,reported_gap=0.),
        dict(dual_bound=base['recomputed_fun']+1e-5,reported_gap=0.),
        dict(solver_fun=base['recomputed_fun']+1e-5),dict(tolerance=0.)]
    for field in ('solver_fun','recomputed_fun','dual_bound','reported_gap','tolerance'):
        for bad in (float('nan'),float('inf'),float('-inf')):cases.append({field:bad})
    for changes in cases:
        try:numerical_certificate(**dict(base,**changes))
        except ValueError:checks.append(True)
        else:raise AssertionError('Invalid certificate accepted: '+str(changes))
    zero=numerical_certificate(0,0.,0.,0.,0.,64*2.220446049250313e-16)
    assert zero['exact_zero_reported_gap']
    import numpy as np
    for x,lower,upper in [(0.5,0.,1.),(1.,0.,0.),(0.,1.,1.),(2.,0.,2.)]:
        try:validate_integer_rows(np.array([x]),np.array([lower]),np.array([upper]),np.eye(1))
        except ValueError:checks.append(True)
        else:raise AssertionError('Invalid integer/constraint solution accepted')
    assert np.array_equal(validate_integer_rows(np.ones(1),np.zeros(1),np.ones(1),np.eye(1)),np.ones(1))
    return dict(status='NUMERICAL_CERTIFICATE_REGRESSION_PASSED',actual_failure_tuple=result,
        invalid_cases_rejected=len(checks),zero_objective_checked=True,GT_reads=0,fit_calls=0)
