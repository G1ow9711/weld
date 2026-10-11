"""Pinned project-local HiGHS backend; full CSC model and native options verified."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import csc_matrix

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'.codex-local/runtime/highspy-1.10.0'
if not RUNTIME.is_dir():
    raise RuntimeError('Pinned isolated highspy1.10.0 runtime missing')
sys.path.insert(0,str(RUNTIME))
import highspy
if not Path(highspy.__file__).resolve().is_relative_to(RUNTIME.resolve()):
    raise RuntimeError('Refuse unbound highspy installation')
if highspy.Highs().version()!='1.10.0':
    raise RuntimeError('Pinned native HiGHS version mismatch')

VERSION='DIRECT_HIGHS_FULL_CSC_OPTION_READBACK_215_v2'
PROGRESS_OBSERVER=None
MODEL_OBSERVER=None


def _add_array(digest,name,value,dtype):
    array=np.ascontiguousarray(value,dtype=dtype)
    digest.update(name.encode('ascii'))
    digest.update(json.dumps(list(array.shape),separators=(',',':')).encode('ascii'))
    digest.update(array.tobytes())


def model_digest(cost,integer,lower,upper,row_lower,row_upper,matrix):
    matrix=matrix.tocsc(copy=True)
    matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    digest=hashlib.sha256(b'full-HighsLp-minimize-offset0-canonical-CSC-v1')
    for name,value,dtype in [('cost',cost,'<f8'),('integrality',integer,'u1'),
        ('lower',lower,'<f8'),('upper',upper,'<f8'),('row_lower',row_lower,'<f8'),
        ('row_upper',row_upper,'<f8'),('values',matrix.data,'<f8'),
        ('indices',matrix.indices,'<i8'),('indptr',matrix.indptr,'<i8')]:
        _add_array(digest,name,value,dtype)
    return digest.hexdigest()


def _readback_options(solver,options):
    values={}
    for name,value in options.items():
        if solver.setOptionValue(name,value)!=highspy.HighsStatus.kOk:
            raise ValueError('Native option rejected: '+name+'='+repr(value))
        status,actual=solver.getOptionValue(name)
        if status!=highspy.HighsStatus.kOk or actual!=value:
            raise ValueError('Native option readback differs: '+name)
        values[name]=actual
    return values


def _options(options):
    supplied=dict(options or {})
    accepted={'mip_rel_gap','mip_abs_gap','mip_feasibility_tolerance','presolve','threads','parallel','time_limit','disp'}
    if set(supplied)-accepted:
        raise ValueError('Unsupported adapter options: '+repr(sorted(set(supplied)-accepted)))
    if supplied.get('mip_abs_gap')!=0.0 or supplied.get('mip_rel_gap')!=0.0:
        raise ValueError('Native backend requires explicit abs/rel gaps zero')
    tolerance=float(supplied.get('mip_feasibility_tolerance',1e-9))
    if tolerance<1e-10 or tolerance>1e-9:
        raise ValueError('Native feasibility tolerance must be1e-10..1e-9')
    threads=supplied.get('threads',1)
    if isinstance(threads,bool) or not isinstance(threads,int) or threads<1:
        raise ValueError('Native positive integer thread count required')
    if supplied.get('parallel',False) is not False:
        raise ValueError('Parallel MIP mode must remain off')
    mapped=dict(output_flag=bool(supplied.get('disp',False)),mip_abs_gap=0.0,mip_rel_gap=0.0,
        mip_feasibility_tolerance=tolerance,primal_feasibility_tolerance=tolerance,
        dual_feasibility_tolerance=tolerance,threads=threads,parallel='off',
        presolve='on' if supplied.get('presolve',True) else 'off',random_seed=0)
    if 'time_limit' in supplied:
        mapped['time_limit']=float(supplied['time_limit'])
    return mapped


def _readback_model(solver,expected):
    lp=solver.getLp()
    if lp.sense_!=highspy.ObjSense.kMinimize or lp.offset_!=0.0:
        raise ValueError('Native objective sense/constant changed')
    if lp.a_matrix_.format_!=highspy.MatrixFormat.kColwise:
        raise ValueError('Native matrix orientation changed')
    matrix=csc_matrix((np.asarray(lp.a_matrix_.value_,np.float64),
        np.asarray(lp.a_matrix_.index_,np.int64),np.asarray(lp.a_matrix_.start_,np.int64)),
        shape=(lp.num_row_,lp.num_col_))
    actual=model_digest(lp.col_cost_,[int(v) for v in lp.integrality_],lp.col_lower_,lp.col_upper_,
                        lp.row_lower_,lp.row_upper_,matrix)
    if actual!=expected:
        raise ValueError('Native full matrix/objective/bounds/integrality readback differs')
    return actual


def _finite(value):
    value=float(value)
    return value if np.isfinite(value) else None


def milp(c,*,integrality,bounds,constraints,options=None):
    """SciPy-shaped return; native bounds/witness retained, no fallback solve."""
    cost=np.ascontiguousarray(c,np.float64)
    raw_integer=np.asarray(integrality)
    if cost.ndim!=1 or raw_integer.shape!=cost.shape or np.any((raw_integer!=0)&(raw_integer!=1)):
        raise ValueError('Complete binary/continuous integrality vector required')
    integer=np.asarray(raw_integer,np.uint8)
    matrix=constraints.A.tocsc()
    lower=np.asarray(bounds.lb,np.float64);upper=np.asarray(bounds.ub,np.float64)
    row_lower=np.asarray(constraints.lb,np.float64);row_upper=np.asarray(constraints.ub,np.float64)
    expected=model_digest(cost,integer,lower,upper,row_lower,row_upper,matrix)
    solver=highspy.Highs()
    native_options=_readback_options(solver,_options(options))
    lp=highspy.HighsLp()
    lp.num_col_=len(cost);lp.num_row_=matrix.shape[0]
    lp.sense_=highspy.ObjSense.kMinimize;lp.offset_=0.0
    lp.col_cost_=cost;lp.col_lower_=lower;lp.col_upper_=upper
    lp.row_lower_=row_lower;lp.row_upper_=row_upper
    lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_=matrix.shape[1];lp.a_matrix_.num_row_=matrix.shape[0]
    lp.a_matrix_.start_=np.asarray(matrix.indptr,np.int64)
    lp.a_matrix_.index_=np.asarray(matrix.indices,np.int64)
    lp.a_matrix_.value_=np.asarray(matrix.data,np.float64)
    lp.integrality_=[highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integer]
    if solver.passModel(lp)!=highspy.HighsStatus.kOk:
        raise ValueError('Native complete model rejected')
    actual=_readback_model(solver,expected)
    del lp
    if MODEL_OBSERVER is not None:
        MODEL_OBSERVER(dict(status='NATIVE_OPTIONS_AND_FULL_MODEL_READBACK_PASS',
            native_version=solver.version(),native_options_verified=native_options,
            original_full_model_hash=expected,native_full_model_hash=actual))
    observer=PROGRESS_OBSERVER
    last=[-float('inf')]
    callback_enabled=False
    if observer is not None:
        def callback(kind,message,data_out,data_in,user):
            now=time.monotonic()
            if now-last[0]<5:
                return
            last[0]=now
            observer(dict(scope='NATIVE_MIN_OBJECTIVE_COMPONENT_WITHOUT_BAD_CONSTANT',
                native_version=solver.version(),native_primal_min=_finite(data_out.mip_primal_bound),
                native_dual_min=_finite(data_out.mip_dual_bound),native_relative_gap=_finite(data_out.mip_gap),
                native_node_count=int(data_out.mip_node_count),native_elapsed_seconds=float(data_out.running_time)))
        status=solver.setCallback(callback,None)
        if status==highspy.HighsStatus.kOk:
            callback_enabled=solver.startCallback(highspy.cb.HighsCallbackType.kCallbackMipLogging)==highspy.HighsStatus.kOk
        if not callback_enabled:
            raise ValueError('Configured native progress callback unavailable')
    run_status=solver.run()
    native_status=solver.getModelStatus()
    info=solver.getInfo();solution=solver.getSolution()
    if native_status==highspy.HighsModelStatus.kOptimal:
        status=0
    elif native_status==highspy.HighsModelStatus.kInfeasible:
        status=2
    elif native_status==highspy.HighsModelStatus.kUnbounded:
        status=3
    elif any(native_status==getattr(highspy.HighsModelStatus,name,None) for name in
             ('kTimeLimit','kIterationLimit','kSolutionLimit','kInterrupt','kMemoryLimit')):
        status=1
    else:
        status=4
    valid=bool(solution.value_valid) and int(info.primal_solution_status)==int(highspy.SolutionStatus.kSolutionStatusFeasible)
    x=np.asarray(solution.col_value,np.float64) if valid else None
    if x is not None and x.shape!=cost.shape:
        raise ValueError('Native witness length differs')
    result=OptimizeResult(status=status,success=status==0,x=x,
        message='HiGHS '+solver.version()+': '+solver.modelStatusToString(native_status),
        fun=_finite(info.objective_function_value) if valid else None,
        mip_dual_bound=_finite(info.mip_dual_bound),mip_gap=_finite(info.mip_gap),
        mip_node_count=int(info.mip_node_count),native_run_status=str(run_status),
        native_model_status=str(native_status),native_backend_version=solver.version(),
        native_options_verified=native_options,native_model_verified=True,
        original_full_model_hash=expected,native_full_model_hash=actual,
        native_progress_callback_enabled=callback_enabled)
    solver.clear()
    return result
