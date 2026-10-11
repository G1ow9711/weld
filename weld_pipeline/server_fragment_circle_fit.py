"""Joint circle parameter estimation on separate observed 2D fragments.

Does not infer common physical identity, support continuity, or weld endpoints.
Callers retain each fragment and its provenance separately.
"""
import numpy as np
from scipy.optimize import least_squares


def fragment_quadrature(points):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('Finite Nx2 points required')
    # Exact consecutive duplicates contain no additional spatial information.
    points=points[np.r_[True,np.any(np.diff(points,axis=0)!=0,axis=1)]]
    if len(points)<2:return None
    lengths=np.linalg.norm(np.diff(points,axis=0),axis=1)
    weights=np.r_[lengths,0.]+np.r_[0.,lengths]
    if weights.sum()<=0:return None
    return points,weights/weights.sum()


def fit_fragment_circle(fragments):
    """One deterministic initialization; each fragment has equal total weight.

    Returns geometry and residuals only, never a joined finite arc. A small
    residual is not sufficient evidence of a common physical contour.
    """
    records=[fragment_quadrature(p) for p in fragments]
    if not records or any(r is None for r in records):return None
    points=np.concatenate([r[0] for r in records]);weights=np.concatenate([r[1]/len(records) for r in records])
    if len(points)<5:return None
    mean=np.sum(weights[:,None]*points,axis=0)
    scale=float(np.sqrt(np.sum(weights*np.square(points-mean).sum(1))))
    if not np.isfinite(scale) or scale<=0:return None
    local=(points-mean)/scale;sqrtw=np.sqrt(weights)
    matrix=np.column_stack((2*local,np.ones(len(local))))
    seed,_,rank,_=np.linalg.lstsq(matrix*sqrtw[:,None],np.square(local).sum(1)*sqrtw,rcond=1e-10)
    if rank<3:return None
    def residual(center):
        radial=np.linalg.norm(local-center,axis=1)
        return sqrtw*(radial-np.sum(weights*radial))
    def jacobian(center):
        delta=center-local
        grad=delta/np.maximum(np.linalg.norm(delta,axis=1)[:,None],np.finfo(float).tiny)
        return sqrtw[:,None]*(grad-np.sum(weights[:,None]*grad,axis=0))
    fit=least_squares(residual,seed[:2],jac=jacobian,max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    if not fit.success or not np.isfinite(fit.x).all():return None
    center=mean+scale*fit.x
    radius=float(np.sum(weights*np.linalg.norm(points-center,axis=1)))
    if not np.isfinite(radius) or radius<=0:return None
    stats=[]
    for p,w in records:
        errors=np.linalg.norm(p-center,axis=1)-radius
        stats.append(dict(mean_radial_error_mm=float(w@errors),rms_radial_error_mm=float(np.sqrt(w@np.square(errors))),max_radial_error_mm=float(abs(errors).max())))
    return dict(center=center.tolist(),radius=radius,fragments=stats,optimizer_evaluations=int(fit.nfev),support_connected=False)
