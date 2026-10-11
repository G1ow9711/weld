"""Local geometric least-squares refinement on the unchanged observed chain."""
import numpy as np
from scipy.optimize import least_squares
from .contract_curve_geometry import CircularArc


def refine_chain_circle(uv,chain,origin,basis,seed):
    chain=np.asarray(chain,np.int64);closed=chain[0]==chain[-1]
    points=uv[chain[:-1] if closed else chain]
    if len(points)<5:return None
    old=CircularArc.from_parameters(seed['arc_parameters']);mean=points.mean(0)
    scale=float(np.sqrt(np.square(points-mean).sum(1).mean()))
    if not np.isfinite(scale) or scale<=0:return None
    local=(points-mean)/scale;initial=((old.center_mm-origin)@basis-mean)/scale
    def residual(center):
        radial=np.linalg.norm(local-center,axis=1)
        return radial-radial.mean()
    def jacobian(center):
        delta=center-local;radial=np.linalg.norm(delta,axis=1)
        gradient=delta/np.maximum(radial[:,None],np.finfo(float).tiny)
        return gradient-gradient.mean(0)
    fit=least_squares(residual,initial,jac=jacobian,method='trf',max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    if not fit.success or not np.isfinite(fit.x).all():return None
    center=mean+scale*fit.x;radial=np.linalg.norm(points-center,axis=1);radius=float(radial.mean())
    oldradial=np.linalg.norm(points-((old.center_mm-origin)@basis),axis=1)
    old_sse=float(np.square(oldradial-old.radius_mm).sum());sse=float(np.square(radial-radius).sum())
    if not np.isfinite(radius) or radius<=0 or sse>old_sse+1e-10*max(1,old_sse):return None
    ordered=uv[chain]-center;angles=np.unwrap(np.arctan2(ordered[:,1],ordered[:,0]));sweep=float(angles[-1]-angles[0]);increments=np.diff(angles)
    if abs(sweep)<1e-10 or np.any(increments*np.sign(sweep)<=0):return None
    if closed:
        if not np.isclose(abs(sweep),2*np.pi,atol=1e-10,rtol=0):return None
        sweep=float(np.sign(sweep)*2*np.pi)
    elif abs(sweep)>=2*np.pi:return None
    direction=ordered[0]/np.linalg.norm(ordered[0]);arc=CircularArc(origin+basis@center,radius,np.cross(basis[:,0],basis[:,1]),basis@direction,sweep,closed)
    return dict(arc_parameters=arc.to_parameters(),support_vertices=len(points),radial_error_max_mm=float(abs(radial-radius).max()),radial_error_rms_mm=float(np.sqrt(sse/len(points))),observed_chain_length_mm=float(np.linalg.norm(np.diff(uv[chain],axis=0),axis=1).sum()),chain_vertex_indices=chain.tolist(),original_radial_sse_mm2=old_sse,refined_radial_sse_mm2=sse,optimizer_evaluations=int(fit.nfev))
