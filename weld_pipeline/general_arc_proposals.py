"""Refine generic circle seeds against dense surface boundary samples."""
import numpy as np
from weld_pipeline.seed_circle_proposals import propose_seed_circles
from weld_pipeline.circle_candidates import fit_xy_circle
from weld_pipeline.arc_surface_candidates import supported_arc


def robust_supported_arc(xyz,proposal,seed=701):
    center=np.asarray(proposal['center_xy_mm']);radius=proposal['radius_mm']
    xy=np.unique(np.round(xyz[abs(np.linalg.norm(xyz[:,:2]-center,axis=1)-radius)<.3,:2],3),axis=0)
    if len(xy)<8:return None
    rng=np.random.default_rng(seed);best=None
    for _ in range(600):
        fit=fit_xy_circle(xy[rng.choice(len(xy),3,replace=False)])
        if fit is None:continue
        c,r,_=fit
        if abs(r-radius)+np.linalg.norm(c-center)>5.:continue
        errors=abs(np.linalg.norm(xy-c,axis=1)-r);mask=errors<.005
        quality=(int(mask.sum()),-float(errors[mask].mean()) if mask.any() else -np.inf)
        if best is None or quality>best[0]:best=(quality,mask)
    if best is None or best[0][0]<8:return None
    c,r,_=fit_xy_circle(xy[best[1]])
    arc=supported_arc(xyz,{'center_xy_mm':c,'radius_mm':r},radial_tolerance_mm=.005)
    if arc is None:return None
    radial=abs(np.linalg.norm(xyz[:,:2]-arc['center_xy_mm'],axis=1)-arc['radius_mm'])
    p=np.unique(np.round(xyz[radial<.005,:2],3),axis=0)
    theta=np.sort(np.arctan2(p[:,1]-arc['center_xy_mm'][1],p[:,0]-arc['center_xy_mm'][0]))
    gaps=np.diff(np.r_[theta,theta[0]+2*np.pi]);inside=np.delete(gaps,np.argmax(gaps))
    # Spurious circles through unrelated wall patches leave long empty angles.
    if inside.max()>max(np.deg2rad(3),3*np.median(inside)):return None
    return {**arc,'proposal_source':'robust_generic_three_point_circle'}


def propose_general_arcs(xyz):
    xyz=np.asarray(xyz,float);candidates,stats=propose_seed_circles(xyz);result=[]
    for proposal in candidates:
        arc=robust_supported_arc(xyz,proposal)
        if arc is None:continue
        if any(np.linalg.norm(np.asarray(arc['center_xy_mm'])-a['center_xy_mm'])+abs(arc['radius_mm']-a['radius_mm'])<.1 for a in result):continue
        result.append(arc)
    return result,{**stats,'supported_general_arcs':len(result)}
