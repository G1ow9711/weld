"""Input graph vertices projected onto finite fitted arc candidates."""
import numpy as np


def arc_vertex_events(arc, uv, origin, basis, band_mm=3.):
    normal=np.cross(basis[:,0],basis[:,1])
    if abs(abs(normal@arc.plane_normal)-1)>1e-10 or abs(normal@(arc.center_mm-origin))>1e-6:
        raise ValueError('Arc and graph must use the same observed plane')
    center=(arc.center_mm-origin)@basis;relative=uv-center
    radial=np.linalg.norm(relative,axis=1)
    keep=np.flatnonzero(abs(radial-arc.radius_mm)<=band_mm)
    u=arc.start_direction@basis;v=np.cross(arc.plane_normal,arc.start_direction)@basis
    angle=np.mod(np.sign(arc.sweep_rad)*np.arctan2(relative[keep]@v,relative[keep]@u),2*np.pi)
    good=angle<=abs(arc.sweep_rad)
    q=np.r_[0.,arc.length_mm,angle[good]*arc.radius_mm]
    source=np.r_[-1,-2,keep[good]].astype(np.int64)
    positions,index=np.unique(q,return_index=True)
    assert np.isfinite(positions).all() and np.all((positions>=0)&(positions<=arc.length_mm))
    return positions,source[index]
