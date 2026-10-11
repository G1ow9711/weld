"""Extend input-fitted arcs only along their observed connected boundary."""
import numpy as np
from weld_pipeline.sjg_polygonal_arcs_921 import fit_connected_polyline
from weld_pipeline.sjg_planar_boundary_curves_894 import boundary_chains
from weld_pipeline.contract_curve_geometry import CircularArc


def extend_run(sample, fitted, tolerance=3.):
    center, radius = fitted['center'], fitted['radius']
    radial = sample-center
    angle = np.arctan2(radial[:, 1], radial[:, 0])
    error = abs(np.linalg.norm(radial, axis=1)-radius)
    delta = (np.diff(angle)+np.pi) % (2*np.pi)-np.pi
    direction = np.sign(fitted['sweep'])
    bound = np.maximum(error[:-1], error[1:])+2*radius*np.sin(abs(delta)/4)**2
    good = (direction*delta >= -1e-8) & (bound <= tolerance)
    lo, hi = fitted['raw_start'], fitted['raw_stop']-1
    while lo > 0 and good[lo-1]:
        lo -= 1
    while hi+1 < len(sample) and good[hi]:
        hi += 1
    sweep = float(delta[lo:hi].sum())
    if not 0 < abs(sweep) <= 2*np.pi+1e-8:
        return fitted
    closed = bool(np.linalg.norm(sample[lo]-sample[hi]) < 1e-9 and abs(abs(sweep)-2*np.pi) < 1e-8)
    if closed:
        sweep = float(np.sign(sweep)*2*np.pi)
    if not closed and abs(sweep) >= 2*np.pi:
        return fitted
    return dict(fitted, raw_start=lo, raw_stop=hi+1, sweep=sweep, closed=closed,
                polyline_bidirectional_bound_mm=float(bound[lo:hi].max()),
                extension_vertices=int(fitted['raw_start']-lo+hi+1-fitted['raw_stop']))


def face_arcs(xyz, face):
    ids=np.asarray(face['point_ids'],np.int64); points=xyz[ids]
    basis=np.asarray(face['basis']); origin=np.asarray(face['center'])
    uv=(points-origin)@basis; normal=np.cross(basis[:,0],basis[:,1]); normal/=np.linalg.norm(normal)
    chains,stats=boundary_chains(face['triangles']); arcs=[]
    for ci,chain in enumerate(chains):
        if len(chain)<4: continue
        sample=uv[chain]; seeds,_=fit_connected_polyline(sample)
        for seed in seeds:
            c=extend_run(sample,seed)
            if not c.get('extension_vertices',0): continue
            radial=sample[c['raw_start']]-c['center'];radial/=np.linalg.norm(radial)
            arc=CircularArc(origin+basis@c['center'],c['radius'],normal,basis@radial,c['sweep'],c['closed'])
            arcs.append(dict(representation=dict(kind='arc',arc_parameters=arc.to_parameters()),length_mm=arc.length_mm,
                support_normal=normal.tolist(),support_intervals_mm=[[0.,arc.length_mm]],unobserved_gaps_mm=[],
                material_pair=None,weld_type=None,second_contact_face_unknown=True,
                status='UNCLASSIFIED_OBSERVED_BOUNDARY_EXTENSION',
                polyline_bidirectional_bound_mm=c['polyline_bidirectional_bound_mm'],
                raw_support_point_ids=ids[chain[c['raw_start']:c['raw_stop']]].tolist(),boundary_chain=ci,
                extension_vertices=c['extension_vertices'],connected_mesh_boundary_preserved=True,
                disconnected_chains_not_joined=True))
    return arcs,dict(**stats,extended_arcs=len(arcs))


def controls():
    theta=np.linspace(0,.7,401);sample=400*np.c_[np.cos(theta),np.sin(theta)]
    seed=dict(center=np.zeros(2),radius=400.,raw_start=100,raw_stop=301,sweep=.35,closed=False)
    extended=extend_run(sample,seed)
    assert extended['raw_start']==0 and extended['raw_stop']==401
    bad=sample.copy();bad[350:]+=100
    clipped=extend_run(bad,seed)
    assert clipped['raw_stop']==350
    reverse=sample.copy();reverse[350:]=sample[349:298:-1]
    assert extend_run(reverse,seed)['raw_stop']<=351
    loop_theta=np.linspace(0,2*np.pi,4001);loop=400*np.c_[np.cos(loop_theta),np.sin(loop_theta)]
    closed=extend_run(loop,seed)
    assert closed['closed'] and abs(closed['sweep'])==2*np.pi
    CircularArc([0,0,0],400,[0,0,1],[1,0,0],closed['sweep'],True)
    return dict(observed_extent_recovered=True,off_circle_break_preserved=True,direction_reversal_stops_extension=True,
                closed_loop_roundoff_normalized=True)
