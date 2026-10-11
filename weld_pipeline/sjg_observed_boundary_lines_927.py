"""Straight carriers from connected observed finite-plane boundaries."""
import numpy as np
from weld_pipeline.sjg_polygonal_arcs_921 import simplify_indices
from weld_pipeline.sjg_planar_boundary_curves_894 import boundary_chains


def polyline_lines(sample,tolerance=.2):
    vertices=simplify_indices(sample,tolerance);segments=[]
    for lo,hi in zip(vertices[:-1],vertices[1:]):
        a,b=sample[lo],sample[hi];vector=b-a;length=float(np.linalg.norm(vector))
        if length<1.:continue
        points=sample[lo:hi+1];q=(points-a)@vector/(length*length)
        distance=np.linalg.norm(points-(a+np.clip(q,0,1)[:,None]*vector),axis=1)
        if distance.max()>tolerance+1e-8 or np.any(np.diff(q)<-1e-8):continue
        segments.append(dict(lo=int(lo),hi=int(hi),endpoints=np.stack([a,b]),length=length,
                             boundary_deviation_mm=float(distance.max())))
    return segments


def face_lines(xyz,face):
    ids=np.asarray(face['point_ids'],np.int64);basis=np.asarray(face['basis']);origin=np.asarray(face['center'])
    uv=(xyz[ids]-origin)@basis;chains,_=boundary_chains(face['triangles']);profiles=[]
    for ci,chain in enumerate(chains):
        for segment in polyline_lines(uv[chain]):
            length=segment['length'];endpoints=origin+segment['endpoints']@basis.T
            profiles.append(dict(representation=dict(kind='line',endpoints_mm=endpoints.tolist()),length_mm=length,
                support_normal=np.asarray(face['normal']).tolist(),support_intervals_mm=[[0.,length]],unobserved_gaps_mm=[],
                material_pair=None,weld_type=None,second_contact_face_unknown=True,
                status='UNCLASSIFIED_OBSERVED_FINITE_PLANE_BOUNDARY_LINE',
                boundary_deviation_mm=segment['boundary_deviation_mm'],boundary_chain=ci,
                raw_support_point_ids=ids[chain[segment['lo']:segment['hi']+1]].tolist(),
                connected_mesh_boundary_preserved=True,disconnected_chains_not_joined=True))
    return profiles


def controls():
    straight=np.c_[np.linspace(0,100,101),np.zeros(101)]
    result=polyline_lines(straight);assert len(result)==1 and result[0]['length']==100
    corner=np.r_[straight,np.c_[np.full(100,100),np.arange(1,101)]]
    result=polyline_lines(corner);assert len(result)==2 and all(s['length']==100 for s in result)
    return dict(straight_recovered=True,right_angle_not_bridged=True)
