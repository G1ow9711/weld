"""Uncertain contact proposals from observed samples on two fitted planes.

This supplements mesh proposals; it does NOT prove material contact. Regularly
sampled edge rows can survive triangulation holes. Unobserved long gaps are kept.
No target, STEP, source identity, or predicted answer is read.
"""
import numpy as np


def sampled_runs(values, minimum_samples=3):
    q=np.unique(np.round(values,6))
    if len(q)<minimum_samples:return []
    gaps=np.diff(q); breaks=[]
    for k,g in enumerate(gaps):
        # The gap being classified cannot inflate its own local spacing.
        before=gaps[max(0,k-4):k];after=gaps[k+1:k+5]
        neighbors=[float(np.median(v)) for v in (before,after) if len(v)]
        if neighbors and g>1.5*max(neighbors)+1e-5:breaks.append(k+1)
    return [v for v in np.split(q,breaks) if len(v)>=minimum_samples]


def proposals(faces,xyz):
    result=[]
    bounds=[(xyz[f['point_ids']].min(0),xyz[f['point_ids']].max(0)) for f in faces]
    for i,a in enumerate(faces):
        points=xyz[a['point_ids']]
        for j in range(i+1,len(faces)):
            if np.any(np.maximum(bounds[i][0],bounds[j][0])>np.minimum(bounds[i][1],bounds[j][1])+.4):continue
            b=faces[j];direction=np.cross(a['normal'],b['normal']);norm=np.linalg.norm(direction)
            if norm<.15:continue
            direction/=norm
            if direction[np.argmax(abs(direction))]<0:direction=-direction
            normals=np.array([a['normal'],b['normal']]);origin=normals.T@np.linalg.solve(normals@normals.T,[a['offset'],b['offset']])
            tolerance=max(1e-6,a.get('max_plane_residual_mm',0)+b.get('max_plane_residual_mm',0))
            shared=points[np.abs(points@b['normal']-b['offset'])<=tolerance]
            for q in sampled_runs((shared-origin)@direction):
                result.append(dict(representation=dict(kind='line',endpoints_mm=[(origin+q[0]*direction).tolist(),(origin+q[-1]*direction).tolist()]),face_indices=[i,j],length_mm=float(q[-1]-q[0]),status='UNCERTAIN_SAMPLED_CONTACT_CHAIN',observed_line_samples=len(q),material_pair=None,weld_type=None,requires_material_contact_classification=True,subsampling_gaps_unresolved=True))
    return result


def controls():
    row=np.arange(0.,201.,10.)
    runs=sampled_runs(row);assert len(runs)==1 and np.allclose(runs[0][[0,-1]],[0,200])
    cut=row[(row<=60)|(row>=120)]
    runs=sampled_runs(cut);assert len(runs)==2 and runs[0][-1]==60 and runs[1][0]==120
    assert sampled_runs(np.array([0.,100.]))==[]
    return dict(regular_rows_preserved=True,observed_large_gap_preserved=True,two_points_insufficient=True,material_identity_not_claimed=True)
