"""Finite candidate envelopes, retaining observed support and unknown gaps."""
import numpy as np
from scipy.spatial import ConvexHull


def clip_line(equations,position,direction):
    a=equations[:,:2]@direction;b=equations[:,:2]@position+equations[:,2]
    parallel=abs(a)<1e-12
    if np.any(b[parallel]>1e-7):return None
    lo=float(np.max(-b[a< -1e-12]/a[a< -1e-12],initial=-np.inf))
    hi=float(np.min(-b[a>1e-12]/a[a>1e-12],initial=np.inf))
    return (lo,hi) if np.isfinite([lo,hi]).all() and hi>lo else None


def complement(intervals,length):
    gaps=[];previous=0.
    for lo,hi in sorted(intervals):
        if lo>previous:gaps.append([previous,lo])
        previous=max(previous,hi)
    if previous<length:gaps.append([previous,length])
    return gaps


def envelope_profiles(xyz,faces,previous,progress=None):
    hulls=[];bounds=[]
    for face in faces:
        p=xyz[face['point_ids']];basis=np.asarray(face['basis']);center=np.asarray(face['center'])
        hulls.append(ConvexHull((p-center)@basis).equations);bounds.append((p.min(0),p.max(0)))
    old={tuple(p['face_indices']):p for p in previous if p['representation']['kind']=='line' and len(p['face_indices'])==2}
    result=[]
    for i,a in enumerate(faces):
        for j in range(i+1,len(faces)):
            if np.any(np.maximum(bounds[i][0],bounds[j][0])>np.minimum(bounds[i][1],bounds[j][1])+.4):continue
            b=faces[j];direction=np.cross(a['normal'],b['normal']);norm=float(np.linalg.norm(direction))
            if norm<.15:continue
            direction/=norm
            if direction[np.argmax(abs(direction))]<0:direction=-direction
            normals=np.stack([a['normal'],b['normal']]);origin=normals.T@np.linalg.solve(normals@normals.T,[a['offset'],b['offset']])
            ranges=[]
            for k,f in ((i,a),(j,b)):
                basis=np.asarray(f['basis']);ranges.append(clip_line(hulls[k],(origin-f['center'])@basis,direction@basis))
            if any(r is None for r in ranges):continue
            lo=max(r[0] for r in ranges);hi=min(r[1] for r in ranges);length=hi-lo
            if length<1.:continue
            support=[]
            if (i,j) in old:
                p=old[i,j];start=np.asarray(p['representation']['endpoints_mm'][0]);shift=float((start-origin)@direction-lo)
                for left,right in p['support_intervals_mm']:
                    left=max(0.,left+shift);right=min(length,right+shift)
                    if right>left:support.append([left,right])
            result.append(dict(representation=dict(kind='line',endpoints_mm=np.stack([origin+lo*direction,origin+hi*direction]).tolist()),
                length_mm=length,face_indices=[i,j],support_normal=np.asarray(a['normal']).tolist(),
                support_intervals_mm=support,unobserved_gaps_mm=complement(support,length),material_pair=None,weld_type=None,
                proposal_branch='finite_convex_envelope',status='UNCLASSIFIED_FINITE_ENVELOPE_WITH_UNKNOWN_SUPPORT',
                envelope_is_not_certified_material_boundary=True,unobserved_gaps_not_filled_in_support=True))
        if progress and i%20==0:progress('FINITE_ENVELOPE',i+1,len(faces),'CARRIERS',len(result))
    return result


def controls():
    points=np.array([[0.,0.],[100.,0.],[100.,20.],[0.,20.]])
    equations=ConvexHull(points).equations
    assert np.allclose(clip_line(equations,np.array([0.,10.]),np.array([1.,0.])),[0,100])
    assert clip_line(equations,np.array([0.,21.]),np.array([1.,0.])) is None
    assert complement([[20.,40.],[60.,80.]],100.)==[[0.,20.],[40.,60.],[80.,100.]]
    return dict(finite_line_clipped_to_observed_envelope=True,off_plane_region_rejected=True,
                leading_internal_and_trailing_unknown_gaps_preserved=True)
