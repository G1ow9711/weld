"""Finite third-surface intersections and endpoint context from input geometry."""
import numpy as np

SUPPORT_TOLERANCE_MM=3.
FEATURES=12


def local_support_distance(triangles,point,tolerance=SUPPORT_TOLERANCE_MM):
    """Exact 2D distance for triangles whose expanded boxes contain the point."""
    candidate=((point>=triangles.min(1)-tolerance)&(point<=triangles.max(1)+tolerance)).all(1)
    v=triangles[candidate]
    if not len(v):return float('inf')
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    denominator=cross(v[:,1]-v[:,0],v[:,2]-v[:,0]);valid=np.abs(denominator)>1e-12
    v=v[valid];denominator=denominator[valid]
    if not len(v):return float('inf')
    a=cross(point-v[:,0],v[:,2]-v[:,0])/denominator
    b=cross(v[:,1]-v[:,0],point-v[:,0])/denominator
    if ((a>=0)&(b>=0)&(a+b<=1)).any():return 0.
    distance=float('inf')
    for i,j in ((0,1),(1,2),(2,0)):
        edge=v[:,j]-v[:,i]
        t=np.clip(((point-v[:,i])*edge).sum(1)/(edge*edge).sum(1),0,1)
        distance=min(distance,float(np.linalg.norm(point-v[:,i]-t[:,None]*edge,axis=1).min()))
    return distance


def finite_events(xyz,faces,profiles,profile_ids,support_loader,progress=None):
    ids=np.asarray(profile_ids,np.int64);selected=[profiles[int(i)] for i in ids]
    origin=np.asarray([p['p0'] for p in selected],float).reshape(-1,3)
    direction=np.asarray([p['direction'] for p in selected],float).reshape(-1,3)
    lengths=np.asarray([p['length'] for p in selected])
    pairs=np.asarray([p['face_indices'] for p in selected],np.int64).reshape(-1,2)
    normals=np.asarray([f['normal'] for f in faces]);events={int(i):[] for i in ids}
    if not len(ids):return events
    for fi,face in enumerate(faces):
        normal=normals[fi];denominator=direction@normal
        eligible=(np.abs(denominator)>1e-6)&(pairs!=fi).all(1)
        q=np.zeros(len(ids));q[eligible]=(float(face['offset'])-origin[eligible]@normal)/denominator[eligible]
        eligible &= (q>=0)&(q<=lengths)
        indices=np.flatnonzero(eligible)
        if not len(indices):continue
        support,triangles=support_loader(face);points=xyz[np.asarray(support,np.int64)]
        intersections=origin[indices]+q[indices,None]*direction[indices]
        box=((intersections>=points.min(0)-SUPPORT_TOLERANCE_MM)&(intersections<=points.max(0)+SUPPORT_TOLERANCE_MM)).all(1)
        indices=indices[box];intersections=intersections[box]
        basis=np.asarray(face['basis']);center=np.asarray(face['center'])
        uv=(points-center)@basis;mesh=uv[np.asarray(triangles,np.int64)]
        projected=(intersections-center)@basis
        for row,point in zip(indices,projected):
            distance=local_support_distance(mesh,point)
            if distance>SUPPORT_TOLERANCE_MM:continue
            cosine=np.sort(np.abs(normals[pairs[row]]@normal))
            events[int(ids[row])].append(dict(q_mm=float(q[row]),face=fi,
                axial_cosine=float(abs(denominator[row])),support_distance_mm=distance,
                pair_cosine_min=float(cosine[0]),pair_cosine_max=float(cosine[1])))
        if progress and fi%50==0:progress('THIRD_SURFACE',fi+1,len(faces))
    for values in events.values():
        values.sort(key=lambda e:(e['q_mm'],e['support_distance_mm'],e['axial_cosine'],e['pair_cosine_min'],e['pair_cosine_max']))
    return events


def adjacency_features(q,events):
    q=np.asarray(q);x=np.zeros((len(q),FEATURES),np.float32)
    # Per side: exists, distance/100mm capped at10, axial cosine,
    # finite support distance/3mm, min/max normal agreement to the pair.
    x[:,1]=x[:,7]=10.
    if not events:return x
    positions=np.asarray([e['q_mm'] for e in events])
    for side in (0,1):
        index=np.searchsorted(positions,q,side='right' if side==0 else 'left')-(side==0)
        valid=(index>=0)&(index<len(events));at=index[valid]
        values=np.asarray([[e['axial_cosine'],e['support_distance_mm']/SUPPORT_TOLERANCE_MM,
            e['pair_cosine_min'],e['pair_cosine_max']] for e in events],np.float32)
        start=6*side;x[valid,start]=1
        x[valid,start+1]=np.minimum(abs(q[valid]-positions[at])/100.,10.)
        x[valid,start+2:start+6]=values[at]
    assert np.isfinite(x).all()
    return x
