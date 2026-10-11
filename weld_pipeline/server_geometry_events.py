"""Pure-geometry candidate events; supervision is owned by the training caller."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.contract_curve_geometry import CircularArc


def geometry(rep):
    if rep['kind']=='line':
        a,b=np.array(rep['endpoints_mm']);length=float(np.linalg.norm(b-a));v=(b-a)/length
        return length,lambda t:a+np.asarray(t)[...,None]*v,lambda t:np.broadcast_to(v,np.asarray(t).shape+(3,)),None
    arc=CircularArc.from_parameters(rep['arc_parameters']);sign=np.sign(arc.sweep_rad)
    return arc.length_mm,arc.point_at,lambda t:sign*np.cross(arc.plane_normal,(arc.point_at(t)-arc.center_mm)/arc.radius_mm),arc


def events(candidate,faces):
    rep=candidate['representation'];length,point,tangent,arc=geometry(rep)
    ns=np.array([f['normal'] for f in faces]);offset=np.array([f['offset'] for f in faces]);values=[0.,length]
    if arc is None:
        v=tangent(0.);den=ns@v;good=abs(den)>1e-8
        values.extend(((offset[good]-ns[good]@point(0.))/den[good]).tolist())
    else:
        u=arc.start_direction;v=np.cross(arc.plane_normal,u)
        aa=arc.radius_mm*(ns@u);bb=arc.radius_mm*(ns@v);rr=np.hypot(aa,bb);delta=offset-ns@arc.center_mm
        good=(rr>1e-8)&(abs(delta)<=rr)
        phi=np.arctan2(bb[good],aa[good]);theta=np.arccos(np.clip(delta[good]/rr[good],-1,1))
        for angles in (phi+theta,phi-theta):values.extend((((np.sign(arc.sweep_rad)*angles)%(2*np.pi))*arc.radius_mm).tolist())
    values=np.asarray(values);values=values[(values>=-1e-6)&(values<=length+1e-6)]
    # Numerical duplicate coalescing only, not a weld length filter.
    values=np.clip(np.unique(np.round(values,6)),0,length)
    return values


def event_features(candidate,faces,tree,positions=None):
    length,point,tangent,arc=geometry(candidate['representation']);s=events(candidate,faces) if positions is None else np.asarray(positions);q=point(s);v=tangent(s)
    normal=np.array(faces[candidate['face_indices'][0]]['normal']);normal=normal[None,:]-np.sum(v*normal,axis=1)[:,None]*v
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12);side=np.cross(v,normal)
    axes=[v,normal,side];features=[]
    for axis in axes:
        locations=q[:,None,:]+np.array([-35.,-15.,-5.,-1.,0.,1.,5.,15.,35.])[None,:,None]*axis[:,None,:]
        d=tree.query(locations.reshape(-1,3),workers=2)[0].reshape(len(q),9)
        features.append(np.log1p(d))
    ns=np.array([f['normal'] for f in faces]);offset=np.array([f['offset'] for f in faces]);dist=np.sort(abs(q@ns.T-offset),axis=1)[:,:6]
    dist=np.pad(dist,((0,0),(0,max(0,6-dist.shape[1]))),constant_values=1000)
    features.append(np.log1p(dist))
    features.append(np.stack([s/length,1-s/length,np.log1p(s),np.log1p(length-s),np.full(len(s),np.log1p(length)),np.full(len(s),0 if arc is None else 1/arc.radius_mm),np.full(len(s),arc is not None),np.full(len(s),len(candidate['face_indices'])),np.full(len(s),candidate.get('requires_material_contact_classification',False))],axis=1))
    return s,q,np.concatenate(features,axis=1).astype(np.float32)


def controls():
    faces=[dict(normal=[1,0,0],offset=5),dict(normal=[0,0,1],offset=0)]
    c=dict(representation=dict(kind='line',endpoints_mm=[[0,0,0],[10,0,0]]),face_indices=[1])
    assert np.allclose(events(c,faces),[0,5,10])
    arc=CircularArc([0,0,0],10,[0,0,1],[1,0,0],np.pi)
    c['representation']=dict(kind='arc',arc_parameters=arc.to_parameters())
    assert np.any(abs(events(c,faces)-10*np.pi/3)<1e-5)
    return dict(line_plane_event=True,circle_plane_event=True,no_supervision_read=True)
