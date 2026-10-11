"""Local finite-surface graph and typed curve query data, separate from labels."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier,curve_point_queries
from weld_pipeline.sjg_local_curve_supervision_915 import event_targets

class CanonicalQueryTree:
    """Query 1 mm representatives, returning original point/feature indices."""
    def __init__(self,xyz):
        keys=np.floor((xyz-xyz.min(0))/1.).astype(np.int32)
        _,self.original_ids=np.unique(keys,axis=0,return_index=True)
        self.tree=cKDTree(xyz[self.original_ids])

    def query(self,positions,k=16,workers=2):
        if k>len(self.original_ids):raise ValueError('Too few canonical observed points')
        distance,ids=self.tree.query(positions,k=k,workers=workers)
        return distance,self.original_ids[ids]


def neighborhood_controls():
    x,y=np.meshgrid(np.arange(9.),np.arange(9.));plane=np.c_[x.ravel(),y.ravel(),np.zeros(81)]
    dense=np.c_[np.linspace(3.9,4.1,1001),np.full(1001,4.),np.zeros(1001)]
    xyz=np.r_[plane,dense];tree=CanonicalQueryTree(xyz);distance,ids=tree.query([[4.,4.,0.]],k=16)
    assert np.all(ids<81) and np.allclose(distance,np.linalg.norm(xyz[ids]-np.array([4.,4.,0.]),axis=2))
    eigen=np.linalg.eigvalsh(np.cov(xyz[ids[0]].T));assert eigen[1]>.1
    return dict(original_feature_indices_preserved=True,collinear_density_does_not_collapse_query_patch=True,grid_mm=1.)


def graph_inputs(xyz,point,faces,support,profiles):
    feature=[]
    for face in faces:
        ids=support[face['point_ids_key']].astype(np.int64);e=point['embedding'][ids]
        n=np.asarray(face['normal']);center=np.asarray(face['center'])
        agreement=np.abs(point['normal'][ids]@n);edge=point['boundary'][ids]
        weight=.1+.9*agreement*(1-.75*edge);weight/=weight.sum()
        mean=(e*weight[:,None]).sum(0);std=np.sqrt(((e-mean)**2*weight[:,None]).sum(0))
        delta=xyz[ids]-center;residual=abs(delta@n)
        feature.append(np.r_[mean,std,n,np.sort(np.log1p(np.ptp(delta,axis=0)))/10,
            residual.mean(),np.quantile(residual,.95),np.log1p(len(ids))/12,
            edge.mean(),edge.max(),agreement.mean(),agreement.std(),float(face['proposal_branch']=='raw_geometry')])
    face_x=np.asarray(feature,np.float32);assert face_x.shape==(len(faces),46)
    centers=np.asarray([f['center'] for f in faces]);normals=np.asarray([f['normal'] for f in faces])
    neighbors=cKDTree(centers).query(centers,k=min(16,len(faces)))[1]
    if neighbors.ndim==1:neighbors=neighbors[:,None]
    relation=np.concatenate([(centers[neighbors]-centers[:,None])/1000,
        np.abs((normals[neighbors]*normals[:,None]).sum(-1))[...,None]],axis=-1).astype(np.float32)
    pairs=[];geometry=[]
    for profile in profiles:
        curve=CurveCarrier(profile['representation']);ids=profile['face_indices'];a=ids[0];b=ids[1] if len(ids)>1 else a
        pairs.append([a,b]);mid=curve.at(curve.length/2);tangent=curve.tangent(curve.length/2)
        normal=np.asarray(profile['support_normal'],float);normal-=float(normal@tangent)*tangent
        norm=np.linalg.norm(normal)
        if norm<1e-8:raise ValueError('Support normal parallel to observed curve')
        normal/=norm;frame=np.stack([tangent,np.cross(normal,tangent),normal],axis=-1)
        coverage=sum(r-l for l,r in profile['support_intervals_mm'])/curve.length
        gap=max((r-l for l,r in profile['unobserved_gaps_mm']),default=0.)/curve.length
        rel_a=(centers[a]-mid)@frame/1000
        rel_b=(centers[b]-mid)@frame/1000 if len(ids)>1 else np.zeros(3)
        ends=curve.at(np.array([0.,curve.length]));tans=curve.tangent(np.array([0.,curve.length]))
        geo=np.r_[float(curve.kind=='arc'),np.log1p(curve.length)/10,
            100/curve.arc.radius_mm if curve.kind=='arc' else 0.,
            float(curve.kind=='arc' and curve.arc.is_closed),np.linalg.norm(ends[1]-ends[0])/curve.length,
            coverage,gap,float(len(ids)>1),abs(normals[a]@normals[b]) if len(ids)>1 else 0.,float(tans[0]@tans[1]),
            rel_a,rel_b,normals[a]@frame,float(profile['proposal_branch']=='raw_geometry')]
        assert geo.shape==(20,);geometry.append(geo)
    geometry=np.asarray(geometry,np.float32).reshape(-1,20);pairs=np.asarray(pairs,np.int64).reshape(-1,2)
    assert np.isfinite(face_x).all() and np.isfinite(geometry).all()
    return dict(face_x=face_x,neighbors=neighbors.astype(np.int64),relation=relation,pairs=pairs,geometry=geometry)

def queries(profiles,nodes,q,xyz,point,tree):
    nodes=np.asarray(nodes,np.int64);q=np.asarray(q,float)
    patch=np.empty((len(q),16,23),np.float32);query=np.empty((len(q),10),np.float32)
    for node in np.unique(nodes):
        ids=np.flatnonzero(nodes==node);p=profiles[int(node)]
        data=curve_point_queries(p['representation'],q[ids],xyz,point,tree,p['support_normal'])
        patch[ids]=data['patch']
        # Native workpiece position is an explicit input, not a source ID or
        # retrieved CAD attribute. The local reference frame is user-specified.
        query[ids]=np.c_[data['query'],data['positions_mm']/1000]
    return patch,query

def sample_training_queries(profiles,supervision,seed,count=2048):
    rng=np.random.default_rng(seed);children=supervision['children'];examples=[]
    if not profiles:raise ValueError('No input-generated curve carriers')
    # Guarantee every representable child has endpoint and interior examples.
    for child in children:
        node=int(rng.choice(child['profiles']))
        event=next(e for e in supervision['by_profile'][str(node)] if e['original_child_id']==child['original_child_id'])
        curve=CurveCarrier(profiles[node]['representation'])
        for position in (event['lo'],event['hi'],event['lo']+event['span_mm']/2):
            if curve.kind=='arc' and curve.arc.is_closed:position%=curve.length
            examples.append((node,float(np.clip(position,0,curve.length))))
    positive_budget=max(count//2,len(examples)) if children else 0
    while len(examples)<positive_budget:
        child=children[int(rng.integers(len(children)))];node=int(rng.choice(child['profiles']))
        event=next(e for e in supervision['by_profile'][str(node)] if e['original_child_id']==child['original_child_id'])
        curve=CurveCarrier(profiles[node]['representation'])
        if rng.random()<.5:position=float(rng.choice([event['lo'],event['hi']]))+rng.uniform(-8,8)
        else:position=event['lo']+rng.uniform(0,event['span_mm'])
        if curve.kind=='arc' and curve.arc.is_closed:position%=curve.length
        examples.append((node,float(np.clip(position,0,curve.length))))
    # Include all carriers as eligible backgrounds; events on positive carriers
    # remain correctly labeled when sampled here.
    lengths=np.asarray([p['length_mm'] for p in profiles],float);prob=np.sqrt(lengths)/np.sqrt(lengths).sum()
    while len(examples)<max(count,positive_budget*2):
        node=int(rng.choice(len(profiles),p=prob));examples.append((node,float(rng.uniform(0,lengths[node]))))
    nodes=np.asarray([a for a,b in examples],np.int64);q=np.asarray([b for a,b in examples],float)
    target=np.zeros((len(q),5),np.float32);mask=np.ones((len(q),5),bool)
    offset=np.zeros((len(q),2),np.float32);offset_mask=np.zeros((len(q),2),bool)
    mask[:,4]=False
    for node in np.unique(nodes):
        ids=np.flatnonzero(nodes==node);events=supervision['by_profile'].get(str(int(node)),[])
        y,o,m=event_targets(profiles[int(node)],events,q[ids]);target[ids,:4]=y;offset[ids]=o;offset_mask[ids]=m
        directions=np.zeros((len(ids),2),bool)
        for event in events:
            inside=np.zeros(len(ids),bool)
            for lo,hi in event['intervals']:inside|=(q[ids]>=lo)&(q[ids]<=hi)
            directions[inside,int(event['target_direction_forward'])]=True
        known=directions.sum(1)==1;target[ids[known],4]=directions[known,1];mask[ids[known],4]=True
    return dict(nodes=nodes,q_mm=q,target=target,target_mask=mask,offset=offset,offset_mask=offset_mask,
                represented_children=len(children),original_G=supervision['original_G'])
