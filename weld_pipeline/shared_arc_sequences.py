"""Arc sequences with the same local 84 feature meanings as straight contacts."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.arc_surface_candidates import arc_surface_candidates
from weld_pipeline.typed_sequences import OFFSETS,DEPTHS,training_targets
from weld_pipeline.typed_contact_model import decode_typed

FEATURE_VERSION='typed_contact_arc_surface_v1'


def arc_points(seq,position_mm):
    curve=seq['curve'];t=np.asarray(position_mm,float)
    angle=curve['sequence_start_angle_rad']+t/curve['radius_mm']
    xy=np.asarray(curve['center_xy_mm'])+curve['radius_mm']*np.column_stack([np.cos(angle),np.sin(angle)])
    return np.column_stack([xy,np.full(len(t),curve['height_mm'])])


def arc_positions(seq,points):
    curve=seq['curve'];p=np.asarray(points,float);center=np.asarray(curve['center_xy_mm'])
    angle=np.arctan2(p[:,1]-center[1],p[:,0]-center[0])
    middle=curve['sequence_start_angle_rad']+(len(seq['xyz'])-1)*seq['sample_step_mm']/curve['radius_mm']/2
    angle=middle+(angle-middle+np.pi)%(2*np.pi)-np.pi
    positions=(angle-curve['sequence_start_angle_rad'])*curve['radius_mm']
    errors=np.linalg.norm(arc_points(seq,positions)-p,axis=1)
    return positions,errors


def make_arc_sequences(xyz,step_mm=2.,candidates_and_stats=None):
    if not np.isfinite(step_mm) or step_mm<=0:raise ValueError('Positive finite step required')
    xyz=np.asarray(xyz,float);candidates,stats=arc_surface_candidates(xyz) if candidates_and_stats is None else candidates_and_stats
    if not candidates:return [],{**stats,'feature_points':0,'feature_version':FEATURE_VERSION}
    center=(xyz.min(0)+xyz.max(0))/2;extent=np.ptp(xyz,axis=0)
    tree=cKDTree(xyz);plane_trees={};cylinder_trees={};sequences=[]
    def distances(tree,queries):
        if tree is None:return np.full(queries.shape[:-1],1.,np.float32)
        d=tree.query(queries.reshape(-1,3),workers=2,distance_upper_bound=50.)[0]
        return (np.minimum(d,50.)/50).reshape(queries.shape[:-1])
    for candidate in candidates:
        arc=candidate['arc'];height=candidate['height_mm'];radius=arc['radius_mm']
        length=radius*arc['sweep_rad'];pad=min(20.,(2*np.pi-arc['sweep_rad'])*radius/4)
        t=np.arange(0,length+2*pad+step_mm/2,step_mm)
        curve={**arc,'height_mm':height,'sequence_start_angle_rad':arc['start_angle_rad']-pad/radius}
        seq={'curve':curve,'sample_step_mm':step_mm,'type':'horizontal','curve_kind':'arc'}
        q=arc_points(seq,t);angle=curve['sequence_start_angle_rad']+t/radius
        n1=np.column_stack([np.cos(angle),np.sin(angle),np.zeros(len(q))])
        u=np.column_stack([-np.sin(angle),np.cos(angle),np.zeros(len(q))])
        sign=-1. if height-arc['z_range_mm'][0]>=arc['z_range_mm'][1]-height else 1.
        n2=np.tile([0.,0.,sign],(len(q),1))
        offset1,offset2=np.meshgrid(OFFSETS,DEPTHS)
        queries=q[:,None]+n1[:,None]*offset1.ravel()[None,:,None]+n2[:,None]*offset2.ravel()[None,:,None]
        near=distances(tree,queries)
        extra=np.column_stack([(q-center)/1000,np.tile(extent/1000,(len(q),1)),
                               t/1000,(t[-1]-t)/1000,u,n1,n2,np.zeros(len(q)),q/10000])
        ci=candidate['circle_index']
        if ci not in cylinder_trees:
            radial=np.linalg.norm(xyz[:,:2]-arc['center_xy_mm'],axis=1)
            p=xyz[np.abs(radial-arc.get('cylinder_radius_mm',radius))<.15]
            cylinder_trees[ci]=cKDTree(p) if len(p) else None
        if height not in plane_trees:
            p=xyz[np.abs(xyz[:,2]-height)<.15]
            plane_trees[height]=cKDTree(p) if len(p) else None
        first=distances(cylinder_trees[ci],q[:,None]+DEPTHS[None,:,None]*n2[:,None])
        second=distances(plane_trees[height],q[:,None]+OFFSETS[None,:,None]*n1[:,None])
        seq.update(xyz=q.astype(np.float32),features=np.column_stack([near,extra,first,second]).astype(np.float32),
                   direction=u[len(u)//2],candidate_id=candidate['id'])
        sequences.append(seq)
    return sequences,{**stats,'feature_points':sum(len(s['xyz']) for s in sequences),'feature_version':FEATURE_VERSION}


def parameter_sequence(seq):
    n=len(seq['xyz']);t=np.arange(n)*seq['sample_step_mm']
    return {**seq,'xyz':np.column_stack([t,np.zeros((n,2))]).astype(np.float32),'direction':np.array([1.,0.,0.])}


def arc_training_targets(sequences,welds,tolerance_mm=.25):
    grouped=[[] for _ in sequences];mapping={};unmapped=[]
    for wi,weld in enumerate(welds):
        if not weld['segments'] or any(s['kind']!='arc' for s in weld['segments']):
            unmapped.append({'index':wi,'id':weld['id'],'reason':'arc-only mapper'});continue
        points=np.array([p for s in weld['segments'] for p in s['points']])
        choices=[]
        for si,seq in enumerate(sequences):
            if seq['type']!=weld['type']:continue
            position,error=arc_positions(seq,points)
            if error.max()>tolerance_mm or position.min()<0 or position.max()>(len(seq['xyz'])-1)*seq['sample_step_mm']:continue
            choices.append((float(error.max()),si))
        if not choices:
            unmapped.append({'index':wi,'id':weld['id'],'reason':'no arc sequence represents full reference'});continue
        _,si=min(choices);mapping[wi]=si;converted=[]
        for s in weld['segments']:
            t,_=arc_positions(sequences[si],[s['points'][0],s['points'][-1]])
            converted.append({'kind':'line','points':[[float(t[0]),0.,0.],[float(t[1]),0.,0.]]})
        grouped[si].append({**weld,'segments':converted})
    targets=[]
    for seq,group in zip(sequences,grouped):
        y,_=training_targets([parameter_sequence(seq)],group);targets.append(y[0])
    return targets,{'total_reference_welds':len(welds),'mapped_to_sequences':len(mapping),
                    'weld_to_sequence':mapping,'unmapped':unmapped}


def decode_curved(sequences,probabilities,threshold=.5):
    result=[]
    for seq,prob in zip(sequences,probabilities):
        if seq.get('curve_kind')!='arc':
            decoded=decode_typed([seq],[prob],threshold)
        else:
            decoded=decode_typed([parameter_sequence(seq)],[prob],threshold)
            for weld in decoded:
                for segment in weld['segments']:
                    a,b=np.asarray(segment['points'])[:,0]
                    positions=np.linspace(a,b,max(2,int(np.ceil(abs(b-a)))+1))
                    segment.update(kind='arc',points=arc_points(seq,positions).tolist(),length_mm=float(abs(b-a)))
        for weld in decoded:
            weld['id']=f'weld_{len(result)+1:04d}';result.append(weld)
    return result
