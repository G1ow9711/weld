"""Point-only geometric features for a learned contact/endpoint decoder."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.geometry_candidates import horizontal_candidates


FEATURE_VERSION='contact_sequence_v1'
STEP_MM=2.
NORMAL_OFFSETS=np.array([-25.,-12.,-6.,0.,6.,12.,25.])
DEPTHS=np.array([-5.,0.,5.,15.,30.,50.,100.])


def make_sequences(xyz):
    _,meta=horizontal_candidates(xyz)
    center=(xyz.min(0)+xyz.max(0))/2
    extent=np.ptp(xyz,axis=0)
    tree=cKDTree(xyz)
    sequences=[]
    for plane in meta['plane_lines']:
        normal=np.array(plane['normal_xy']);direction=np.array(plane['direction_xy'])
        if direction[np.argmax(np.abs(direction))]<0:direction=-direction
        lo,hi=sorted(np.array(plane['range_mm'])*np.sign(direction@np.array(plane['direction_xy'])))
        t=np.arange(lo-20,hi+20+STEP_MM/2,STEP_MM)
        q=np.column_stack([normal[None]*plane['offset_mm']+t[:,None]*direction[None],np.full(len(t),meta['base_z_mm'])])
        offsets=np.array([[*(normal*d),z*meta['web_sign']] for z in DEPTHS for d in NORMAL_OFFSETS])
        query=(q[:,None]+offsets[None]).reshape(-1,3)
        distances=tree.query(query,workers=2)[0].reshape(len(t),-1)
        extra=np.column_stack([(q-center)/1000,np.tile(extent/1000,(len(q),1)),
            (t-t[0])/1000,(t[-1]-t)/1000,np.tile(direction,(len(q),1))])
        feat=np.column_stack([np.minimum(distances,50)/50,extra]).astype(np.float32)
        sequences.append({'xyz':q.astype(np.float32),'features':feat,'direction':np.r_[direction,0.].astype(np.float32),'sample_step_mm':STEP_MM})
    return sequences,{k:v for k,v in meta.items() if k!='plane_lines'}


def training_targets(sequences,welds):
    """Explicit training/evaluation adapter. Never called by feature extraction."""
    targets=[];mapped=set()
    for seq in sequences:
        q=seq['xyz'];u=seq['direction'];t=(q-q[0])@u
        target=np.zeros((len(q),5),np.float32)
        for wi,w in enumerate(welds):
            if w['type']!='horizontal' or any(s['kind']!='line' for s in w['segments']):continue
            points=np.array([p for s in w['segments'] for p in [s['points'][0],s['points'][-1]]])
            residual=np.linalg.norm(np.cross(points-q[0],u),axis=1)
            if residual.max()>.2:continue
            position=(points-q[0])@u
            if position.min()<t[0]-1 or position.max()>t[-1]+1:continue
            mapped.add(wi)
            for si in range(len(w['segments'])):
                a,b=sorted(position[2*si:2*si+2]);target[(t>=a)&(t<=b),0]=1
            start,end=float(position.min()),float(position.max())
            for value,channel in [(start,1),(end,2)]:
                target[:,channel]=np.maximum(target[:,channel],np.exp(-.5*((t-value)/2.)**2))
            for s in w['segments'][:-1]:
                value=(np.array(s['points'][-1])-q[0])@u
                target[:,3]=np.maximum(target[:,3],np.exp(-.5*((t-value)/2.)**2))
            # Front/back are tied to the ORIGINAL global ends, not to every
            # internal segment. Weld-level Any flags are retained separately.
            for point,key in [(points[0],'WorkRangeFrontSplit'),(points[-1],'WorkRangeBackSplit')]:
                if w['metadata'][key]:
                    value=(point-q[0])@u
                    target[:,4]=np.maximum(target[:,4],np.exp(-.5*((t-value)/2.)**2))
        targets.append(target)
    return targets,{'total_reference_welds':len(welds),'mapped_to_sequences':len(mapped),
        'missing_reference_indices':sorted(set(range(len(welds)))-mapped)}
