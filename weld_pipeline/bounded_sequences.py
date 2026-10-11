"""Equivalent features with nearest-neighbour queries bounded at the 50 mm clipping radius."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.robust_surface_planes import surface_candidates

FEATURE_VERSION='typed_contact_sequence_v4'
STEP_MM=2.
OFFSETS=np.array([-25.,-12.,-6.,0.,6.,12.,25.])
DEPTHS=np.array([-5.,0.,5.,15.,30.,50.,100.])


def make_sequences(xyz):
    candidates,meta=surface_candidates(xyz)
    center=(xyz.min(0)+xyz.max(0))/2;extent=np.ptp(xyz,axis=0)
    z,counts=np.unique(np.round(xyz[:,2],2),return_counts=True)
    levels=z[counts>=max(50,len(xyz)*.03)]
    frame=np.array(meta['xy_frame']);specs=[]
    for level in levels:
        near=xyz[np.abs(xyz[:,2]-level)<.015]
        for pi,p in enumerate(meta['vertical_planes']):
            if not p['z_range_mm'][0]-.05<=level<=p['z_range_mm'][1]+.05:continue
            n=np.array(p['normal_xy']);u=np.array(p['direction_xy'])
            line=near[np.abs(near[:,:2]@n-p['offset_mm'])<.12]
            if len(line)<3:continue
            t=line[:,:2]@u
            if np.ptp(t)<10:continue
            a=np.r_[n*p['offset_mm']+u*t.min(),level]
            b=np.r_[n*p['offset_mm']+u*t.max(),level]
            sign=-1. if level-p['z_range_mm'][0]>=p['z_range_mm'][1]-level else 1.
            specs.append(('horizontal',a,b,np.r_[n,0.],np.array([0.,0.,sign])))
    for c in candidates:
        if c['type']=='vertical':
            a,b=np.array(c['segments'][0]['points'])
            specs.append(('vertical',a,b,np.r_[frame[0],0.],np.r_[frame[1],0.]))
    tree=cKDTree(xyz);sequences=[]
    for typ,a,b,n1,n2 in specs:
        length=float(np.linalg.norm(b-a));u=(b-a)/length
        t=np.arange(-20,length+20+STEP_MM/2,STEP_MM)
        q=a[None]+t[:,None]*u
        offsets=np.array([n1*x+n2*y for y in (DEPTHS if typ=='horizontal' else OFFSETS) for x in OFFSETS])
        distances=tree.query((q[:,None]+offsets[None]).reshape(-1,3),workers=2,distance_upper_bound=50.)[0].reshape(len(q),-1)
        extra=np.column_stack([(q-center)/1000,np.tile(extent/1000,(len(q),1)),
            (t-t[0])/1000,(t[-1]-t)/1000,np.tile(np.r_[u,n1,n2],(len(q),1)),
            np.full(len(q),typ=='vertical'),q/10000])
        features=np.column_stack([np.minimum(distances,50)/50,extra]).astype(np.float32)
        sequences.append({'xyz':q.astype(np.float32),'features':features,'direction':u,
            'sample_step_mm':STEP_MM,'type':typ})
    return sequences,{**meta,'sequence_levels_mm':levels.tolist(),'sequence_count':len(sequences),
        'feature_points':sum(len(s['xyz']) for s in sequences),'feature_version':FEATURE_VERSION}

