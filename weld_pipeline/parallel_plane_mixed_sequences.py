"""Equivalent query batches using one reusable worker pool per point cloud."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.parallel_surface_planes import surface_candidates
from weld_pipeline.bounded_sequences import STEP_MM,OFFSETS,DEPTHS,FEATURE_VERSION
from weld_pipeline.indexed_surface_features import PlaneIndex
from weld_pipeline.shared_curved_contacts import append_curved_contacts

def make_fast_sequences(xyz):
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
    tree=cKDTree(xyz)
    def make_one(spec):
        typ,a,b,n1,n2=spec
        length=float(np.linalg.norm(b-a));u=(b-a)/length
        t=np.arange(-20,length+20+STEP_MM/2,STEP_MM)
        q=a[None]+t[:,None]*u
        offsets=np.array([n1*x+n2*y for y in (DEPTHS if typ=='horizontal' else OFFSETS) for x in OFFSETS])
        queries=(q[:,None]+offsets[None]).reshape(-1,3)
        distances=tree.query(queries,workers=1,distance_upper_bound=50.)[0].reshape(len(q),-1)
        extra=np.column_stack([(q-center)/1000,np.tile(extent/1000,(len(q),1)),
            (t-t[0])/1000,(t[-1]-t)/1000,np.tile(np.r_[u,n1,n2],(len(q),1)),
            np.full(len(q),typ=='vertical'),q/10000])
        features=np.column_stack([np.minimum(distances,50)/50,extra]).astype(np.float32)
        return {'xyz':q.astype(np.float32),'features':features,'direction':u,
            'sample_step_mm':STEP_MM,'type':typ}
    with ThreadPoolExecutor(max_workers=8) as pool:
        sequences=list(pool.map(make_one,specs))
    return sequences,{**meta,'sequence_levels_mm':levels.tolist(),'sequence_count':len(sequences),
        'feature_points':sum(len(s['xyz']) for s in sequences),'feature_version':FEATURE_VERSION}

from weld_pipeline.pooled_plane_mixed_sequences import append_fast_surface_features


def make_fast_mixed_sequences(xyz):
    straight,geometry=make_fast_sequences(xyz)
    straight,surface=append_fast_surface_features(xyz,straight)
    sequences,curves=append_curved_contacts(xyz,straight)
    return sequences,{'straight_geometry':geometry,'surface':surface,'curves':curves,'optimization':'parallel_plane_neighbors_and_reused_query_pools'}
