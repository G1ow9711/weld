"""Equivalent query batches using one reusable worker pool per point cloud."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.robust_surface_planes import surface_candidates
from weld_pipeline.bounded_sequences import STEP_MM,OFFSETS,DEPTHS,FEATURE_VERSION
from weld_pipeline.indexed_surface_features import PlaneIndex
from weld_pipeline.mixed_sequences import append_curved_contacts

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

def append_fast_surface_features(xyz,sequences):
    cache={};index=PlaneIndex(xyz)
    stats={'plane_trees':0,'empty_planes':0,'query_points':0}
    def distance(q,axis,offsets,cross):
        axis=np.asarray(axis,float);axis/=np.linalg.norm(axis)
        offset=float(q[0]@axis)
        if axis[np.argmax(np.abs(axis))]<0:axis=-axis;offset=-offset
        key=tuple(np.round(axis,9))+(round(offset,4),)
        if key not in cache:
            points=index.select(axis,offset)
            cache[key]=cKDTree(points) if len(points) else None
            stats['plane_trees']+=1;stats['empty_planes']+=int(not len(points))
        queries=(q[:,None]+offsets[None,:,None]*cross).reshape(-1,3)
        tree=cache[key]
        d=tree.query(queries,workers=8 if len(queries)>=32768 else 1,distance_upper_bound=50.)[0] if tree is not None else np.full(len(queries),50.)
        stats['query_points']+=len(queries)
        return np.minimum(d,50.).reshape(len(q),-1)/50.
    output=[]
    for seq in sequences:
        base=seq['features']
        if base.shape[1]!=70:raise ValueError('Expected original 70-channel features')
        q=seq['xyz'].astype(float)
        n1,n2=base[len(base)//2,60:63].astype(float),base[len(base)//2,63:66].astype(float)
        first=distance(q,n1,DEPTHS if seq['type']=='horizontal' else OFFSETS,n2)
        second=distance(q,n2,OFFSETS,n1)
        output.append({**seq,'features':np.column_stack([base,first,second]).astype(np.float32)})
    stats['projection_indexes']=len(index.indexes);stats['points_examined_for_plane_masks']=index.examined_points
    return output,stats


def make_fast_mixed_sequences(xyz):
    straight,geometry=make_fast_sequences(xyz)
    straight,surface=append_fast_surface_features(xyz,straight)
    sequences,curves=append_curved_contacts(xyz,straight)
    return sequences,{'straight_geometry':geometry,'surface':surface,'curves':curves,'optimization':'pooled_queries_indexed_planes'}
