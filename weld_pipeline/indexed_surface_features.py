"""Exact contact-plane filtering with sorted projection indexes for fewer scans."""
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.typed_sequences import OFFSETS,DEPTHS


class PlaneIndex:
    def __init__(self,xyz):
        self.xyz=xyz
        self.center=(xyz.min(0)+xyz.max(0))/2
        self.radius=np.maximum(np.abs(xyz.min(0)-self.center),np.abs(xyz.max(0)-self.center))
        self.coordinate_bound=max(1.,float(np.abs(xyz).max()))
        self.indexes=[]
        self.examined_points=0

    def select(self,axis,offset,tolerance=.15):
        axis=np.asarray(axis,float)
        if not np.isclose(np.linalg.norm(axis),1.,atol=1e-10):raise ValueError('Unit plane normal required')
        matches=[(np.linalg.norm(axis-ref),ref,values,ids) for ref,values,ids in self.indexes]
        choice=min(matches,key=lambda r:r[0]) if matches else None
        if choice is None or choice[0]>.01:
            ref=axis.copy();projection=(self.xyz-self.center)@ref
            ids=np.argsort(projection,kind='stable');values=projection[ids]
            self.indexes.append((ref,values,ids))
        else:_,ref,values,ids=choice
        relative_offset=offset-self.center@axis
        # AABB bound safely contains the entire oblique plane slab. The final
        # predicate is the original exact world-coordinate plane test.
        margin=float(self.radius@np.abs(axis-ref))+tolerance
        guard=64*np.finfo(float).eps*max(self.coordinate_bound,abs(offset))
        left,right=np.searchsorted(values,[relative_offset-margin-guard,relative_offset+margin+guard],side='left')
        selected=np.sort(ids[left:right])
        self.examined_points+=len(selected)
        candidates=self.xyz[selected]
        return candidates[np.abs(candidates@axis-offset)<tolerance]


def append_indexed_surface_features(xyz,sequences):
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
        d=tree.query(queries,workers=2,distance_upper_bound=50.)[0] if tree is not None else np.full(len(queries),50.)
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
