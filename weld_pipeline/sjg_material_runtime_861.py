"""XYZ-only within-block material affinities stitched by original point indices."""
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from weld_pipeline.sjg_ptv3_runtime_807 import core_partitions
from weld_pipeline.sjg_ptv3_801 import voxelize

RANKS=(1,2,4,8,16,32,64,128)


def select_neighbors(tree,xyz,core):
    _,near=tree.query(xyz[core],k=min(129,len(xyz)),workers=2)
    if near.ndim==1:near=near[:,None]
    valid=near!=core[:,None];rank=valid.cumsum(1);chosen=np.full((len(core),len(RANKS)),-1,np.int64)
    for j,target in enumerate(RANKS):
        hit=valid&(rank==target);exists=hit.any(1);chosen[exists,j]=near[exists,hit[exists].argmax(1)]
    return chosen


def affinity_components(neighbors,distance,threshold=.5):
    known=(neighbors>=0)&np.isfinite(distance);same=known&(distance<=threshold)
    a=np.broadcast_to(np.arange(len(neighbors))[:,None],neighbors.shape)[same];b=neighbors[same]
    graph=csr_matrix((np.ones(len(a),np.uint8),(a,b)),shape=(len(neighbors),len(neighbors)))
    count,component=connected_components(graph,directed=False,return_labels=True)
    return component.astype(np.int32),int(count)


def infer_material_relations(model,xyz_mm,progress=None,resource_check=None):
    xyz=np.asarray(xyz_mm,np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<16 or not np.isfinite(xyz).all():raise ValueError('finite XYZ required')
    tree=cKDTree(xyz);parts=list(core_partitions(xyz));seen=np.zeros(len(xyz),np.uint8)
    neighbors=np.full((len(xyz),len(RANKS)),-1,np.int32);material_distance=np.full(neighbors.shape,np.nan,np.float32)
    surface_distance=np.full(neighbors.shape,np.nan,np.float32);normal=np.empty((len(xyz),3),np.float32)
    boundary=np.empty(len(xyz),np.float32);core_block=np.full(len(xyz),-1,np.int32);model.eval()
    with torch.inference_mode():
        for bi,core in enumerate(parts):
            if resource_check:resource_check()
            near=np.asarray(tree.query(xyz[core].mean(0),k=min(8192,len(xyz)))[1],np.int64).reshape(-1)
            context=near[~np.isin(near,core)][:8192-len(core)];ids=np.r_[core,context].astype(np.int64)
            block=voxelize(xyz[ids]);surface,normals,edges,material=model(block);count=len(core)
            normals=normals[:count].cpu().numpy();normal[core]=normals;boundary[core]=edges[:count].sigmoid().cpu().numpy()
            local=np.full(len(xyz),-1,np.int32);local[ids]=np.arange(len(ids),dtype=np.int32)
            selected=select_neighbors(tree,xyz,core);neighbors[core]=selected.astype(np.int32)
            local_neighbors=local[selected.clip(0)];observed=(selected>=0)&(local_neighbors>=0)
            for value,dest in [(surface,surface_distance),(material,material_distance)]:
                value=value.cpu().numpy();distance=np.linalg.norm(value[:count,None,:]-value[local_neighbors.clip(0)],axis=-1)
                distance[~observed]=np.nan;dest[core]=distance
            seen[core]+=1;core_block[core]=bi
            if progress:progress(bi+1,len(parts),len(core),int(observed.sum()),int(observed.size))
    assert np.all(seen==1) and np.all(core_block>=0) and np.isfinite(normal).all()
    component,count=affinity_components(neighbors,material_distance)
    return dict(neighbors=neighbors,material_distance=material_distance,surface_distance=surface_distance,
        normal=normal,boundary=boundary,material_component=component,core_coverage=seen,core_block=core_block,
        component_count=np.array(count,np.int64))
