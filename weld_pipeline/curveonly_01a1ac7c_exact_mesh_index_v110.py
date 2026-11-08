"""Exact-query indexing only: reuse V94 arithmetic on safely pruned triangles.

For triangle center c and vertex radius r, every triangle point lies in its
closed ball. Distance(q,triangle) >= max(0,|q-c|-r). A known feasible triangle
distance U therefore safely prunes centers outside U+r. Radius buckets keep
large planar triangles from enlarging every small-triangle search.
"""
import inspect,types,time
import numpy as np
from scipy.spatial import cKDTree
from . import curveonly_01a1ac7c_local_material_v94 as previous
from . import curveonly_01a1ac7c_finite_shadow_v73 as shadow
NAMES=previous.NAMES
DEFLECTION_MM=previous.DEFLECTION_MM
PRUNE_GUARD_MM=1e-7


def index(mesh,projected):
    tri=np.asarray(mesh,float);dimension=2 if projected else 3
    centers=tri[:,:,:dimension].mean(1)
    radius=np.linalg.norm(tri[:,:,:dimension]-centers[:,None],axis=2).max(1)
    bucket=np.floor(np.log2(np.maximum(radius,1e-12))).astype(np.int64);groups=[]
    for label in np.unique(bucket):
        ids=np.flatnonzero(bucket==label);groups.append(dict(ids=ids,radius=float(radius[ids].max()),tree=cKDTree(centers[ids])))
    return dict(mesh=mesh,centers=centers,tree=cKDTree(centers),groups=groups,dimension=dimension)


def closest(points,mesh,projected=False,cache=None):
    points=np.asarray(points,float);mesh=np.asarray(mesh,float);cache={}if cache is None else cache
    key=(id(mesh),bool(projected));entry=cache.get(key)
    if entry is None or entry['mesh']is not mesh:entry=index(mesh,projected);cache[key]=entry
    n=len(points);dimension=entry['dimension'];_,seed=entry['tree'].query(points[:,:dimension],workers=1)
    seed=np.asarray(seed).reshape(-1);upper=np.empty(n)
    for ti in np.unique(seed):
        rows=np.flatnonzero(seed==ti);upper[rows]=previous.closest(points[rows],mesh[ti:ti+1],projected)['distance_mm']
    selected=[{int(ti)}for ti in seed]
    for group in entry['groups']:
        near=group['tree'].query_ball_point(points[:,:dimension],upper+group['radius']+PRUNE_GUARD_MM,workers=1)
        for values,local_ids in zip(selected,near):values.update(map(int,group['ids'][local_ids]))
    # Keep original global triangle order, including all numerical ties.
    batches={}
    for qi,ids in enumerate(selected):batches.setdefault(tuple(sorted(ids)),[]).append(qi)
    result=dict(distance_mm=np.empty(n),closest_point=np.empty((n,3)),minimum_height_mm=np.empty(n),maximum_height_mm=np.empty(n))
    for ids,rows in batches.items():
        rows=np.asarray(rows,int);part=previous.closest(points[rows],mesh[np.asarray(ids,int)],projected)
        for name in result:result[name][rows]=part[name]
    assert np.isfinite(result['distance_mm']).all()
    return result


def owner_function(packet):
    lengths=[float(previous.cad.arclength(c['points'])[-1])for c in packet['candidates']]
    closed=[np.linalg.norm(np.asarray(c['points'])[0]-c['points'][-1])<=1e-6 for c in packet['candidates']]
    def owner_at(candidate_index,s,actual_packet,stack=()):
        assert actual_packet is packet
        if candidate_index in stack:raise ValueError('Cyclic input source provenance')
        candidate=packet['candidates'][candidate_index];path=candidate.get('source_material_path')
        if not path:return int(candidate.get('material',-1))
        if closed[candidate_index]:s=s%lengths[candidate_index]
        for member in path:
            if member['start_mm']-1e-8<=s<=member['end_mm']+1e-8:
                local=s-member['start_mm']
                if member['reversed']:local=member['end_mm']-member['start_mm']-local
                return owner_at(int(member['curve']),local,packet,stack+(candidate_index,))
        return -1
    return owner_at


def build(packet,solids):
    source=inspect.getsource(previous.build)
    old="if np.any(owners<0)or np.any(owners>=len(solids)):raise ValueError('Qualified source has unresolved owner')"
    assert source.count(old)==1;source=source.replace(old,'valid_owner=(owners>=0)&(owners<len(solids))')
    old='(owners!=mi)&(lower[:,mi]<=second+1e-7)';assert source.count(old)==1
    source=source.replace(old,'valid_owner&(owners!=mi)&(lower[:,mi]<=second+1e-7)')
    cache={};proxy=types.ModuleType('curveonly_cached_owner');proxy.__dict__.update(shadow.__dict__)
    proxy.owner_at=owner_function(packet)
    namespace=dict(previous.__dict__,shadow=proxy,closest=lambda q,m,projected=False:closest(q,m,projected,cache))
    exec(compile(source,__file__,'exec'),namespace);values,receipt=namespace['build'](packet,solids)
    receipt=dict(receipt,unknown_owner_policy='Keep source geometry and raw observations; unresolved owner has material availability0.',
        unknown_material_rows=int((values[:,0]==0).sum()),known_owner_computation_unchanged_from_v94=True,
        no_unknown_to_nonweld_conversion=True,exact_triangle_arithmetic='V94 unchanged on bound-certified triangle subset in original order',
        query_index='Radius-bucket center trees, guard1e-7mm for pruning only, no geometry/scoring tolerance change',
        immutable_input_length_and_closure_cache=True)
    return values,receipt
