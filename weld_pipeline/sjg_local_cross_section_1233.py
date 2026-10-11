"""Pure-point local transverse support evidence; no material truth inferred.

Each face is queried at the joint and +/-5,15,40 mm inside its tangent plane.
Signed transverse coordinates are pooled as unordered pairs. Site quantiles
make the summary independent of carrier traversal order. Missing face support
has explicit masks. These 93 columns are NOT legacy selector U160.
"""
import argparse,gc,time,traceback
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_relative_geometry_1135 import transformations

RADII=np.array([5.,15.,40.])
QUANTILES=np.array([0.,.25,.5,.75,1.])
SCHEMA='LOCAL_TWO_FACE_TRANSVERSE_SUPPORT_93_V1'


def face_sites(tree,positions,tangents,normal,spacing):
    """Distance-difference geometry removes the shared NN sampling floor."""
    transverse=np.cross(tangents,np.asarray(normal,float))
    norms=np.linalg.norm(transverse,axis=-1)
    if np.any(norms<1e-10):raise ValueError('Face normal parallel to local joint tangent')
    transverse/=norms[...,None]
    offsets=np.r_[0.,np.column_stack((-RADII,RADII)).ravel()]
    query=positions[...,None,:]+transverse[...,None,:]*offsets[:,None]
    distances=tree.query(query.reshape(-1,3),workers=2)[0].reshape(*positions.shape[:-1],7)
    result=np.empty_like(distances)
    result[...,0]=np.log1p(distances[...,0]/max(1.,float(spacing)))
    # Large positive means the displaced query moved away from observed face.
    # Negative values are retained: the center may already lie outside support.
    delta=(distances[...,1:]**2-distances[...,:1]**2)/np.repeat(RADII**2,2)
    result[...,1:]=np.sort(np.clip(delta.reshape(*delta.shape[:-1],3,2),-4.,4.),axis=-1).reshape(delta.shape)
    return result


def summarize(sites,known,normal_dot):
    # sites: profiles x 16 sites x 2 faces x 7 channels
    n=len(sites);result=np.zeros((n,2,93),np.float32)
    per_face=np.quantile(sites,QUANTILES,axis=1).transpose(1,2,0,3).reshape(n,2,35)
    center=np.exp(-sites[:,:,:,0]);joint=np.min(center,axis=2)
    contrast=sites[:,:,:,2::2]-sites[:,:,:,1::2]
    # Both-face edge evidence remains continuous, not a hard weldability rule.
    pair=np.concatenate((joint[:,:,None],np.minimum(contrast[:,:,0],contrast[:,:,1])),axis=-1)
    pair_summary=np.quantile(pair,QUANTILES,axis=1).transpose(1,0,2).reshape(n,20)
    for side in (0,1):
        slots=[0,1] if side==0 else [1,0]
        result[:,side,:70]=per_face[:,slots].reshape(n,70)
        result[:,side,70:90]=pair_summary*np.all(known,axis=1)[:,None]
        result[:,side,90:92]=known[:,slots]
        result[:,side,92]=np.abs(normal_dot)
    return result.reshape(n*2,93)


def controls(dest):
    grid=np.arange(-60.,61.,1.)
    xx,yy=np.meshgrid(grid,grid)
    plane=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
    positions=np.array([[[-20.,0.,0.],[0.,0.,0.],[20.,0.,0.]]])
    tangent=np.broadcast_to([1.,0.,0.],positions.shape);normal=np.array([0.,0.,1.])
    tree=cKDTree(plane);ref=face_sites(tree,positions,tangent,normal,2.)
    reverse=face_sites(tree,positions[:,::-1],-tangent[:,::-1],normal,2.)[:,::-1]
    flipped=face_sites(tree,positions,tangent,-normal,2.)
    assert np.array_equal(ref,reverse) and np.array_equal(ref,flipped)
    errors=[]
    for name,R,shift in transformations():
        got=face_sites(cKDTree(plane@R.T+shift),positions@R.T+shift,tangent@R.T,normal@R.T,2.)
        error=float(np.max(np.abs(got-ref)));assert error<1e-10;errors.append(error)
    edge=plane[plane[:,1]>=0];edge_value=face_sites(cKDTree(edge),positions,tangent,normal,2.)
    assert np.max(np.abs(ref[...,1:]))<1e-12
    assert np.allclose(edge_value[...,1::2],0) and np.allclose(edge_value[...,2::2],1)
    write(dest/'cross_section_controls.json',dict(status='TRANSVERSE_GEOMETRY_CONTROLS_PASSED',
        rigid_max_error=max(errors),line_reversal_exact=True,normal_sign_flip_exact=True,
        synthetic_plane_interior_and_edge_distinguished=True,no_material_role_truth_claim=True))


def prepare(key,dest,whole_run='run380'):
    start=time.perf_counter();check_resources();case=load_case(key);g=case['graph'];n=len(g.profiles)
    folder=dest/f'source_{key}';folder.mkdir(exist_ok=False)
    query=np.empty((n,16,3));tangents=np.empty_like(query);normals=np.zeros((n,2,3));groups=defaultdict(list)
    for pid,p in enumerate(g.profiles):
        curve=g.curve(pid);q=(np.arange(16)+.5)/16*curve.length
        query[pid]=curve.at(q);tangents[pid]=curve.tangent(q)
        for side,f in enumerate(p.get('face_indices',[])[:2]):
            v=np.asarray(case['surfaces'][f].get('normal',[0,0,0]),float);norm=np.linalg.norm(v)
            if norm>1e-10:normals[pid,side]=v/norm;groups[int(f)].append((pid,side))
        if pid%4096==0:g._curves.clear()
    case['x']._mmap.close();g._curves.clear()
    cohort=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/cohort['input_file'])==cohort['input_sha256']
    xyz=np.load(OUT/cohort['input_file'],mmap_mode='r')
    sites=np.zeros((n,16,2,7),np.float32);known=np.zeros((n,2),bool);parity=[]
    spacing={r['face']:r['spacing_median_mm'] for r in read(OUT/f'{whole_run}/source_{key}/receipt.json')['face_statistics']}
    with np.load(OUT/f'run185/source_{key}/surface_support.npz') as support:
        for fi,(face_id,entries) in enumerate(groups.items()):
            sk=case['surfaces'][face_id].get('point_ids_key')
            if sk is None or sk not in support or not len(support[sk]):continue
            points=np.asarray(xyz[support[sk]],float);tree=cKDTree(points)
            valid=[]
            for pid,side in entries:
                if np.min(np.linalg.norm(np.cross(tangents[pid],normals[pid,side]),axis=-1))>=1e-10:valid.append((pid,side))
            for first in range(0,len(valid),256):
                ix=np.array(valid[first:first+256],int);ids,sides=ix.T
                values=face_sites(tree,query[ids],tangents[ids],normals[ids,sides,None,:],spacing.get(face_id,1.))
                sites[ids,:,sides]=values;known[ids,sides]=True
            if valid and len(parity)<4:
                pid,side=valid[0];a=face_sites(tree,query[pid:pid+1],tangents[pid:pid+1],normals[pid,side],spacing.get(face_id,1.))
                b=face_sites(tree,query[pid:pid+1,::-1],-tangents[pid:pid+1,::-1],-normals[pid,side],spacing.get(face_id,1.))[:,::-1]
                error=float(np.max(np.abs(a-b)));assert error<1e-12;parity.append(dict(profile=pid,face=face_id,error=error))
            del tree,points
            if fi%64==0:print('LOCAL_SECTION',key,'FACES',fi+1,'/',len(groups),flush=True);check_resources()
    features=summarize(sites,known,np.sum(normals[:,0]*normals[:,1],axis=-1))
    assert np.isfinite(features).all()
    np.save(folder/'local_features.npy',features)
    write(folder/'receipt.json',dict(status='LOCAL_CROSS_SECTION_INPUT_READY',schema=SCHEMA,source=key,
        rows=len(features),columns=93,input_sha256=cohort['input_sha256'],geometry_sha256=case['geometry_sha256'],
        feature_sha256=digest(folder/'local_features.npy'),both_observed_profiles=int(known.all(1).sum()),
        radii_mm=RADII.tolist(),sites_per_carrier=16,GT_reads=0,STEP_reads=0,fit_calls=0,
        runtime_surface_roles='UNKNOWN_NOT_INFERRED',raw_point_reversal_checks=parity,seconds=time.perf_counter()-start))
    xyz._mmap.close();print('LOCAL_SECTION_READY',key,'SECONDS',round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sources',nargs='+',required=True);a=ap.parse_args()
    dest=OUT/a.run;dest.mkdir(exist_ok=True);controls(dest)
    write(dest/'local_feature_protocol.json',dict(schema=SCHEMA,sources=a.sources,source_sha256=digest(__file__),
        new_features='Local face extension at fixed physical scales; no whole-face centroid, bounds or count',
        TRAIN=['21','BM79A_native_local','10'],DEV=['64'],external_consultation_paused=True))
    try:
        for key in a.sources:prepare(key,dest);gc.collect()
    except BaseException:write(dest/'local_feature_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
