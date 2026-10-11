"""Pure XYZ multiscale quadratic-fit residuals inspired by the supplied MLS demo.

Original points are never displaced or removed. Invalid fits remain masked.
This bounded-neighbour implementation is not a PCL byte-for-byte reproduction.
"""
import argparse
import gc
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources,resources,reclaim_own_idle_pages
from weld_pipeline.sjg_hierarchical_features_1029 import positions
from weld_pipeline.sjg_parent_ordered_bins_1074 import profile_bins

RADII=(10.,30.)
MAX_NEIGHBORS=64
CHANNELS=('plane_residual_over_radius','quadratic_residual_over_radius',
          'plane_rms_over_radius','quadratic_rms_over_radius','planarity',
          'surface_variation','neighbor_fraction','plane_valid','quadratic_valid')


def quadratic_features(centers,neighbors,radii=RADII):
    centers=np.asarray(centers,np.float64);neighbors=np.asarray(neighbors,np.float64)
    relative=neighbors-centers[:,None]
    squared=np.square(relative).sum(-1)
    # Account only for subtraction/rotation roundoff at an exact radius boundary.
    # This is not a geometric merge tolerance and never changes curve topology.
    roundoff=64*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(centers))),float(np.max(np.abs(neighbors))))
    output=[]
    for radius in radii:
        inside=squared<=(radius+roundoff)*(radius+roundoff)
        weights=np.exp(-squared/(radius*radius))*inside
        weight_sum=weights.sum(1)
        w=weights/np.maximum(weight_sum[:,None],np.finfo(float).tiny)
        centroid=np.einsum('bk,bkj->bj',w,relative)
        centered=(relative-centroid[:,None])/radius
        covariance=np.einsum('bk,bki,bkj->bij',w,centered,centered)
        eigen,basis=np.linalg.eigh(covariance)
        eigen=np.maximum(eigen,0)
        plane_valid=(inside.sum(1)>=6)&(eigen[:,2]>1e-14)&(eigen[:,1]>1e-8*eigen[:,2])
        # Columns: two tangent axes, then the least-variance normal axis.
        frame=basis[:,:,[2,1,0]]
        local=np.einsum('bki,bij->bkj',centered,frame)
        at_center=np.einsum('bi,bij->bj',-centroid/radius,frame)
        u,v,z=local[:,:,0],local[:,:,1],local[:,:,2]
        design=np.stack((np.ones_like(u),u,v,u*u,u*v,v*v),axis=-1)
        matrix=np.einsum('bk,bki,bkj->bij',w,design,design)
        rhs=np.einsum('bk,bki,bk->bi',w,design,z)
        singular=np.linalg.eigvalsh(matrix)
        quad_valid=plane_valid&(singular[:,0]>1e-10*np.maximum(singular[:,-1],1e-30))
        coefficients=np.zeros((len(centers),6),np.float64)
        if np.any(quad_valid):coefficients[quad_valid]=np.linalg.solve(matrix[quad_valid],rhs[quad_valid])
        a,b=at_center[:,0],at_center[:,1]
        center_design=np.stack((np.ones_like(a),a,b,a*a,a*b,b*b),axis=-1)
        fitted=np.einsum('bi,bi->b',center_design,coefficients)
        observed_fit=np.einsum('bki,bi->bk',design,coefficients)
        features=np.c_[np.abs(at_center[:,2])*plane_valid,
            np.abs(at_center[:,2]-fitted)*quad_valid,
            np.sqrt(eigen[:,0])*plane_valid,
            np.sqrt(np.einsum('bk,bk->b',w,np.square(z-observed_fit)))*quad_valid,
            (eigen[:,1]-eigen[:,0])/np.maximum(eigen[:,2],1e-30),
            eigen[:,0]/np.maximum(eigen.sum(1),1e-30),
            inside.mean(1),plane_valid.astype(float),quad_valid.astype(float)]
        assert np.isfinite(features).all()
        output.append(features)
    return np.concatenate(output,axis=1).astype(np.float32)


def controls():
    u,v=np.meshgrid(np.arange(-3.,4.),np.arange(-3.,4.))
    flat=np.c_[u.ravel(),v.ravel(),np.zeros(u.size)]
    curved=flat.copy();curved[:,2]=0.05*(curved[:,0]**2+curved[:,1]**2)
    fold=flat.copy();fold[:,2]=np.abs(fold[:,0])
    sparse=np.c_[np.arange(-24.,25.),np.zeros(49),np.zeros(49)]
    patches=np.stack((flat,curved,fold,sparse))
    result=quadratic_features(np.zeros((4,3)),patches)
    assert np.all(result[0,[0,1,2,3,9,10,11,12]]<1e-10)
    assert result[0,8]==1 and result[1,8]==1
    assert result[1,1]<result[1,0] and result[1,1]<1e-5
    assert result[2,1]>1e-4
    assert result[3,7]==0 and result[3,8]==0
    angle=.73;rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    shift=np.array([1200.,-2500.,780.])
    moved=patches@rotation.T+shift
    moved_result=quadratic_features(np.tile(shift,(4,1)),moved)
    error=float(np.max(np.abs(result-moved_result)))
    assert np.allclose(result,moved_result,atol=2e-6,rtol=2e-6),error
    return dict(status='PURE_XYZ_QUADRATIC_GEOMETRY_CONTROLS_PASSED',
        flat_plane_residual_mm=float(result[0,1]*10),
        curved_plane_residual_mm=float(result[1,0]*10),
        curved_quadratic_residual_mm=float(result[1,1]*10),
        crease_quadratic_residual_mm=float(result[2,1]*10),
        collinear_neighborhood_marked_unknown=True,rigid_motion_feature_max_error=error,
        results=result.tolist(),radii_mm=list(RADII),channels=list(CHANNELS),
        output_coordinates_changed=False,optimizer_updates=0)


def prepare(key):
    dest=OUT/'run259'/('source_'+key);dest.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();check_resources();write(dest/'controls.json',controls())
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/case['input_file'])==case['input_sha256']
    xyz=np.load(OUT/case['input_file'],mmap_mode='r')
    db=next(c for c in read(OUT/'run201/all_domains_frozen_before_labels.json')['cases'] if c['source_key']==key)
    assert digest(OUT/'run201'/db['json_file'])==db['json_sha256']
    assert digest(OUT/'run201'/db['mapping_file'])==db['mapping_sha256']
    domain=read(OUT/'run201'/db['json_file'])
    with np.load(OUT/'run201'/db['mapping_file']) as z:pf=z['profile_family'];transform=z['local_to_canonical']
    if key=='21':
        binding=read(OUT/'run224/input_freeze.json')
        assert digest(OUT/'run224/input_domain.npz')==binding['files']['input_domain.npz']
        with np.load(OUT/'run224/input_domain.npz') as z:
            queries=z['query_xyz'];query_ids=z['query_ids'];cells=z['cells'];families=z['family']
        offsets=np.searchsorted(families,np.arange(len(domain['families'])+1))
    else:
        folder=OUT/'run252'/('source_'+key);r=read(folder/'receipt.json')
        for name in ('cells.f64','family_offsets.npy'):assert digest(folder/name)==r['files'][name]
        cells=np.memmap(folder/'cells.f64',mode='r',dtype='<f8',shape=(r['cells'],4))
        offsets=np.load(folder/'family_offsets.npy')
        if key=='10':
            b=read(OUT/'run255/receipt.json')
            assert digest(OUT/'run255/all_query_xyz.npy')==b['files']['all_query_xyz.npy']
            assert digest(OUT/'run255/cell_query_ids.npy')==b['files']['cell_query_ids.npy']
            queries=np.load(OUT/'run255/all_query_xyz.npy',mmap_mode='r')
            query_ids=np.load(OUT/'run255/cell_query_ids.npy',mmap_mode='r')
        else:
            assert digest(folder/'query_xyz.npy')==r['files']['query_xyz.npy']
            queries=np.load(folder/'query_xyz.npy',mmap_mode='r')
            lookup={tuple(float(v) for v in q):i for i,q in enumerate(queries)}
            query_ids=np.empty((len(cells),3),np.int64)
            for fid,family in enumerate(domain['families']):
                a,b=offsets[fid:fid+2];part=cells[a:b]
                q=np.c_[part[:,0],(part[:,0]+part[:,1])/2,part[:,1]].ravel()
                coords=positions(family,q);coords[coords==0]=0
                query_ids[a:b]=np.asarray([lookup[tuple(float(v) for v in p)] for p in coords]).reshape(-1,3)
            del lookup
    tree=cKDTree(xyz)
    nearest=np.empty(len(queries),np.int64)
    for first in range(0,len(queries),4096):
        _,nearest[first:first+4096]=tree.query(queries[first:first+4096],k=1,workers=2)
    original_ids,inverse=np.unique(nearest,return_inverse=True)
    geometry=np.empty((len(original_ids),len(RADII)*len(CHANNELS)),np.float32)
    for first in range(0,len(original_ids),256):
        ids=original_ids[first:first+256];center=np.asarray(xyz[ids],np.float64)
        _,neighbor=tree.query(center,k=min(MAX_NEIGHBORS,len(xyz)),workers=2)
        geometry[first:first+len(ids)]=quadratic_features(center,xyz[neighbor])
        if first%8192==0:
            if resources()['available_physical_bytes']<1879048192:reclaim_own_idle_pages()
            check_resources();print('MLS_GEOMETRY_FITS',key,first+len(ids),'/',len(original_ids),flush=True)
    np.savez(dest/'original_point_geometry.npz',original_point_ids=original_ids,features=geometry,
             query_to_original_feature=inverse,query_nearest_original_id=nearest)
    del tree,xyz,nearest;gc.collect();reclaim_own_idle_pages();check_resources()
    output=np.lib.format.open_memmap(dest/'geometry_ordered_bins.npy',mode='w+',dtype=np.float32,
                                    shape=(len(pf),16,len(RADII)*len(CHANNELS)*3))
    covered=np.zeros(len(pf),bool)
    for fid,family in enumerate(domain['families']):
        a,b=offsets[fid:fid+2];values=geometry[inverse[query_ids[a:b]]].reshape(b-a,-1)
        ids=np.asarray(family['profiles'],np.int64)
        assert np.all(pf[ids]==fid) and not covered[ids].any()
        output[ids]=profile_bins(cells[a:b],values,transform[ids],family['circumference_mm'])
        covered[ids]=True
        if fid%4096==0:
            if resources()['available_physical_bytes']<1879048192:output.flush();reclaim_own_idle_pages()
            check_resources();print('MLS_PROFILE_BINS',key,int(covered.sum()),'/',len(pf),flush=True)
    assert covered.all();output.flush()
    assert digest(OUT/case['input_file'])==case['input_sha256']
    write(dest/'receipt.json',dict(status='ALL_PROFILE_MLS_GEOMETRY_READY',source=key,
        profiles=len(pf),families=len(domain['families']),query_positions=len(queries),
        original_points_fitted=len(original_ids),new_geometry_KD_queries=len(queries)+len(original_ids),
        geometry_stage_cost_additional_to_base_features=True,radii_mm=list(RADII),max_neighbors=MAX_NEIGHBORS,
        channels_per_radius=list(CHANNELS),shape=list(output.shape),base_bin_channels=188,
        geometry_bin_channels=output.shape[-1],combined_bin_channels=188+output.shape[-1],
        quadratic_valid_fraction_by_radius=[float(geometry[:,i*len(CHANNELS)+8].mean()) for i in range(len(RADII))],
        source_input_sha256=case['input_sha256'],input_hash_after_equal=True,
        bin_sha256=digest(dest/'geometry_ordered_bins.npy'),point_geometry_sha256=digest(dest/'original_point_geometry.npz'),
        no_point_removal=True,no_coordinate_displacement=True,no_candidate_filter=True,
        target_reads=0,STEP_reads=0,optimizer_updates=0,seconds=time.perf_counter()-start,utc=utc()))
    print('MLS_GEOMETRY_READY',key,len(pf),'BIN_CHANNELS',output.shape[-1],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source');a=ap.parse_args()
    if not a.source:
        print(controls());return
    try:prepare(a.source)
    except BaseException:
        dest=OUT/'run259'/('source_'+a.source)
        if dest.exists():write(dest/'failure.json',dict(error=traceback.format_exc(),optimizer_updates=0,utc=utc()))
        raise


if __name__=='__main__':main()
