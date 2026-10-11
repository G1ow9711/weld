"""Augment local sequences with input-only 3D neighborhood and finite context."""
import argparse,gc,time
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_relative_geometry_1135 import transformations
SCHEMA='LOCAL_GLOBAL_JOINT_SEQUENCE_52_STEP5_V1'


def neighborhood(tree,xyz,p,t,q,length):
    distance,ix=tree.query(p,k=32,workers=2);relative=xyz[ix]-p[:,None,:]
    centered=relative-relative.mean(1)[:,None,:]
    covariance=np.einsum('nki,nkj->nij',centered,centered)/32
    eigen=np.sqrt(np.maximum(np.linalg.eigvalsh(covariance),0))
    along=np.einsum('nki,ni->nk',relative,t)
    radius=np.linalg.norm(relative-along[:,:,None]*t[:,None,:],axis=2)
    stats=np.column_stack((np.abs(along.mean(1)),along.std(1),np.max(np.abs(along),axis=1),
        radius.mean(1),radius.std(1),radius.max(1)))
    far=p[:,None,:]+t[:,None,:]*np.array([-500.,-150.,-50.,50.,150.,500.])[None,:,None]
    far_d=tree.query(far.reshape(-1,3),workers=2)[0].reshape(-1,6)
    ordered=np.column_stack([np.sort(far_d[:,[i,5-i]],axis=1) for i in (2,1,0)])
    bounds=np.sort(np.column_stack((q,length-q)),axis=1)
    return np.column_stack((np.log1p(distance[:,[0,1,3,7,15,31]]),np.log1p(eigen),np.log1p(stats),
        eigen[:,0]/np.maximum(eigen[:,-1],1e-12),np.log1p(ordered),np.log1p(bounds))).astype(np.float32)


def controls(dest):
    rng=np.random.default_rng(1031);xyz=rng.normal(size=(1000,3))*30
    p=rng.normal(size=(10,3));t=np.tile([1.,0,0],(10,1));q=np.arange(10.)+1.;L=np.full(10,30.)
    tree=cKDTree(xyz);ref=neighborhood(tree,xyz,p,t,q,L)
    assert np.array_equal(ref,neighborhood(tree,xyz,p,-t,L-q,L))
    errors=[]
    for _,R,shift in transformations():
        moved=xyz@R.T+shift;value=neighborhood(cKDTree(moved),moved,p@R.T+shift,t@R.T,q,L)
        error=float(np.max(np.abs(value-ref)));assert error<2e-6;errors.append(error)
    write(dest/'neighborhood_controls.json',dict(rigid_max_error=max(errors),tangent_reverse_exact=True,
        world_coordinates_and_source_ids_not_features=True))


def prepare(key,dest,base_run='run386'):
    start=time.perf_counter();old=OUT/f'{base_run}/source_{key}';r=read(old/'receipt.json')
    assert digest(old/'features.npy')==r['features_sha256'] and digest(old/'sequence_map.npz')==r['map_sha256']
    with np.load(old/'sequence_map.npz') as z:m={k:z[k] for k in z.files}
    case=load_case(key);case['x']._mmap.close();g=case['graph']
    cohort=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/cohort['input_file'])==r['input_sha256']
    xyz=np.load(OUT/cohort['input_file'],mmap_mode='r');tree=cKDTree(xyz)
    folder=dest/f'source_{key}';folder.mkdir(exist_ok=False)
    source=np.load(old/'features.npy',mmap_mode='r');x=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float32,shape=(len(source),52));x[:,:28]=source
    for j,pid in enumerate(m['profile_ids']):
        a,b=m['offsets'][j:j+2];curve=g.curve(int(pid));q=m['q_mm'][a:b]
        for first in range(0,len(q),4096):
            qq=q[first:first+4096];x[a+first:a+first+len(qq),28:]=neighborhood(tree,xyz,curve.at(qq),curve.tangent(qq),qq,np.full(len(qq),curve.length))
        if j%128==0:print('GLOBAL_NEIGHBORHOOD',key,j+1,'/',len(m['profile_ids']),flush=True)
    x.flush();x._mmap.close();source._mmap.close();xyz._mmap.close();np.savez(folder/'sequence_map.npz',**m)
    write(folder/'receipt.json',dict(r,schema=SCHEMA,columns=52,features_sha256=digest(folder/'features.npy'),
        map_sha256=digest(folder/'sequence_map.npz'),base_run=base_run,global_raw_point_neighborhood=True,seconds=time.perf_counter()-start))
    print('GLOBAL_NEIGHBORHOOD_READY',key,round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sources',nargs='+',required=True);a=ap.parse_args()
    dest=OUT/a.run;dest.mkdir(exist_ok=False);controls(dest)
    write(dest/'registered_protocol.json',dict(schema=SCHEMA,source_sha256=digest(__file__),sources=a.sources,
        reason='Resolve TRAIN-only local descriptor ambiguity without changing labels',GT_reads=0,STEP_reads=0,fit_calls=0))
    for key in a.sources:prepare(key,dest);gc.collect()

if __name__=='__main__':main()
