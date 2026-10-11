"""Add direction-invariant distances to each observed face's projected extent."""
import argparse,gc,time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
SCHEMA='LOCAL_GLOBAL_FINITE_FACE_SEQUENCE_56_STEP5_V1'


def extent_distances(q,lo,hi):
    d=np.sort(np.column_stack((q-lo,hi-q)),axis=1)
    return (np.sign(d)*np.log1p(np.abs(d))).astype(np.float32)


def prepare(key,dest,base_run='run388'):
    start=time.perf_counter();source=OUT/f'{base_run}/source_{key}';r=read(source/'receipt.json')
    assert digest(source/'features.npy')==r['features_sha256'] and digest(source/'sequence_map.npz')==r['map_sha256']
    with np.load(source/'sequence_map.npz') as z:m={k:z[k] for k in z.files}
    case=load_case(key);case['x']._mmap.close();g=case['graph']
    cohort=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/cohort['input_file'])==r['input_sha256'];xyz=np.load(OUT/cohort['input_file'],mmap_mode='r')
    base=np.load(source/'features.npy',mmap_mode='r');folder=dest/f'source_{key}';folder.mkdir(exist_ok=False)
    x=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float32,shape=(len(base),56));x[:,:52]=base;x[:,52:]=0
    checks=[]
    with np.load(OUT/f'run185/source_{key}/surface_support.npz') as support:
        for j,pid in enumerate(m['profile_ids']):
            curve=g.curve(int(pid));a,b=m['offsets'][j:j+2];q=m['q_mm'][a:b]
            for side,fid in enumerate(g.profiles[int(pid)].get('face_indices',[])[:2]):
                sk=case['surfaces'][fid].get('point_ids_key')
                if sk is None or sk not in support or not len(support[sk]):continue
                p=xyz[support[sk]]
                # These are projected observed extents, not certified boundaries.
                projected=(p-curve.ends[0])@curve.direction if curve.kind=='line' else curve.project(p)[0]
                lo,hi=float(projected.min()),float(projected.max());value=extent_distances(q,lo,hi)
                x[a:b,52+2*side:54+2*side]=value
                if len(checks)<4:
                    reverse=extent_distances(curve.length-q,curve.length-hi,curve.length-lo)
                    error=float(np.max(np.abs(value-reverse)));assert error<2e-6;checks.append(error)
            if j%256==0:print('FINITE_CONTEXT',key,j+1,'/',len(m['profile_ids']),flush=True)
    x.flush();x._mmap.close();base._mmap.close();xyz._mmap.close();np.savez(folder/'sequence_map.npz',**m)
    write(folder/'receipt.json',dict(r,schema=SCHEMA,columns=56,base_run=base_run,features_sha256=digest(folder/'features.npy'),
        map_sha256=digest(folder/'sequence_map.npz'),extent_reversal_checks=checks,face_extent_is_observed_not_physical_truth=True,
        seconds=time.perf_counter()-start))
    print('FINITE_CONTEXT_READY',key,round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sources',nargs='+',required=True);a=ap.parse_args()
    dest=OUT/a.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(schema=SCHEMA,source_sha256=digest(__file__),sources=a.sources,
        reason='TRAIN conflicts differ in finite observed face extent',GT_reads=0,STEP_reads=0,fit_calls=0))
    for key in a.sources:prepare(key,dest);gc.collect()

if __name__=='__main__':main()
