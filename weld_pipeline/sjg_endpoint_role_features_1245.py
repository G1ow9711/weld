"""Input-only geometric features for both inward endpoint roles.

One shared classifier sees q, inward/outward context and geometric evidence.
Neither endpoint source provenance nor workpiece/face identity is a feature.
"""
import argparse
import gc
import time
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_dense_joint_sequence_1237 import ordered_features

SCHEMA='INWARD_ENDPOINT_GEOMETRY_180_V1'
COLS=180


def group_events(q,length):
    starts=np.r_[0,np.flatnonzero(np.diff(q)>1e-6)+1]
    ends=np.r_[starts[1:],len(q)]
    values=np.array([q[a:b].mean() for a,b in zip(starts,ends)])
    assert np.max(np.maximum(-values,values-length))<=1e-8
    return np.clip(values,0,length),starts,ends


def interpolate(grid,x,q):
    return np.column_stack([np.interp(q,grid,x[:,j]) for j in range(x.shape[1])])


def role_features(q,grid,x,p,length,common,inward,closed):
    queries=[q,q+inward*50.,q-inward*50.]
    queries=[np.mod(v,length) if closed else np.clip(v,0,length) for v in queries]
    blocks=[interpolate(grid,x,v) for v in queries]
    probs=np.column_stack([np.interp(v,grid,p) for v in queries])
    inside=length-q if inward>0 else q
    outside=q if inward>0 else length-q
    scalar=np.column_stack((np.full(len(q),np.log1p(length)),np.log1p(inside),np.log1p(outside),common))
    result=np.column_stack((*blocks,probs,scalar)).astype(np.float32)
    assert result.shape==(len(q),COLS) and np.isfinite(result).all()
    return result


def controls(dest):
    rng=np.random.default_rng(661);grid=np.linspace(0,1000,201);x=rng.normal(size=(201,56));p=rng.random(201)
    q=np.array([0.,20.,107.,566.,999.,1000.]);common=np.zeros((len(q),6))
    a=role_features(q,grid,x,p,1000.,common,1,False)
    b=role_features(1000-q,grid,x[::-1],p[::-1],1000.,common,-1,False)
    error=float(np.max(np.abs(a-b)));assert error<1e-6
    write(dest/'role_feature_controls.json',dict(line_parameter_reversal_role_swap_max_abs=error,
        feature_columns=COLS,endpoint_origin_kind_not_a_feature=True,raw_XYZ_not_a_feature=True))


def prepare(key,dest,event_run='run395',sequence_run='run390',occupancy_run='run392'):
    start=time.perf_counter();src=OUT/f'{event_run}/source_{key}';er=read(src/'receipt.json')
    assert er['GT_reads']==0 and digest(src/'boundary_events.npz')==er['events_sha256']
    with np.load(src/'boundary_events.npz') as z:e={k:z[k] for k in z.files}
    sequence=OUT/f'{sequence_run}/source_{key}';sr=read(sequence/'receipt.json')
    assert digest(sequence/'features.npy')==sr['features_sha256'] and digest(sequence/'sequence_map.npz')==sr['map_sha256']
    with np.load(sequence/'sequence_map.npz') as z:m={k:z[k] for k in z.files}
    local=np.load(sequence/'features.npy',mmap_mode='r')
    predictions=OUT/f'{occupancy_run}/proposal_{key}';pr=read(predictions/'prediction_receipt.json')
    assert pr['GT_reads']==0 and pr['model_sha256']==read(OUT/'run392/training_receipt.json')['model_sha256']
    with np.load(predictions/'occupancy.npz') as z:o={k:z[k] for k in z.files}
    assert np.array_equal(m['admitted_rows'],o['admitted_rows']) and np.array_equal(m['admitted_rows'],e['admitted_rows'])
    c=load_case(key);g=c['graph'];c['x']._mmap.close()
    co=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/co['input_file'])==co['input_sha256']
    points=np.load(OUT/co['input_file'],mmap_mode='r');tree=cKDTree(points)
    cache={};profile_lookup={int(p):j for j,p in enumerate(m['profile_ids'])}
    for j,pid in enumerate(e['profile_ids']):
        pid=int(pid);curve=g.curve(pid);a,b=e['offsets'][j:j+2]
        q,starts,ends=group_events(e['q_mm'][a:b],curve.length)
        common=np.zeros((len(q),6));common[:,0]=np.log1p(tree.query(curve.at(q),workers=2)[0])
        for k,(lo,hi) in enumerate(zip(starts+a,ends+a)):
            ids=np.flatnonzero(e['kinds'][lo:hi]==2)+lo
            if len(ids):
                common[k,1]=np.max(e['incidence'][ids]);excess=e['hull_signed_excess_mm'][ids]
                v=excess[np.argmin(np.abs(excess))];common[k,2]=np.sign(v)*np.log1p(abs(v))
                common[k,3]=1.;common[k,5]=np.log1p(len(np.unique(e['surface_ids'][ids])))
            common[k,4]=bool(np.any(e['finite_support'][lo:hi]))
        cache[pid]=(q,common,curve.length,curve.kind=='arc' and curve.arc.is_closed)
    rows=e['admitted_rows'];sizes=[len(cache[int(row)//2][0]) for row in rows]
    offsets=np.r_[0,np.cumsum(sizes)];folder=dest/f'source_{key}';folder.mkdir(exist_ok=False)
    features=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float32,shape=(int(offsets[-1])*2,COLS))
    qblocks=[]
    for i,row in enumerate(rows):
        row=int(row);pid=row//2;q,common,length,closed=cache[pid];j=profile_lookup[pid]
        a,b=m['offsets'][j:j+2];c0,d=o['offsets'][i:i+2]
        assert d-c0==b-a
        grid=m['q_mm'][a:b];xx=ordered_features(local[a:b],row%2);p=o['probabilities'][c0:d]
        begin=2*offsets[i];n=len(q)
        features[begin:begin+n]=role_features(q,grid,xx,p,length,common,1,closed)
        features[begin+n:begin+2*n]=role_features(q,grid,xx,p,length,common,-1,closed)
        qblocks.append(q)
    features.flush();features._mmap.close();local._mmap.close();points._mmap.close()
    np.savez(folder/'role_map.npz',rows=rows,offsets=offsets,q_mm=np.concatenate(qblocks))
    write(folder/'receipt.json',dict(status='PURE_INPUT_ENDPOINT_ROLE_FEATURES_FROZEN',source=key,schema=SCHEMA,
        rows=int(offsets[-1])*2,columns=COLS,ordered_carriers=len(rows),
        feature_sha256=digest(folder/'features.npy'),map_sha256=digest(folder/'role_map.npz'),
        events_sha256=er['events_sha256'],occupancy_sha256=digest(predictions/'occupancy.npz'),
        local_features_sha256=sr['features_sha256'],GT_reads=0,STEP_reads=0,fits=0,
        feature_source_kind_not_included=True,raw_XYZ_and_identity_not_included=True,seconds=time.perf_counter()-start))
    print('ENDPOINT_FEATURES_READY',key,'ROWS',int(offsets[-1])*2,'COLS',COLS,'SECONDS',round(time.perf_counter()-start,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sources',nargs='+',required=True)
    ap.add_argument('--event-run',default='run395')
    args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False);controls(dest)
    write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),sources=args.sources,event_run=args.event_run,GT_reads=0,STEP_reads=0,fits=0))
    for key in args.sources:prepare(key,dest,args.event_run);gc.collect()


if __name__=='__main__':main()
