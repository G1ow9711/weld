"""Same registered 142-channel semantics, batched rigid consistency operations."""
import argparse
import gc
import shutil
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_joint_object_features_1106 import side_rows
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
from weld_pipeline.sjg_relative_geometry_1135 import KEYS,transformations,support_boundaries,boundary_context,audit_labels,local_values


def local_batch(points,tangents,origins,t0,normals,radial,radial_valid):
    e1=t0/np.linalg.norm(t0,axis=1)[:,None];e2=np.zeros_like(e1);source=np.zeros(len(points),np.int8)
    for k,v in enumerate((normals[:,0],normals[:,1],radial),1):
        u=v-e1*np.einsum('ni,ni->n',v,e1)[:,None];n=np.linalg.norm(u,axis=1)
        good=(source==0)&(n>256*np.finfo(float).eps*np.maximum(1.,np.linalg.norm(v,axis=1)))
        if k==3:good &= radial_valid
        e2[good]=u[good]/n[good,None];source[good]=k
    E=np.stack([e1,e2,np.cross(e1,e2)],axis=2);E[source==0]=0
    p=np.einsum('nji,nik->njk',points-origins[:,None,:],E)/1000
    t=np.einsum('nji,nik->njk',tangents,E);n=np.einsum('nji,nik->njk',normals,E)
    scale=np.maximum(1.,np.maximum(np.abs(origins).max(1),np.abs(points).max((1,2))))
    p[np.abs(p)<=256*np.finfo(float).eps*scale[:,None,None]/1000]=0
    t[np.abs(t)<=256*np.finfo(float).eps]=0;n[np.abs(n)<=256*np.finfo(float).eps]=0
    values=np.concatenate([p.reshape(-1,9),t.reshape(-1,6),n.reshape(-1,6)],axis=1).astype(np.float32)
    return values,source


def prepare(key,dest,transform_observer=None):
    started=time.perf_counter();directory=dest/f'source_{key}';directory.mkdir()
    feature_run='run293' if key=='64' else 'run308' if key in ('4','25','36','42') else 'run287'
    alias_run='run293' if key=='64' else 'run288'
    geometry_run='run293' if key=='64' else 'run285'
    basepath=OUT/f'{feature_run}/source_{key}/object_features.npy'
    assert digest(basepath)==read(basepath.parent/'receipt.json')['features_sha256']
    aliaspath=OUT/f'{alias_run}/source_{key}_alias_map.npz'
    binding=next(v for v in read(OUT/alias_run/'all_alias_maps_frozen_before_labels.json')['cases'] if v['source']==key)
    assert digest(aliaspath)==binding['map_sha256']
    with np.load(aliaspath) as z:canonical=z['canonical_fragments']
    geometry=next(v for v in read(OUT/geometry_run/'all_geometry_frozen_before_labels.json')['cases'] if v['source']==key)
    profilepath=OUT/f'run185/source_{key}/profiles.json';surfacepath=profilepath.parent/'surfaces.json'
    assert digest(profilepath)==geometry['source_files']['profiles.json']
    assert digest(surfacepath)==geometry['source_files']['surfaces.json']
    assert digest(OUT/geometry_run/geometry['geometry_file'])==geometry['geometry_sha256']
    profiles=read_profile_geometry(profilepath);surfaces=read(surfacepath)
    with np.load(OUT/geometry_run/geometry['geometry_file']) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
    n=len(canonical);points=np.zeros((n,3,3));tangents=np.zeros((n,2,3));origins=np.zeros((n,3));t0=np.zeros((n,3))
    normals=np.zeros((n,2,3));radial=np.zeros((n,3));radial_valid=np.zeros(n,bool);contexts=np.zeros((n,8),np.float32)
    pids=np.searchsorted(graph.offsets,canonical,side='right')-1
    unique,starts,counts=np.unique(pids,return_index=True,return_counts=True)
    assert np.all(np.diff(pids)>=0)
    for pid,first,count in zip(unique,starts,counts):
        selected=slice(first,first+count);fr=canonical[selected];spans=graph.spans[fr]
        curve=graph.curve(pid);profile=profiles[int(pid)]
        all_spans=graph.spans[graph.offsets[pid]:graph.offsets[pid+1]]
        closed=curve.kind=='arc' and curve.arc.is_closed
        contexts[selected]=boundary_context(spans,support_boundaries(all_spans,curve.length,closed),curve.length,closed)
        q=np.column_stack([spans[:,0],spans[:,1],spans[:,:2].mean(1)])
        points[selected]=curve.at(q);tangents[selected]=curve.tangent(q[:,:2])
        origin=curve.at(np.array([0.]))[0];origins[selected]=origin;t0[selected]=curve.tangent(np.array([0.]))[0]
        if curve.kind=='arc':radial[selected]=(origin-curve.arc.center_mm)/curve.arc.radius_mm;radial_valid[selected]=True
        for side,face in enumerate(profile.get('face_indices',[])[:2]):
            raw=np.asarray(surfaces[int(face)].get('normal',[0,0,0]),np.float64);norm=np.linalg.norm(raw)
            if norm:normals[selected,side]=raw/norm
        graph._curves.clear()
        if int(pid)%4096==0:check_resources()
    del graph,profiles,surfaces;gc.collect()
    base=np.load(basepath,mmap_mode='r')
    output=np.lib.format.open_memmap(directory/'features.npy',mode='w+',dtype=np.float32,shape=(n*2,142));output[:]=0
    maximum={name:0. for name,_,_ in transformations()};valid=0;invalid=0
    for first in range(0,n,8192):
        last=min(first+8192,n);s=slice(first,last);fr=canonical[s];index=np.arange(first,last)
        for side in (0,1):
            rows=index*2+side;ordered=normals[s] if side==0 else normals[s,::-1]
            output[rows,:128]=base[fr] if side==0 else side_rows(base,fr,np.ones(len(fr),np.int8))
            values,source=local_batch(points[s],tangents[s],origins[s],t0[s],ordered,radial[s],radial_valid[s])
            # Compare one row per block to original scalar semantics, including invalid frames.
            scalar,_,ss=local_values(points[s][:1],tangents[s][:1],origins[first],t0[first],ordered[0],radial[first] if radial_valid[first] else None)
            assert int(source[0])==ss and np.allclose(values[0],scalar[0],rtol=8*np.finfo(np.float32).eps,atol=2e-12)
            output[rows,84:105]=values;output[rows,128:136]=contexts[s]
            output[rows,136]=source!=0;output[rows,137]=source;output[rows,138:142]=0
            valid+=int((source!=0).sum());invalid+=int((source==0).sum())
            for name,R,t in transformations():
                altered,ss=local_batch(points[s]@R.T+t,tangents[s]@R.T,origins[s]@R.T+t,t0[s]@R.T,
                                       ordered@R.T,radial[s]@R.T,radial_valid[s])
                if transform_observer is not None:transform_observer(name,rows,84,altered)
                assert np.array_equal(source,ss)
                delta=np.abs(values.astype(float)-altered.astype(float));maximum[name]=max(maximum[name],float(delta.max(initial=0)))
                ok=np.isclose(values,altered,rtol=8*np.finfo(np.float32).eps,atol=2e-12)
                if not ok.all():
                    bad=np.argwhere(~ok)[0];row,column=map(int,bad)
                    write(directory/'equivariance_failure.json',dict(object_row=int(rows[row]),transform=name,column=column+84,
                        reference=float(values[row,column]),transformed=float(altered[row,column]),max_difference=float(delta.max())))
                    raise ValueError('Relative representation consistency failed')
            check_resources()
    output.flush();assert np.isfinite(output).all();output._mmap.close();base._mmap.close()
    receipt=dict(source=key,rows=n*2,columns=142,features_sha256=digest(directory/'features.npy'),
        alias_sha256=binding['map_sha256'],geometry_sha256=geometry['geometry_sha256'],frame_valid_rows=valid,frame_invalid_rows=invalid,
        transforms_maximum_absolute_feature_difference=maximum,equivariance_passed=True,
        scalar_semantics_controls_passed=True,GT_reads=0,STEP_reads=0,seconds=time.perf_counter()-started)
    write(directory/'receipt.json',receipt);print('BATCH_RELATIVE_INPUT_READY',key,receipt['rows'],receipt['seconds'],flush=True)
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run327');ap.add_argument('--worker',action='store_true')
    args=ap.parse_args();dest=OUT/args.run
    if args.worker:
        started=time.perf_counter()
        try:
            cases=[]
            for key in KEYS:cases.append(prepare(key,dest));gc.collect()
            write(dest/'all_relative_inputs_frozen_before_labels.json',dict(cases=cases,GT_reads=0))
            control=audit_labels(dest)
            write(dest/'worker_receipt.json',dict(status=control['status'],cases=cases,classification_conflict_rows=control['classification_conflict_rows'],
                endpoint_feasibility=control['endpoint_feasibility'],fit_calls=0,seconds=time.perf_counter()-started,full_contract_pass=False))
        except BaseException:
            write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0,seconds=time.perf_counter()-started));raise
        return
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir()
    for file in (__file__,ROOT/'weld_pipeline/sjg_relative_geometry_1135.py'):shutil.copy2(file,snapshot)
    old=read(OUT/'run326/registered_protocol.json')
    write(dest/'registered_protocol.json',dict(old,implementation='batch matrix operations; same fields, scope, numeric tolerance',
        prior_attempt='run326',prior_failure_preserved=True,utc=utc()))
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_relative_geometry_batch_1136','--run',args.run,'--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='RELATIVE_CONTROL_INCOMPLETE',watchdog=result));raise RuntimeError('Relative control incomplete')
    receipt=read(dest/'worker_receipt.json');receipt['external_wall_seconds']=result['wall_seconds']
    write(dest/'receipt.json',receipt);print('RELATIVE_GATE_RESULT',receipt,flush=True)


if __name__=='__main__':main()
