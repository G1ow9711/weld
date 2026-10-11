"""TRAIN-derived extension: retain known single-face distances and relative support context.

152 channels = Pro324 relative142 + two ordered (centroid XYZ, radius, valid) tuples.
No ID, absolute coordinate, new sample, threshold selection, or label change.
"""
import argparse
import gc
import shutil
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_stream_profile_geometry_1149 import read_profile_geometry
from weld_pipeline.sjg_relative_geometry_1135 import KEYS,frame,transformations,audit_labels


def prepare(key,dest,prior_run='run327',transform_observer=None):
    started=time.perf_counter();directory=dest/f'source_{key}';directory.mkdir()
    geometry_run='run293' if key=='64' else 'run285'
    alias_run='run293' if key=='64' else 'run288'
    prior=OUT/f'{prior_run}/source_{key}';pr=read(prior/'receipt.json')
    assert digest(prior/'features.npy')==pr['features_sha256']
    alias=OUT/f'{alias_run}/source_{key}_alias_map.npz';assert digest(alias)==pr['alias_sha256']
    with np.load(alias) as z:canonical=z['canonical_fragments']
    g=next(v for v in read(OUT/geometry_run/'all_geometry_frozen_before_labels.json')['cases'] if v['source']==key)
    folder=OUT/f'run185/source_{key}'
    for name in ('profiles.json','surfaces.json','surface_support.npz'):
        assert digest(folder/name)==g['source_files'][name]
    assert digest(OUT/geometry_run/g['geometry_file'])==g['geometry_sha256']
    cohort=next(v for v in read(OUT/'run98/local_cohort.json')['cases'] if v['source_key']==key)
    xyzfile=OUT/cohort['input_file'];assert digest(xyzfile)==cohort['input_sha256']
    profiles=read_profile_geometry(folder/'profiles.json');surfaces=read(folder/'surfaces.json')
    with np.load(OUT/geometry_run/g['geometry_file']) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
    n=len(canonical);queries=np.empty((n,16,3));origins=np.empty((n,3));frames=np.zeros((n,2,3,3));sources=np.zeros((n,2),np.int8)
    face_ids=np.full((n,2),-1,np.int32)
    pids=np.searchsorted(graph.offsets,canonical,side='right')-1
    unique,starts,counts=np.unique(pids,return_index=True,return_counts=True)
    for pid,first,count in zip(unique,starts,counts):
        selected=slice(first,first+count);spans=graph.spans[canonical[selected]]
        curve=graph.curve(pid);profile=profiles[int(pid)];faces=profile.get('face_indices',[])[:2]
        q=spans[:,0,None]+(np.arange(16)+.5)*(spans[:,1]-spans[:,0])[:,None]/16
        queries[selected]=curve.at(q);origin=curve.at(np.array([0.]))[0];origins[selected]=origin
        t0=curve.tangent(np.array([0.]))[0];normals=np.zeros((2,3))
        for side,face_id in enumerate(faces):
            face_ids[selected,side]=face_id;raw=np.asarray(surfaces[face_id].get('normal',[0,0,0]),float);norm=np.linalg.norm(raw)
            if norm:normals[side]=raw/norm
        radial=None if curve.kind=='line' else (origin-curve.arc.center_mm)/curve.arc.radius_mm
        for side in (0,1):
            E,source=frame(origin,t0,normals if side==0 else normals[::-1],radial)
            frames[selected,side]=E;sources[selected,side]=source
        graph._curves.clear()
        if int(pid)%4096==0:check_resources()
    del graph,profiles;gc.collect()
    base=np.load(prior/'features.npy',mmap_mode='r')
    output=np.lib.format.open_memmap(directory/'features.npy',mode='w+',dtype=np.float32,shape=(n*2,152));output[:]=0
    for first in range(0,n*2,8192):output[first:first+8192,:142]=base[first:first+8192]
    # Release copied mmap pages before sparse face updates; semantic fields are
    # already in the output and need no second mapped 142-column table.
    output.flush();output._mmap.close();base._mmap.close()
    output=np.load(directory/'features.npy',mmap_mode='r+')
    check_resources();xyz=np.load(xyzfile,mmap_mode='r');supports=np.load(folder/'surface_support.npz')
    new_distance_entries=0;context_rows=0;maximum=0.;max_radius_drift=0.;face_bindings=[]
    for face_id in np.unique(face_ids[face_ids>=0]):
        sk=surfaces[int(face_id)].get('point_ids_key')
        ids=supports[sk] if sk is not None and sk in supports else []
        if not len(ids):continue
        points=np.asarray(xyz[ids],np.float64);center=points.mean(0);centered=points-center
        radius=float(np.linalg.norm(centered,axis=1).max(initial=0));tree=cKDTree(points)
        # Exact enclosing radius about observed centroid is a finite support extent, not thickness truth.
        for _,R,t in transformations():
            rr=float(np.linalg.norm(centered@R.T,axis=1).max(initial=0))
            max_radius_drift=max(max_radius_drift,abs(rr-radius));assert abs(rr-radius)<=1e-9+1e-12*radius
        for original_side in (0,1):
            fragments=np.flatnonzero(face_ids[:,original_side]==face_id)
            for start in range(0,len(fragments),2048):
                ix=fragments[start:start+2048]
                query=queries[ix].reshape(-1,3);d,_=tree.query(query,k=1,workers=2)
                distance=np.log1p(d.reshape(-1,16)/10).astype(np.float32)
                for hypothesis in (0,1):
                    slot=original_side if hypothesis==0 else 1-original_side;rows=ix*2+hypothesis
                    # Preserve all original two-face channels. Restore only a known side hidden by pair masking.
                    needs_restore=output[rows,119]==0
                    assert np.all(output[rows,117+slot]==1),'Previously known face mask differs'
                    rr=rows[needs_restore]
                    for bin_id in range(16):output[rr,bin_id*5+slot]=distance[needs_restore,bin_id]
                    new_distance_entries+=int(needs_restore.sum())*16
                    E=frames[ix,hypothesis];source=sources[ix,hypothesis];valid=source!=0
                    local=np.einsum('ni,nij->nj',center-origins[ix],E)/1000
                    context=np.column_stack([local,np.full(len(ix),radius/1000),valid]).astype(np.float32)
                    context[~valid]=0
                    output[rows,142+slot*5:147+slot*5]=context;context_rows+=int(valid.sum())
                    for name,R,t in transformations():
                        transformed_E=np.einsum('ij,njk->nik',R,E)
                        transformed=np.einsum('ni,nij->nj',(center@R.T+t)-(origins[ix]@R.T+t),transformed_E)/1000
                        changed=np.column_stack([transformed,np.full(len(ix),radius/1000),valid]).astype(np.float32);changed[~valid]=0
                        if transform_observer is not None:transform_observer(name,rows,142+slot*5,changed)
                        diff=float(np.abs(context.astype(float)-changed.astype(float)).max(initial=0));maximum=max(maximum,diff)
                        if not np.allclose(context,changed,rtol=8*np.finfo(np.float32).eps,atol=2e-12):
                            raise ValueError(f'Relative support context consistency failed: {key}/{face_id}/{name}')
                check_resources()
        face_bindings.append(dict(face=int(face_id),support_points=len(points),observed_radius_mm=radius))
        del points,centered,tree;gc.collect()
    output.flush();assert np.isfinite(output).all()
    # All original relative geometry, semantic masks and both-face support features are preserved.
    base=np.load(prior/'features.npy',mmap_mode='r')
    for first in range(0,n*2,8192):
        stop=min(first+8192,n*2)
        assert np.array_equal(output[first:stop,80:142],base[first:stop,80:142])
        both=base[first:stop,119]!=0
        assert np.array_equal(output[first:stop,:80][both],base[first:stop,:80][both])
    output._mmap.close();base._mmap.close();xyz._mmap.close();supports.close()
    receipt=dict(source=key,rows=n*2,columns=152,features_sha256=digest(directory/'features.npy'),
        alias_sha256=pr['alias_sha256'],geometry_sha256=pr['geometry_sha256'],base_relative_sha256=pr['features_sha256'],
        original_input_sha256=cohort['input_sha256'],restored_known_distance_entries=new_distance_entries,
        valid_face_context_entries=context_rows,maximum_rigid_context_difference=maximum,maximum_radius_drift_mm=max_radius_drift,
        unchanged_relative_and_mask_channels=True,unchanged_both_known_face_bins=True,
        face_statistics=face_bindings,GT_reads=0,STEP_reads=0,seconds=time.perf_counter()-started)
    write(directory/'receipt.json',receipt);print('OBSERVED_FACE_INPUT_READY',key,receipt['seconds'],flush=True)
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run329');ap.add_argument('--worker',action='store_true')
    args=ap.parse_args();dest=OUT/args.run
    if args.worker:
        start=time.perf_counter()
        try:
            cases=[]
            for key in KEYS:cases.append(prepare(key,dest));gc.collect()
            write(dest/'all_relative_inputs_frozen_before_labels.json',dict(cases=cases,GT_reads=0))
            control=audit_labels(dest)
            write(dest/'worker_receipt.json',dict(status=control['status'],cases=[{k:v for k,v in c.items() if k!='face_statistics'} for c in cases],
                classification_conflict_rows=control['classification_conflict_rows'],endpoint_feasibility=control['endpoint_feasibility'],
                fit_calls=0,seconds=time.perf_counter()-start,full_contract_pass=False))
        except BaseException:write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0,seconds=time.perf_counter()-start));raise
        return
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir()
    for p in (__file__,ROOT/'weld_pipeline/sjg_relative_geometry_1135.py'):shutil.copy2(p,snapshot)
    write(dest/'registered_protocol.json',dict(status='KNOWN_FACE_RELATIVE_CONTEXT_CONTROL_REGISTERED',
        TRAIN=list(KEYS),G=213,columns=152,fit_calls=0,wall_seconds=180,
        local_extension_to_Pro324='TRAIN counterexamples reveal masked known single-face input; restore known bins and append relative observed face centroid/radius/valid',
        no_new_samples=True,no_label_changes=True,no_DEV_selection=True,
        context_per_ordered_face=['centroid_local_x_m','centroid_local_y_m','centroid_local_z_m','observed_centroid_radius_m','valid'],
        unknown_second_face_not_invented=True,original_geometry_and_masks_preserved=True,utc=utc()))
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_observed_face_features_1138','--run',args.run,'--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='OBSERVED_FACE_CONTROL_INCOMPLETE',watchdog=result));raise RuntimeError('Observed-face control incomplete')
    receipt=read(dest/'worker_receipt.json');receipt['external_wall_seconds']=result['wall_seconds']
    write(dest/'receipt.json',receipt);print('OBSERVED_FACE_GATE_RESULT',receipt['status'],receipt['classification_conflict_rows'],receipt['endpoint_feasibility'],flush=True)


if __name__=='__main__':main()
