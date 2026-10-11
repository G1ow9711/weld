"""Pro324: 142-channel relative representation and zero-fit three-TRAIN gates."""
import argparse
import gc
import hashlib
import importlib.util
import shutil
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_joint_object_features_1106 import side_rows

KEYS=('21','BM79A_native_local','10')


def transformations():
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);a=np.deg2rad(37.)
    cross=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    oblique=np.eye(3)*np.cos(a)+(1-np.cos(a))*np.outer(axis,axis)+np.sin(a)*cross
    return [('x100',np.eye(3),np.array([100.,0.,0.])),('y100',np.eye(3),np.array([0.,100.,0.])),
            ('z100',np.eye(3),np.array([0.,0.,100.])),
            ('z90',np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]),np.zeros(3)),
            ('axis123_37deg',oblique,np.zeros(3))]


def frame(origin,tangent,normals,radial):
    e1=tangent/np.linalg.norm(tangent)
    for source,v in enumerate([*normals]+([] if radial is None else [radial]),1):
        u=v-e1*np.dot(v,e1);n=np.linalg.norm(u)
        if n>256*np.finfo(float).eps*max(1.,np.linalg.norm(v)):
            e2=u/n;e3=np.cross(e1,e2)
            return np.column_stack([e1,e2,e3]),source
    return np.zeros((3,3)),0


def local_values(points,tangents,origin,t0,normals,radial):
    E,source=frame(origin,t0,normals,radial)
    p=(points-origin)@E/1000.;t=tangents@E;n=normals@E
    # Canonicalize arithmetic zero only; no geometric merging or GT tolerance.
    bound=256*np.finfo(float).eps*max(1.,float(np.abs(origin).max()),float(np.abs(points).max()))/1000
    p[np.abs(p)<=bound]=0.
    t[np.abs(t)<=256*np.finfo(float).eps]=0.
    n[np.abs(n)<=256*np.finfo(float).eps]=0.
    v=np.concatenate([p.reshape(len(points),9),t.reshape(len(points),6),np.tile(n.reshape(1,6),(len(points),1))],axis=1)
    return v.astype(np.float32),E,source


def support_boundaries(spans,length,closed):
    supported=spans[:,2]>0
    changes=spans[1:,0][supported[1:]!=supported[:-1]].tolist()
    if closed:
        if supported[0]!=supported[-1]:changes.append(0.)
    else:
        if supported[0]:changes.append(0.)
        if supported[-1]:changes.append(float(length))
    return np.array(sorted(changes),np.float64)


def boundary_context(spans,boundaries,length,closed):
    v=np.zeros((len(spans),8),np.float32)
    if not len(boundaries):return v
    for i,(lo,hi) in enumerate(spans[:,:2]):
        for j,q in enumerate((lo,hi)):
            if closed:
                backward=-np.min(np.mod(q-boundaries,length));forward=np.min(np.mod(boundaries-q,length))
                v[i,j*2:j*2+2]=[backward/1000,forward/1000];v[i,4+j*2:6+j*2]=1
            else:
                prev=boundaries[boundaries<=q];nxt=boundaries[boundaries>=q]
                if len(prev):v[i,j*2]=(prev[-1]-q)/1000;v[i,4+j*2]=1
                if len(nxt):v[i,j*2+1]=(nxt[0]-q)/1000;v[i,5+j*2]=1
    return v


def prepare(key,dest):
    started=time.perf_counter();directory=dest/f'source_{key}';directory.mkdir()
    basepath=OUT/f'run287/source_{key}/object_features.npy'
    assert digest(basepath)==read(basepath.parent/'receipt.json')['features_sha256']
    aliaspath=OUT/f'run288/source_{key}_alias_map.npz'
    binding=next(v for v in read(OUT/'run288/all_alias_maps_frozen_before_labels.json')['cases'] if v['source']==key)
    assert digest(aliaspath)==binding['map_sha256']
    with np.load(aliaspath) as z:canonical=z['canonical_fragments']
    geometry=next(v for v in read(OUT/'run285/all_geometry_frozen_before_labels.json')['cases'] if v['source']==key)
    profilepath=OUT/f'run185/source_{key}/profiles.json';surfacepath=profilepath.parent/'surfaces.json'
    assert digest(profilepath)==geometry['source_files']['profiles.json']
    assert digest(surfacepath)==geometry['source_files']['surfaces.json']
    assert digest(OUT/'run285'/geometry['geometry_file'])==geometry['geometry_sha256']
    profiles=read(profilepath);surfaces=read(surfacepath)
    with np.load(OUT/'run285'/geometry['geometry_file']) as z:graph=JointSegmentGraph(profiles,z['offsets'],z['spans'])
    base=np.load(basepath,mmap_mode='r')
    output=np.lib.format.open_memmap(directory/'features.npy',mode='w+',dtype=np.float32,shape=(len(canonical)*2,142))
    output[:]=0
    for start in range(0,len(canonical),8192):
        fr=canonical[start:start+8192];rows=np.arange(start,start+len(fr))*2
        output[rows,:128]=base[fr];output[rows+1,:128]=side_rows(base,fr,np.ones(len(fr),np.int8))
    pids=np.searchsorted(graph.offsets,canonical,side='right')-1
    max_abs={name:0. for name,_,_ in transformations()};valid=0;invalid=0
    assert np.all(np.diff(pids)>=0)
    unique_pids,starts,counts=np.unique(pids,return_index=True,return_counts=True)
    for pid,first,count in zip(unique_pids,starts,counts):
        indices=np.arange(first,first+count);fr=canonical[indices];spans=graph.spans[fr]
        curve=graph.curve(pid);profile=profiles[int(pid)]
        all_spans=graph.spans[graph.offsets[pid]:graph.offsets[pid+1]]
        closed=curve.kind=='arc' and curve.arc.is_closed
        boundaries=support_boundaries(all_spans,curve.length,closed)
        context=boundary_context(spans,boundaries,curve.length,closed)
        q=np.column_stack([spans[:,0],spans[:,1],spans[:,:2].mean(1)])
        points=curve.at(q);tangents=curve.tangent(q[:,:2])
        origin=curve.at(np.array([0.]))[0];t0=curve.tangent(np.array([0.]))[0]
        radial=None if curve.kind=='line' else (origin-curve.arc.center_mm)/curve.arc.radius_mm
        normals=np.zeros((2,3))
        for side,face in enumerate(profile.get('face_indices',[])[:2]):
            raw=np.asarray(surfaces[int(face)].get('normal',[0,0,0]),np.float64);norm=np.linalg.norm(raw)
            if norm:normals[side]=raw/norm
        for side in (0,1):
            ordered=normals if side==0 else normals[::-1]
            values,E,source=local_values(points,tangents,origin,t0,ordered,radial)
            rows=indices*2+side;output[rows,84:105]=values
            output[rows,128:136]=context;output[rows,136]=bool(source);output[rows,137]=source
            # No previously confirmed process-gravity vector is supplied by this input schema.
            output[rows,138:142]=0
            valid+=len(rows)*bool(source);invalid+=len(rows)*(not bool(source))
            for name,R,t in transformations():
                altered,_,new_source=local_values(points@R.T+t,tangents@R.T,origin@R.T+t,t0@R.T,
                                                 ordered@R.T,None if radial is None else radial@R.T)
                assert source==new_source,f'Frame source changed: {key}/{pid}/{side}/{name}'
                difference=float(np.abs(values.astype(float)-altered.astype(float)).max(initial=0))
                max_abs[name]=max(max_abs[name],difference)
                if not np.allclose(values,altered,rtol=8*np.finfo(np.float32).eps,atol=2e-12):
                    write(directory/'equivariance_failure.json',dict(profile=int(pid),side=side,transform=name,
                          maximum_absolute_difference=difference,reference=values.tolist(),transformed=altered.tolist()))
                    raise ValueError('Relative representation numeric consistency failed')
        if int(pid)%1024==0:check_resources()
    output.flush();assert np.isfinite(output).all();output._mmap.close();base._mmap.close()
    receipt=dict(source=key,rows=2*len(canonical),columns=142,features_sha256=digest(directory/'features.npy'),
        alias_sha256=binding['map_sha256'],geometry_sha256=geometry['geometry_sha256'],
        frame_valid_rows=valid,frame_invalid_rows=invalid,transforms_maximum_absolute_feature_difference=max_abs,
        equivariance_passed=True,GT_reads=0,STEP_reads=0,seconds=time.perf_counter()-started)
    write(directory/'receipt.json',receipt);print('RELATIVE_INPUT_READY',key,receipt['rows'],receipt['seconds'],flush=True)
    return receipt


def audit_labels(dest):
    frozen=OUT/'run302/source_snapshot/sjg_joint_forest_train_1109.py'
    spec=importlib.util.spec_from_file_location('_relative_original_labels',frozen)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    tables={key:np.load(dest/f'source_{key}/features.npy',mmap_mode='r') for key in KEYS}
    seen={};conflicts=[];count=0;positive_groups={};cases=[]
    for key in KEYS:
        case=old.load_case(key,supervision=True);x=tables[key];assert len(x)==len(case['y'])
        np.savez(dest/f'source_{key}/original_supervision.npz',labels=case['y'],ratios=case['q'],canonical=case['canonical'])
        for row in range(len(x)):
            v=np.array(x[row],copy=True);v[v==0]=0
            h=hashlib.blake2b(v.tobytes(),digest_size=16).digest();label=int(case['y'][row])
            original=seen.get(h)
            if original is None:seen[h]=(key,row,label)
            else:
                pk,pr,pl=original
                assert np.array_equal(tables[pk][pr],x[row]),'Hash collision, not equal input'
                if pl!=label:
                    count+=1
                    if len(conflicts)<30:conflicts.append(dict(first=dict(source=pk,row=pr,label=pl),second=dict(source=key,row=row,label=label)))
            if label:
                fragment=int(case['canonical'][row//2]);curve=case['graph'].curve(case['graph'].profile(fragment))
                span=float(np.diff(case['graph'].spans[fragment,:2])[0])
                positive_groups.setdefault(h,[]).append(dict(source=key,row=row,label=label,ratio=case['q'][row].tolist(),
                    span_mm=span,radius_mm=None if curve.kind=='line' else curve.arc.radius_mm))
        cases.append(dict(source=key,rows=len(x),positive_rows=int((case['y']!=0).sum()),G=case['G'],
            witness_indices_identical=True,original_alias_sha256=case['alias_map_sha256']))
        del case;gc.collect();check_resources()
        print('RELATIVE_LABEL_SCAN',key,'CLASS_CONFLICTS',count,flush=True)
    # Endpoint feasibility is required only after the classification necessary condition passes.
    endpoint_status='NOT_RUN_CLASSIFICATION_GATE_FAILED';endpoint_conflicts=[]
    if count==0:
        endpoint_status='PASSED'
        def intersect(a,b):return [(max(lo,lc),min(hi,hc)) for lo,hi in a for lc,hc in b if max(lo,lc)<=min(hi,hc)]
        for members in positive_groups.values():
            allowed=[[(0.,1.)],[(0.,1.)]]
            for m in members:
                radius=m['radius_mm'];span=m['span_mm']
                if radius is not None and 2*radius<=3:continue
                delta=3/span if radius is None else 2*radius*np.arcsin(3/(2*radius))/span
                period=None if radius is None else 2*np.pi*radius/span
                for endpoint,q in enumerate(m['ratio']):
                    parts=[]
                    for k in ([0] if period is None else [-2,-1,0,1,2]):
                        center=q+(0 if period is None else k*period)
                        lo,hi=max(0.,center-delta),min(1.,center+delta)
                        if lo<=hi:parts.append((lo,hi))
                    allowed[endpoint]=intersect(allowed[endpoint],parts)
            if not all(allowed) or min(a for a,b in allowed[0])>=max(b for a,b in allowed[1]):
                endpoint_status='FAILED';endpoint_conflicts.append(members)
    report=dict(status='RELATIVE_SUPERVISION_READY' if count==0 and endpoint_status=='PASSED' else 'RELATIVE_SUPERVISION_CONFLICT_STOP',
        cases=cases,classification_conflict_rows=count,classification_examples=conflicts,
        endpoint_feasibility=endpoint_status,endpoint_conflicts=endpoint_conflicts,
        full_G=sum(c['G'] for c in cases),no_rows_deleted=True,DEV_labels_read=0,fit_calls=0)
    write(dest/'supervision_control.json',report)
    for x in tables.values():x._mmap.close()
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run326');ap.add_argument('--worker',action='store_true')
    args=ap.parse_args();dest=OUT/args.run
    if args.worker:
        started=time.perf_counter()
        try:
            cases=[]
            for key in KEYS:cases.append(prepare(key,dest));gc.collect()
            write(dest/'all_relative_inputs_frozen_before_labels.json',dict(cases=cases,GT_reads=0))
            control=audit_labels(dest)
            write(dest/'worker_receipt.json',dict(status=control['status'],cases=cases,
                classification_conflict_rows=control['classification_conflict_rows'],endpoint_feasibility=control['endpoint_feasibility'],
                fit_calls=0,seconds=time.perf_counter()-started,full_contract_pass=False,utc=utc()))
        except BaseException:
            write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0,seconds=time.perf_counter()-started));raise
        return
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir()
    shutil.copy2(__file__,snapshot)
    write(dest/'registered_protocol.json',dict(status='RELATIVE_GEOMETRY_3TRAIN_V1_REGISTERED',TRAIN=list(KEYS),G=213,
        columns=142,transforms=[dict(name=n,R=R.tolist(),translation_mm=t.tolist()) for n,R,t in transformations()],
        numerical_comparison=dict(rtol=8*np.finfo(np.float32).eps,atol=2e-12),
        numerical_zero='256*float64_epsilon*coordinate_scale; vectors 256*epsilon, not GT tolerance',
        boundary_distance_unit='metres signed along original carrier',frame_sources=['UNKNOWN','ORDERED_FACE_A','ORDERED_FACE_B','ARC_RADIAL'],
        process_direction='UNKNOWN; no world-axis assumption',control_wall_seconds=180,fit_calls=0,
        pro_receipt_sha256=digest(OUT/'pro_review_run324/receipt.json'),utc=utc()))
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_relative_geometry_1135','--run',args.run,'--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='RELATIVE_CONTROL_INCOMPLETE',watchdog=result));raise RuntimeError('Relative control incomplete')
    receipt=read(dest/'worker_receipt.json');receipt['external_wall_seconds']=result['wall_seconds']
    write(dest/'receipt.json',receipt);print('RELATIVE_GATE_RESULT',receipt,flush=True)


if __name__=='__main__':main()
