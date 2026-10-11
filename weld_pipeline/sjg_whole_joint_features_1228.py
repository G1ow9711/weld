"""Point-cloud-only whole-carrier evidence, independent of sampling partitions.

160 NEW channels. These are not the old selector U160 and must never be passed
to its normalization or weights. IDs are stored only for row association.
"""
import argparse
from collections import defaultdict
import gc
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, check_resources
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_relative_geometry_1135 import frame

SCHEMA='WHOLE_FINITE_JOINT_POINT_SUPPORT_160_V1'


def coverage_bins(spans,length,bins=8):
    edges=np.linspace(0,length,bins+1)
    overlap=np.maximum(0,np.minimum(edges[1:,None],spans[None,:,1])-np.maximum(edges[:-1,None],spans[None,:,0]))
    return np.r_[overlap@(spans[:,2]>0)/(length/bins),overlap@(spans[:,3]>0)/(length/bins)]


def surface_context(points,origin,E,spacing):
    local=(points-origin)@E/1000.
    return np.r_[1.,np.log1p(len(points)),np.log1p(spacing[0]),np.log1p(spacing[1]),
                 local.mean(0),local.std(0),local.min(0),local.max(0)]


def controls(dest):
    a=np.array([[0,30,1,0],[30,40,0,1],[40,100,1,0]],float)
    b=np.array([[0,10,1,0],[10,30,1,0],[30,35,0,1],[35,40,0,1],[40,100,1,0]],float)
    assert np.allclose(coverage_bins(a,100),coverage_bins(b,100),atol=1e-15,rtol=0)
    from weld_pipeline.sjg_relative_geometry_1135 import transformations
    points=np.array([[0,0,0],[20,0,0],[20,30,0],[0,30,0]],float)
    origin=np.array([2.,3.,4.]);t=np.array([1.,0.,0.]);normals=np.array([[0.,0.,1.],[0.,1.,0.]])
    E,_=frame(origin,t,normals,None);reference=surface_context(points,origin,E,[10,20])
    maximum=0.
    for name,R,shift in transformations():
        moved,_=frame(origin@R.T+shift,t@R.T,normals@R.T,None)
        delta=float(np.abs(surface_context(points@R.T+shift,origin@R.T+shift,moved,[10,20])-reference).max())
        assert delta<1e-12;maximum=max(maximum,delta)
    write(dest/'input_controls.json',dict(status='WHOLE_JOINT_INPUT_CONTROLS_PASSED',
        partition_subdivision_invariant=True,rigid_surface_context_maximum_error=maximum,
        no_claim_sampling_pattern_invariance=True,fit_calls=0,GT_reads=0))


def prepare(key,dest):
    begin=time.perf_counter();check_resources();case=load_case(key);graph=case['graph'];n=len(graph.profiles)
    directory=dest/f'source_{key}';directory.mkdir(exist_ok=False)
    print('WHOLE_JOINT_INPUT',key,'PROFILES',n,flush=True)
    x=np.lib.format.open_memmap(directory/'joint_features.npy',mode='w+',dtype=np.float32,shape=(n*2,160));x[:]=0
    query=np.empty((n,16,3));origins=np.empty((n,3));frames=np.empty((n,2,3,3));valid_frames=np.zeros((n,2),bool)
    groups=defaultdict(list)
    for pid,p in enumerate(graph.profiles):
        curve=graph.curve(pid);origin=curve.at(np.array([0.]))[0];tangent=curve.tangent(np.array([0.]))[0]
        normals=np.zeros((2,3));known=np.zeros(2,bool)
        for side,face_id in enumerate(p.get('face_indices',[])[:2]):
            raw=np.array(case['surfaces'][face_id].get('normal',[0,0,0]),float);norm=np.linalg.norm(raw)
            if norm>0:normals[side]=raw/norm;known[side]=True
            groups[int(face_id)].append((pid,side))
        radial=None if curve.kind=='line' else (origin-curve.arc.center_mm)/curve.arc.radius_mm
        query[pid]=curve.at((np.arange(16)+.5)/16*curve.length);origins[pid]=origin
        spans=graph.spans[graph.offsets[pid]:graph.offsets[pid+1]];coverage=coverage_bins(spans,curve.length)
        for side in (0,1):
            normals2=normals if side==0 else normals[::-1];known2=known if side==0 else known[::-1]
            E,source=frame(origin,tangent,normals2,radial);frames[pid,side]=E;valid_frames[pid,side]=bool(source)
            x[pid*2+side,96:112]=[np.log1p(curve.length),curve.kind=='line',curve.kind=='arc',
                curve.kind=='arc' and curve.arc.is_closed,0 if curve.kind=='line' else 1/curve.arc.radius_mm,
                0 if curve.kind=='line' else curve.arc.sweep_rad,abs(normals2[0]@normals2[1]),normals2[0]@normals2[1],
                tangent@normals2[0],tangent@normals2[1],*known2,1 if side==0 else -1,bool(source),coverage[:8].mean(),coverage[8:].mean()]
            x[pid*2+side,144:160]=coverage
        if pid%4096==0:graph._curves.clear()
    graph._curves.clear();case['x']._mmap.close()
    cohort=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/cohort['input_file'])==cohort['input_sha256']
    xyz=np.load(OUT/cohort['input_file'],mmap_mode='r');distances=np.zeros((n,16,2));nearest=np.zeros((n,16,2,3))
    seen=np.zeros((n,2),bool);scales=np.ones((n,2));face_stats=[]
    with np.load(OUT/f'run185/source_{key}/surface_support.npz') as support:
        for face_index,(face_id,entries) in enumerate(groups.items()):
            sk=case['surfaces'][face_id].get('point_ids_key')
            if sk is None or sk not in support or not len(support[sk]):continue
            points=np.asarray(xyz[support[sk]],float);tree=cKDTree(points)
            # Fixed, geometry-only support subsample; bounds are sampled extents,
            # not certified physical boundaries or approved thickness labels.
            sample=points[np.linspace(0,len(points)-1,min(512,len(points)),dtype=int)]
            if len(points)>1:
                dd=tree.query(sample,k=2,workers=2)[0][:,1];positive=dd[dd>1e-9]
                spacing=np.quantile(positive,[.5,.9]) if len(positive) else np.zeros(2)
            else:spacing=np.zeros(2)
            scale=spacing[0] if spacing[0]>1e-9 else 1.
            for first in range(0,len(entries),1024):
                items=np.array(entries[first:first+1024],int);ids,sides=items.T
                d,ix=tree.query(query[ids].reshape(-1,3),k=1,workers=2)
                distances[ids,:,sides]=d.reshape(-1,16);nearest[ids,:,sides]=points[ix].reshape(-1,16,3)
                seen[ids,sides]=True;scales[ids,sides]=scale
                for pid,original_side in items:
                    for side in (0,1):
                        if valid_frames[pid,side]:
                            values=surface_context(sample,origins[pid],frames[pid,side],spacing)
                            values[1]=np.log1p(len(points))
                            slot=original_side if side==0 else 1-original_side
                            x[pid*2+side,112+slot*16:128+slot*16]=values
            face_stats.append(dict(face=face_id,points=len(points),spacing_median_mm=float(spacing[0]),spacing_p90_mm=float(spacing[1])))
            del tree,points,sample
            if face_index%32==0:
                print('WHOLE_JOINT_FACES',key,face_index+1,'/',len(groups),flush=True);check_resources()
    for first in range(0,n,4096):
        ids=np.arange(first,min(first+4096,n));both=seen[ids].all(1)
        separation=np.linalg.norm(nearest[ids,:,0]-nearest[ids,:,1],axis=2)
        for side in (0,1):
            slots=[0,1] if side==0 else [1,0];bins=np.zeros((len(ids),16,6),np.float32)
            for slot,original in enumerate(slots):
                bins[:,:,slot]=np.where(seen[ids,original,None],np.log1p(distances[ids,:,original]/10),0)
                bins[:,:,2+slot]=np.where(seen[ids,original,None],np.log1p(distances[ids,:,original]/scales[ids,original,None]),0)
            bins[:,:,4]=np.where(both[:,None],np.log1p(separation/10),0);bins[:,:,5]=both[:,None]
            x[ids*2+side,:96]=bins.reshape(-1,96)
    assert np.isfinite(x).all();x.flush();x._mmap.close();xyz._mmap.close()
    write(directory/'receipt.json',dict(status='WHOLE_JOINT_POINT_INPUT_READY',schema=SCHEMA,source=key,profiles=n,rows=2*n,columns=160,
        feature_sha256=digest(directory/'joint_features.npy'),input_sha256=cohort['input_sha256'],geometry_sha256=case['geometry_sha256'],
        face_statistics=face_stats,raw_xyz_unchanged=True,GT_reads=0,STEP_reads=0,fit_calls=0,
        fragments_preserved=True,physical_boundary_claim=False,sampled_face_context_max_points=512,
        feature_file_not_selector_U160=True,seconds=time.perf_counter()-begin))
    print('WHOLE_JOINT_INPUT_READY',key,'SECONDS',round(time.perf_counter()-begin,2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--sources',nargs='+',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);controls(dest)
    write(dest/'registered_protocol.json',dict(schema=SCHEMA,sources=args.sources,TRAIN=['21','BM79A_native_local','10'],DEV=['64'],
        purpose='Whole-joint front-end input and supervision preparation; not model acceptance',external_consultation_paused=True,
        preserve_best='run379',source_sha256=digest(__file__),GT_reads=0,STEP_reads=0,model_fits=0))
    try:
        for key in args.sources:prepare(key,dest);gc.collect()
        write(dest/'receipt.json',dict(status='WHOLE_JOINT_INPUTS_COMPLETE',sources=args.sources,fit_calls=0,full_contract_pass=False))
    except BaseException:write(dest/'failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise


if __name__=='__main__':main()
