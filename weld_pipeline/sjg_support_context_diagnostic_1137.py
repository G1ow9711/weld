"""TRAIN-only diagnosis of missing single-face support and finite face context."""
import argparse
import gc
import shutil
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_relative_geometry_1135 import frame,transformations


def worker(dest):
    start=time.perf_counter();check_resources()
    evidence=read(OUT/'run327/conflict_geometry_trace.json');records=evidence['rows']
    cohort=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='10')
    xyzfile=OUT/cohort['input_file'];assert digest(xyzfile)==cohort['input_sha256']
    folder=OUT/'run185/source_10';binding=next(c for c in read(OUT/'run285/all_geometry_frozen_before_labels.json')['cases'] if c['source']=='10')
    for name in ('profiles.json','surfaces.json','surface_support.npz'):
        assert digest(folder/name)==binding['source_files'][name]
    profiles=read(folder/'profiles.json');surfaces=read(folder/'surfaces.json')
    xyz=np.load(xyzfile,mmap_mode='r');support=np.load(folder/'surface_support.npz')
    old=np.load(OUT/'run287/source_10/object_features.npy',mmap_mode='r')
    contexts=[];distances=[];knowns=[];details=[];cache={};maximum=0.
    for record in records:
        row=record['row'];profile=profiles[record['profile']];curve=CurveCarrier(profile['representation'])
        origin=curve.at(np.array([0.]))[0];t0=curve.tangent(np.array([0.]))[0]
        radial=None if curve.kind=='line' else (origin-curve.arc.center_mm)/curve.arc.radius_mm
        faces=profile.get('face_indices',[])[:2];normals=np.zeros((2,3))
        for side,face in enumerate(faces):
            raw=np.asarray(surfaces[face].get('normal',[0,0,0]),float);norm=np.linalg.norm(raw)
            if norm:normals[side]=raw/norm
        if row%2:faces=faces[::-1];normals=normals[::-1]
        E,source=frame(origin,t0,normals,radial)
        lo,hi=record['span_mm'];query=curve.at(lo+(np.arange(16)+.5)*(hi-lo)/16)
        context=np.zeros((2,10),np.float32);distance=np.zeros((16,2),np.float32);known=np.zeros(2,bool)
        counts=[]
        for side,face in enumerate(faces):
            if face not in cache:
                support_key=surfaces[face].get('point_ids_key')
                ids=support[support_key] if support_key is not None and support_key in support else []
                points=np.asarray(xyz[ids],np.float64).reshape(-1,3)
                cache[face]=(points,cKDTree(points) if len(points) else None)
            points,tree=cache[face];counts.append(len(points))
            if not len(points):continue
            known[side]=True
            d,ix=tree.query(query,k=1,workers=1);distance[:,side]=np.log1p(d/10)
            if not source:continue
            local=(points-origin)@E/1000
            raw_context=np.r_[local.mean(0),local.min(0),local.max(0),1.]
            context[side]=raw_context
            for name,R,t in transformations():
                new_origin=origin@R.T+t
                new_E,new_source=frame(new_origin,t0@R.T,normals@R.T,None if radial is None else radial@R.T)
                assert new_source==source
                transformed=((points@R.T+t)-new_origin)@new_E/1000
                changed=np.r_[transformed.mean(0),transformed.min(0),transformed.max(0),1.].astype(np.float32)
                diff=float(np.max(np.abs(context[side].astype(float)-changed.astype(float))));maximum=max(maximum,diff)
                assert np.allclose(context[side],changed,rtol=8*np.finfo(np.float32).eps,atol=2e-12)
                # Rigid distances to exact original nearest witnesses, independent of local-frame construction.
                new_d=np.linalg.norm((query@R.T+t)-(points[ix]@R.T+t),axis=1)
                assert np.allclose(d,new_d,rtol=1e-10,atol=1e-9)
        contexts.append(context.ravel());distances.append(distance.ravel());knowns.append(known)
        old_bins=old[record['fragment'],:80].reshape(16,5)
        details.append(dict(row=row,face_count=len(faces),support_point_counts=counts,
            old_pair_bins_all_zero=bool(np.all(old_bins==0)),known_face_count=int(known.sum()),
            own_face_distance_signal_nonzero=bool(np.any(distance!=0)),frame_source=source,
            original_label=record['label'],certified_compatible_positive=record['certified_compatible_event_count']>0))
        check_resources()
    contexts=np.asarray(contexts,np.float32);distances=np.asarray(distances,np.float32)
    index={r['row']:i for i,r in enumerate(records)};pairs=[]
    for p in evidence['pairs']:
        a,b=index[p['first']],index[p['second']]
        pairs.append(dict(first=p['first'],second=p['second'],
            finite_support_context_different=not np.array_equal(contexts[a],contexts[b]),
            maximum_context_difference_mm=float(np.max(np.abs(contexts[a]-contexts[b]))*1000),
            known_face_distance_different=not np.array_equal(distances[a],distances[b]),
            old_negative_has_certified_positive=p['original_negative_has_certified_positive_alternative']))
    np.savez(dest/'diagnostic_descriptors.npz',rows=np.array([r['row'] for r in records]),
             finite_face_context=contexts,known_face_distance_bins=distances,known_faces=np.array(knowns))
    summary=dict(examined_rows=len(records),single_face_rows=sum(d['face_count']==1 for d in details),
        old_pair_bins_all_zero_rows=sum(d['old_pair_bins_all_zero'] for d in details),
        known_face_signal_nonzero_rows=sum(d['own_face_distance_signal_nonzero'] for d in details),
        conflicting_comparisons=len(pairs),separated_by_finite_support_context=sum(p['finite_support_context_different'] for p in pairs),
        separated_by_known_face_distance=sum(p['known_face_distance_different'] for p in pairs),
        maximum_context_rigid_difference=maximum)
    support.close();xyz._mmap.close();old._mmap.close()
    write(dest/'worker_receipt.json',dict(status='MISSING_INPUT_CONTEXT_DIAGNOSTIC_COMPLETE',summary=summary,
        rows=details,pairs=pairs,fit_calls=0,DEV_reads=0,STEP_reads=0,seconds=time.perf_counter()-start,
        all_TRAIN_feature_generation_or_conflict_gate=False,extraction_correct_rate=None,extraction_precision_mm=None))
    print('SUPPORT_CONTEXT_DIAGNOSTIC',summary,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run328');ap.add_argument('--worker',action='store_true')
    args=ap.parse_args();dest=OUT/args.run
    if args.worker:
        try:worker(dest)
        except BaseException:write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise
        return
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    dest.mkdir(exist_ok=False);shutil.copy2(__file__,dest/'source_snapshot.py')
    write(dest/'registered_protocol.json',dict(status='TRAIN_CONFLICT_INPUT_CONTEXT_DIAGNOSTIC_REGISTERED',source='10',
        selection='all 18 rows from run327 conflict trace; diagnostic only, not model training',
        context_fields_per_ordered_face=['support_centroid_local_xyz_m','support_min_local_xyz_m','support_max_local_xyz_m','known'],
        nearest_support='16 original bin centers, per-known-face log1p(distance/10), unknown separately masked',
        context_dimensions=20,coordinates_from='original PCD support, not STEP or GT',
        class_labels_not_modified=True,fit_calls=0,wall_seconds=180,utc=utc()))
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_support_context_diagnostic_1137','--run',args.run,'--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='CONTEXT_DIAGNOSTIC_INCOMPLETE',watchdog=result));raise RuntimeError('Context diagnostic incomplete')
    receipt=read(dest/'worker_receipt.json');receipt['external_wall_seconds']=result['wall_seconds']
    write(dest/'receipt.json',receipt)


if __name__=='__main__':main()
