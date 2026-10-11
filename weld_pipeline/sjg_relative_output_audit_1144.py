"""Frozen-model rigid feature intervention; no new extraction score or fit.

Recomputes relative geometry and observed centroids from float64 source geometry.
Keeps the original candidate set, distance bins and masks; this is not a raw-PCD
preprocessing or whole graph equivariance claim.
"""
import argparse,gc,pickle,shutil,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_relative_geometry_1135 import transformations
from weld_pipeline.sjg_relative_geometry_batch_1136 import prepare as relative
from weld_pipeline.sjg_observed_face_features_1138 import prepare as observed


def worker(dest):
    start=time.perf_counter();path=OUT/'run330/model.pkl'
    assert digest(path)==read(OUT/'run330/receipt.json')['model_sha256']
    with path.open('rb') as f:models=pickle.load(f)
    classifier=models['object'];endpoint=models['endpoint'];classifier.verbose=0;endpoint.verbose=0
    positive=np.flatnonzero(classifier.classes_!=0);results=[]
    base_dir=dest/'recomputed142';base_dir.mkdir()
    observed_dir=dest/'recomputed152';observed_dir.mkdir()
    for key,run in [('21','run329'),('BM79A_native_local','run329'),('10','run329'),('64','run335')]:
        check_resources();xfile=OUT/run/f'source_{key}/features.npy'
        assert digest(xfile)==read(xfile.parent/'receipt.json')['features_sha256']
        original=np.load(xfile,mmap_mode='r');folder=dest/f'source_{key}';folder.mkdir()
        variants={}
        for name,_,_ in transformations():
            a=np.lib.format.open_memmap(folder/f'{name}.npy',mode='w+',dtype=np.float32,shape=original.shape)
            for first in range(0,len(a),8192):a[first:first+8192]=original[first:first+8192]
            variants[name]=a
        def observe(name,rows,column,values):variants[name][rows,column:column+values.shape[1]]=values
        relative(key,base_dir,transform_observer=observe)
        observed(key,observed_dir,prior_run=str(base_dir.relative_to(OUT)),transform_observer=observe)
        rebuilt=np.load(observed_dir/f'source_{key}/features.npy',mmap_mode='r')
        assert np.array_equal(rebuilt,original),'Observer must preserve actual fitted input exactly'
        rebuilt._mmap.close()
        alias_run='run293' if key=='64' else 'run288';geometry_run='run293' if key=='64' else 'run285'
        with np.load(OUT/alias_run/f'source_{key}_alias_map.npz') as z:canonical=z['canonical_fragments']
        g=next(v for v in read(OUT/geometry_run/'all_geometry_frozen_before_labels.json')['cases'] if v['source']==key)
        with np.load(OUT/geometry_run/g['geometry_file']) as z:spans=z['spans'][canonical]
        lengths=np.repeat(spans[:,1]-spans[:,0],2)
        def predict(a):
            p=[];q=[];category=[]
            for first in range(0,len(a),8192):
                v=classifier.predict_proba(a[first:first+8192]);p.append(v[:,positive].sum(1))
                category.append(classifier.classes_[positive[v[:,positive].argmax(1)]])
                q.append(np.sort(endpoint.predict(a[first:first+8192]),axis=1))
            return np.concatenate(p),np.concatenate(q),np.concatenate(category)
        p0,q0,c0=predict(original);case=dict(source=key,rows=len(original),baseline_positive=int((p0>=.5).sum()),transforms=[])
        for name,a in variants.items():
            a.flush();p,q,c=predict(a);active=(p0>=.5)|(p>=.5)
            travel=np.abs(q-q0).max(1)*lengths
            case['transforms'].append(dict(name=name,maximum_feature_difference=float(np.abs(a-original).max()),
                maximum_probability_difference=float(np.abs(p-p0).max()),admission_changes=int(((p>=.5)!=(p0>=.5)).sum()),
                active_category_changes=int(((c!=c0)&active).sum()),active_endpoint_parameter_travel_max_mm=float(travel[active].max(initial=0)),
                active_endpoint_parameter_travel_over_3mm=int(((travel>3)&active).sum()),positive_count=int((p>=.5).sum()),
                all_rows_endpoint_parameter_travel_max_mm=float(travel.max(initial=0)),features_sha256=digest(folder/f'{name}.npy')))
            a._mmap.close();check_resources()
        original._mmap.close();results.append(case);write(dest/'progress.json',dict(cases=results))
        print('FROZEN_OUTPUT_RIGID_CASE',key,case['transforms'],flush=True);del original,variants,spans,lengths;gc.collect()
    receipt=dict(status='FROZEN_RIGID_FEATURE_OUTPUT_AUDIT_COMPLETE',cases=results,fit_calls=0,GT_reads=0,STEP_reads=0,
        model_sha256=digest(path),seconds=time.perf_counter()-start,new_extraction_score=False,
        whole_pipeline_equivariance_proven=False,scope='frozen candidate features; recomputed local geometry and centroids only')
    write(dest/'worker_receipt.json',receipt)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');args=ap.parse_args();dest=OUT/'run338'
    if args.worker:
        try:worker(dest)
        except BaseException:write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise
        return
    dest.mkdir(exist_ok=False);snap=dest/'source_snapshot';snap.mkdir()
    for name in ('sjg_relative_output_audit_1144.py','sjg_relative_geometry_batch_1136.py','sjg_observed_face_features_1138.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,snap)
    write(dest/'registered_protocol.json',dict(status='FROZEN_POST_FAILURE_DIAGNOSTIC',fit_calls=0,wall_seconds=180,
        no_model_or_threshold_changes=True,no_DEV_labels=True,utc=utc()))
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_relative_output_audit_1144','--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='FROZEN_AUDIT_INCOMPLETE',watchdog=result));raise RuntimeError('Frozen audit incomplete')
    r=read(dest/'worker_receipt.json');r['external_wall_seconds']=result['wall_seconds'];write(dest/'receipt.json',r)
    print('FROZEN_OUTPUT_AUDIT_COMPLETE',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
