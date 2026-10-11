"""Finish run338's untested cases with sparse feature patches and block prediction."""
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
    base_dir=dest/'recomputed142';base_dir.mkdir();observed_dir=dest/'recomputed152';observed_dir.mkdir()
    columns=np.r_[84:105,142:152]
    for key,run in [('10','run329'),('64','run335')]:
        check_resources();xfile=OUT/run/f'source_{key}/features.npy'
        assert digest(xfile)==read(xfile.parent/'receipt.json')['features_sha256']
        original=np.load(xfile,mmap_mode='r');folder=dest/f'source_{key}';folder.mkdir();variants={}
        for name,_,_ in transformations():
            a=np.lib.format.open_memmap(folder/f'{name}_patch.npy',mode='w+',dtype=np.float32,shape=(len(original),31))
            for first in range(0,len(a),4096):a[first:first+4096]=original[first:first+4096,columns]
            variants[name]=a
        def observe(name,rows,column,values):
            offset=column-84 if column==84 else 21+column-142
            variants[name][rows,offset:offset+values.shape[1]]=values
        relative(key,base_dir,transform_observer=observe)
        observed(key,observed_dir,prior_run=str(base_dir.relative_to(OUT)),transform_observer=observe)
        rebuilt=np.load(observed_dir/f'source_{key}/features.npy',mmap_mode='r')
        for first in range(0,len(original),4096):
            assert np.array_equal(rebuilt[first:first+4096],original[first:first+4096])
        rebuilt._mmap.close()
        alias_run='run293' if key=='64' else 'run288';geometry_run='run293' if key=='64' else 'run285'
        with np.load(OUT/alias_run/f'source_{key}_alias_map.npz') as z:canonical=z['canonical_fragments']
        g=next(v for v in read(OUT/geometry_run/'all_geometry_frozen_before_labels.json')['cases'] if v['source']==key)
        with np.load(OUT/geometry_run/g['geometry_file']) as z:spans=z['spans'][canonical]
        lengths=np.repeat(spans[:,1]-spans[:,0],2)
        def predict(patch=None):
            p=[];q=[];category=[];delta=0.
            for first in range(0,len(original),4096):
                x=np.array(original[first:first+4096],copy=True)
                if patch is not None:
                    delta=max(delta,float(np.abs(x[:,columns]-patch[first:first+4096]).max(initial=0)))
                    x[:,columns]=patch[first:first+4096]
                v=classifier.predict_proba(x);p.append(v[:,positive].sum(1))
                category.append(classifier.classes_[positive[v[:,positive].argmax(1)]])
                q.append(np.sort(endpoint.predict(x),axis=1))
            return np.concatenate(p),np.concatenate(q),np.concatenate(category),delta
        p0,q0,c0,_=predict();case=dict(source=key,rows=len(original),baseline_positive=int((p0>=.5).sum()),transforms=[])
        for name,a in variants.items():
            a.flush();p,q,c,delta=predict(a);active=(p0>=.5)|(p>=.5);travel=np.abs(q-q0).max(1)*lengths
            case['transforms'].append(dict(name=name,maximum_feature_difference=delta,
                maximum_probability_difference=float(np.abs(p-p0).max()),admission_changes=int(((p>=.5)!=(p0>=.5)).sum()),
                active_category_changes=int(((c!=c0)&active).sum()),active_endpoint_parameter_travel_max_mm=float(travel[active].max(initial=0)),
                active_endpoint_parameter_travel_over_3mm=int(((travel>3)&active).sum()),positive_count=int((p>=.5).sum()),
                all_rows_endpoint_parameter_travel_max_mm=float(travel.max(initial=0)),patch_sha256=digest(folder/f'{name}_patch.npy')))
            a._mmap.close();check_resources()
        original._mmap.close();results.append(case);write(dest/'progress.json',dict(cases=results))
        print('FROZEN_STREAM_RIGID_CASE',key,case['transforms'],flush=True);del original,variants,spans,lengths;gc.collect()
    write(dest/'worker_receipt.json',dict(status='REMAINING_FROZEN_RIGID_OUTPUT_AUDIT_COMPLETE',cases=results,fit_calls=0,GT_reads=0,STEP_reads=0,
        model_sha256=digest(path),seconds=time.perf_counter()-start,new_extraction_score=False,
        whole_pipeline_equivariance_proven=False,scope='frozen candidate features; recomputed local geometry and centroids only'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');args=ap.parse_args();dest=OUT/'run339'
    if args.worker:
        try:worker(dest)
        except BaseException:write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise
        return
    assert read(OUT/'run338/failure.json')['watchdog']['remaining_job_processes_after_wait']==0
    check_resources();dest.mkdir(exist_ok=False);snap=dest/'source_snapshot';snap.mkdir()
    for name in ('sjg_relative_output_stream_audit_1145.py','sjg_relative_geometry_batch_1136.py','sjg_observed_face_features_1138.py'):
        shutil.copy2(ROOT/'weld_pipeline'/name,snap)
    write(dest/'registered_protocol.json',dict(status='REMAINING_CASES_STREAMED_FROZEN_DIAGNOSTIC',sources=['10','64'],
        prior_failure_preserved='run338',prior_verified=['21','BM79A_native_local'],fit_calls=0,wall_seconds=180,
        sparse_patch_columns=31,prediction_block_rows=4096,no_model_or_threshold_changes=True,no_DEV_labels=True,utc=utc()))
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    result=supervise([str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_relative_output_stream_audit_1145','--worker'],dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='FROZEN_STREAM_AUDIT_INCOMPLETE',watchdog=result));raise RuntimeError('Frozen audit incomplete')
    r=read(dest/'worker_receipt.json');r['external_wall_seconds']=result['wall_seconds'];write(dest/'receipt.json',r)
    print('FROZEN_STREAM_OUTPUT_AUDIT_COMPLETE',result['wall_seconds'],flush=True)


if __name__=='__main__':main()
