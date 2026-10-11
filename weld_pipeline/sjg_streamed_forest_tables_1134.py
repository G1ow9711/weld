"""Prepare original seven-case forest tables without retaining graphs or duplicate X arrays.

This is data IO only. No fit, label change, feature change, or new trial authorization.
"""
import argparse
import gc
import hashlib
import importlib.util
import shutil
import time
import traceback
from contextlib import ExitStack
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT, OUT, read, write, digest, utc, check_resources
from weld_pipeline.sjg_joint_object_features_1106 import side_rows

KEYS=('21','4','10','25','36','42','BM79A_native_local')
SCHEMAS={'object_x':('<f4',128),'object_y':('<i2',1),'object_w':('<f8',1),
         'endpoint_x':('<f4',129),'endpoint_y':('<f8',2),'endpoint_w':('<f8',1)}


def inputs(key):
    feature_run='run308' if key in ('4','25','36','42') else 'run287'
    return dict(features=OUT/feature_run/f'source_{key}/object_features.npy',
                feature_receipt=OUT/feature_run/f'source_{key}/receipt.json',
                aliases=OUT/f'run288/source_{key}_alias_map.npz',
                labels=OUT/f'run318/source_{key}_count_targets.npz',
                instances=OUT/f'run318/source_{key}_interval_instances.json')


def prepare_tables(dest):
    """All original known rows and ordinal instances, same case/row order as 1130."""
    dest.mkdir(exist_ok=False)
    target=read(OUT/'run318/receipt.json')
    assert target['status']=='ORDERED_INTERVAL_SUPERVISION_READY'
    K=int(target['K']);bindings=[];offsets={'object':0,'endpoint':0};max_private=0
    aliases=read(OUT/'run288/all_alias_maps_frozen_before_labels.json')['cases']
    with ExitStack() as stack:
        streams={name:stack.enter_context((dest/(name+'.bin')).open('xb')) for name in SCHEMAS}
        for key in KEYS:
            source=inputs(key);bound={name:digest(p) for name,p in source.items()}
            assert bound['features']==read(source['feature_receipt'])['features_sha256']
            assert bound['aliases']==next(a['map_sha256'] for a in aliases if a['source']==key)
            with np.load(source['aliases']) as z:canonical=z['canonical_fragments']
            with np.load(source['labels']) as z:y=z['labels'];w=z['weights']
            assert len(y)==len(w)==2*len(canonical)
            known=np.flatnonzero(y>=0)
            instances=[e for e in read(source['instances'])['instances'] if e['count_is_known']]
            base=np.load(source['features'],mmap_mode='r')
            before=offsets.copy()
            for first in range(0,len(known),8192):
                rows=known[first:first+8192]
                x=side_rows(base,canonical[rows//2],rows%2)
                assert np.isfinite(x).all()
                streams['object_x'].write(x.astype('<f4',copy=False).tobytes())
                assert np.array_equal(y[rows],y[rows].astype('<i2'))
                streams['object_y'].write(y[rows].astype('<i2').tobytes())
                streams['object_w'].write(w[rows].astype('<f8').tobytes())
                max_private=max(max_private,check_resources()['private_commit_bytes'])
            for first in range(0,len(instances),8192):
                part=instances[first:first+8192];rows=np.array([e['row'] for e in part],np.int64)
                x=np.empty((len(part),129),np.float32)
                x[:,:128]=side_rows(base,canonical[rows//2],rows%2)
                x[:,128]=np.array([e['k'] for e in part])/K
                streams['endpoint_x'].write(x.astype('<f4',copy=False).tobytes())
                streams['endpoint_y'].write(np.array([e['ratio'] for e in part],dtype='<f8').tobytes())
                streams['endpoint_w'].write(np.array([e['weight'] for e in part],dtype='<f8').tobytes())
                max_private=max(max_private,check_resources()['private_commit_bytes'])
            offsets['object']+=len(known);offsets['endpoint']+=len(instances)
            bindings.append(dict(source=key,sha256=bound,object_start=before['object'],object_stop=offsets['object'],
                endpoint_start=before['endpoint'],endpoint_stop=offsets['endpoint'],
                all_input_rows=len(y),known_object_rows=len(known),masked_unknown_rows=int((y<0).sum()),
                endpoint_instances=len(instances)))
            base._mmap.close()
            del base,canonical,y,w,known,instances,x
            gc.collect()
            print('STREAM_TABLE_SOURCE',key,offsets,flush=True)
    assert offsets['endpoint']==target['E']
    files={}
    for name,(dtype,columns) in SCHEMAS.items():
        count=offsets['endpoint' if name.startswith('endpoint') else 'object']
        p=dest/(name+'.bin')
        assert p.stat().st_size==count*columns*np.dtype(dtype).itemsize
        files[name]=dict(file=p.name,dtype=dtype,shape=[count,columns] if columns>1 else [count],sha256=digest(p))
    receipt=dict(status='STREAMED_ORIGINAL_TABLES_PREPARED',files=files,cases=bindings,K=K,
        object_rows=offsets['object'],endpoint_rows=offsets['endpoint'],TRAIN_G=target['G'],
        features_changed=False,labels_changed=False,weights_changed=False,subsampling=False,
        fit_calls=0,DEV_reads=0,STEP_reads=0,maximum_observed_preparation_private_bytes=max_private)
    write(dest/'receipt.json',receipt)
    return receipt


def verify_tables(dest):
    """Compare every streamed row against the old eager construction, one case at a time."""
    table=read(dest/'receipt.json')
    frozen=OUT/'run319/source_snapshot/sjg_joint_object_features_1106.py'
    protocol=read(OUT/'run319/registered_protocol.json')
    assert digest(frozen)==protocol['code_bindings'][frozen.name]
    spec=importlib.util.spec_from_file_location('_original_1106_stream_control',frozen)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    checks=[];max_private=0
    with ExitStack() as stack:
        streams={name:stack.enter_context((dest/(name+'.bin')).open('rb')) for name in SCHEMAS}
        for binding in table['cases']:
            key=binding['source'];source=inputs(key)
            assert binding['sha256']=={name:digest(p) for name,p in source.items()}
            with np.load(source['aliases']) as z:canonical=z['canonical_fragments']
            with np.load(source['labels']) as z:y=z['labels'];w=z['weights']
            instances=[e for e in read(source['instances'])['instances'] if e['count_is_known']]
            base=np.load(source['features'],mmap_mode='r')
            # Exactly the former loader's construction, without unrelated Python geometry objects.
            x=np.empty((2*len(canonical),128),np.float32)
            x[0::2]=base[canonical];x[1::2]=old.side_rows(base,canonical,np.ones(len(canonical),np.int8))
            known=np.flatnonzero(y>=0)
            def compare(name,expected):
                dtype,columns=SCHEMAS[name]
                observed=np.fromfile(streams[name],dtype=dtype,count=expected.size).reshape(expected.shape)
                assert np.array_equal(observed,expected),f'Exact table mismatch: {key}/{name}'
            for first in range(0,len(known),8192):
                rows=known[first:first+8192]
                compare('object_x',x[rows]);compare('object_y',y[rows]);compare('object_w',w[rows])
            ex=np.empty((len(instances),129),np.float32)
            ex[:,:128]=x[[e['row'] for e in instances]];ex[:,128]=np.array([e['k'] for e in instances])/table['K']
            compare('endpoint_x',ex)
            compare('endpoint_y',np.asarray([e['ratio'] for e in instances]))
            compare('endpoint_w',np.asarray([e['weight'] for e in instances]))
            max_private=max(max_private,check_resources()['private_commit_bytes'])
            checks.append(dict(source=key,object_rows=len(known),endpoint_rows=len(instances),all_six_arrays_exact=True))
            base._mmap.close();del base,x,ex,canonical,y,w,known,instances
            gc.collect()
            print('EXACT_OLD_TABLE_PARITY',key,flush=True)
        assert all(not stream.read(1) for stream in streams.values())
    assert all(digest(dest/v['file'])==v['sha256'] for v in table['files'].values())
    control=dict(status='EXACT_OLD_TABLE_PARITY_PASSED',cases=checks,fit_calls=0,
        maximum_observed_control_private_bytes=max_private,
        reference_feature_code_sha256=digest(frozen),covers_all_rows_features_labels_weights_and_order=True)
    write(dest/'parity_control.json',control)
    return control


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run325');ap.add_argument('--worker',action='store_true')
    args=ap.parse_args();dest=OUT/args.run
    if args.worker:
        started=time.perf_counter()
        try:
            prepare_start=time.perf_counter();table=prepare_tables(dest/'tables');prepare_seconds=time.perf_counter()-prepare_start
            verify_start=time.perf_counter();control=verify_tables(dest/'tables');verify_seconds=time.perf_counter()-verify_start
            write(dest/'worker_receipt.json',dict(status='STREAMED_TABLE_IO_VALIDATED',
                object_rows=table['object_rows'],endpoint_rows=table['endpoint_rows'],
                maximum_observed_preparation_private_bytes=table['maximum_observed_preparation_private_bytes'],
                maximum_observed_control_private_bytes=control['maximum_observed_control_private_bytes'],
                prepare_seconds=prepare_seconds,parity_seconds=verify_seconds,seconds=time.perf_counter()-started,
                fit_calls=0,new_model_created=False,extraction_correct_rate=None,extraction_precision_mm=None,
                full_contract_pass=False,training_budget_exemption=False,training_pipeline_integrated=False))
        except BaseException:
            write(dest/'worker_failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise
        return
    from weld_pipeline.sjg_owned_watchdog_1101 import supervise
    dest.mkdir(exist_ok=False);snapshot=dest/'source_snapshot';snapshot.mkdir()
    for p in (ROOT/'weld_pipeline/sjg_streamed_forest_tables_1134.py',ROOT/'weld_pipeline/sjg_joint_object_features_1106.py'):
        shutil.copy2(p,snapshot/p.name)
    write(dest/'registered_protocol.json',dict(status='ZERO_FIT_IO_CONTROL_REGISTERED',
        TRAIN=list(KEYS),original_supervision='run318',G=363,original_feature_dimension=128,
        known_rows_all=True,full_exact_comparison_to_old_construction=True,fit_calls=0,wall_seconds=180,utc=utc()))
    command=[str(ROOT/'tools/python/cpython-3.10-windows-x86_64-none/python.exe'),'-u','-m',
        'weld_pipeline.sjg_streamed_forest_tables_1134','--run',args.run,'--worker']
    result=supervise(command,dest,180.)
    if result['termination_reason'] or result['exit_code']!=0:
        write(dest/'failure.json',dict(status='STREAMED_TABLE_IO_INCOMPLETE',watchdog=result));raise RuntimeError('IO control incomplete')
    receipt=read(dest/'worker_receipt.json');receipt['external_wall_seconds']=result['wall_seconds']
    write(dest/'receipt.json',receipt)
    print('STREAMED_TABLE_IO_VALIDATED',receipt,flush=True)


if __name__=='__main__':main()
