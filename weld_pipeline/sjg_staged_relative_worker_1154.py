"""One stage per process for the fixed Pro345 seven-case model."""
import argparse,gc,pickle,platform,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS,feature_path

SETTINGS=dict(n_estimators=64,max_depth=32,min_samples_leaf=1,max_features='sqrt',bootstrap=False,
    n_jobs=1,random_state=1031,warm_start=False,verbose=2)


def save_component(dest,name,model,bindings):
    import sklearn
    assert len(model.estimators_)==64
    state=forest_state_sha256(model);path=dest/(name+'.pkl')
    atomic_binary_write(path,lambda f:pickle.dump(model,f,protocol=5))
    write(dest/(name+'_saved.json'),dict(status='COMPONENT_SAVED_PENDING_INDEPENDENT_VERIFY',name=name,
        component_sha256=digest(path),numeric_state_sha256=state,trees=64,input_bindings=bindings,
        parameters=model.get_params(deep=False),numpy=np.__version__,sklearn=sklearn.__version__,python=platform.python_version(),utc=utc()))


def load_component(dest,name,verified=True):
    import sklearn
    r=read(dest/(name+('_verified.json' if verified else '_saved.json')))
    assert r['status']==('COMPONENT_VERIFIED' if verified else 'COMPONENT_SAVED_PENDING_INDEPENDENT_VERIFY')
    assert r['numpy']==np.__version__ and r['sklearn']==sklearn.__version__ and r['python']==platform.python_version()
    path=dest/(name+'.pkl');assert digest(path)==r['component_sha256']
    with path.open('rb') as f:model=pickle.load(f)
    assert len(model.estimators_)==64 and forest_state_sha256(model)==r['numeric_state_sha256']
    return model,r


def layout(dest):
    source=OUT/'run345';bindings=read(source/'saved_tables_exact_control.json');assert bindings['status']=='ALL_SEVEN_SAVED_TABLES_EXACT'
    for name,r in bindings['files'].items():assert digest(source/(name+'.bin'))==r['sha256']
    output={}
    for name,shape in [('object_x',(1134182,152)),('endpoint_x',(13533,153))]:
        x=np.memmap(source/(name+'.bin'),dtype='<f4',mode='r',shape=shape)
        path=dest/(name+'_F.npy');y=np.lib.format.open_memmap(path,mode='w+',dtype='<f4',shape=shape,fortran_order=True)
        # Bound transient copies; no second whole-table ndarray.
        for first in range(0,shape[0],4096):y[first:first+4096]=x[first:first+4096]
        y.flush();y._mmap.close();y=np.load(path,mmap_mode='r')
        assert y.flags.f_contiguous and y.dtype==x.dtype and y.shape==x.shape
        for first in range(0,shape[0],4096):assert np.array_equal(x[first:first+4096],y[first:first+4096])
        x._mmap.close();y._mmap.close();del x,y;gc.collect();check_resources()
        output[name]=dict(file=path.name,sha256=digest(path),logical_equal=True,shape=list(shape),dtype='<f4')
    write(dest/'layout.json',dict(status='F_ORDER_LOGICAL_PARITY_PASSED',features=output,original=bindings,
        labels_and_weights_reused_without_changes=True,real_data_fit_calls=0))


def software_control(dest):
    from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
    from weld_pipeline.sjg_finite_layout_control_1157 import verify_finite_layout
    rng=np.random.default_rng(1031);x=rng.normal(size=(384,152)).astype(np.float32);w=rng.uniform(.1,1.5,384)
    y=((x[:,1]>0).astype(int)+2*(x[:,7]>.2)).astype(np.int16);q=np.c_[x[:,3],x[:,5]]
    result={}
    for name,cls,target in [('object',ExtraTreesClassifier,y),('endpoint',ExtraTreesRegressor,q)]:
        data=x if name=='object' else np.column_stack([x,np.tile(np.arange(1,13)/12,32)]).astype(np.float32)
        params=dict(SETTINGS,verbose=0);a=cls(**params).fit(np.array(data,order='C'),target,sample_weight=w)
        b=cls(**params).fit(np.array(data,order='F'),target,sample_weight=w)
        result[name]=dict(verify_finite_layout(a,b,data),numeric_state_sha256=forest_state_sha256(a))
        del a,b;gc.collect()
    write(dest/'layout_software_control.json',dict(status='SYNTHETIC_C_F_FOREST_PARITY_PASSED',controls=result,
        synthetic_fit_calls=4,real_data_fit_calls=0,no_padded_struct_or_pickle_comparison=True))


def fit_component(dest,name):
    from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
    layout_receipt=read(dest/'layout.json');assert read(dest/'layout_software_control.json')['status']=='SYNTHETIC_C_F_FOREST_PARITY_PASSED'
    source=OUT/'run345';count=1134182 if name=='object' else 13533
    fx=dest/layout_receipt['features'][name+'_x']['file'];assert digest(fx)==layout_receipt['features'][name+'_x']['sha256']
    x=np.load(fx,mmap_mode='r');yf=source/(name+'_y.bin');wf=source/(name+'_w.bin')
    for key,p in ((name+'_y',yf),(name+'_w',wf)):assert digest(p)==layout_receipt['original']['files'][key]['sha256']
    y=np.memmap(yf,dtype='<i2' if name=='object' else '<f8',mode='r',shape=(count,) if name=='object' else (count,2))
    w=np.memmap(wf,dtype='<f8',mode='r',shape=(count,))
    for first in range(0,count,4096):
        assert np.isfinite(x[first:first+4096]).all() and np.isfinite(y[first:first+4096]).all() and np.isfinite(w[first:first+4096]).all()
    cls=ExtraTreesClassifier if name=='object' else ExtraTreesRegressor
    print('STAGED_FIT',name,'ROWS',count,'TREES',64,flush=True)
    model=cls(**SETTINGS).fit(x,y,sample_weight=w)
    for a in (x,y,w):a._mmap.close()
    del x,y,w;gc.collect();check_resources()
    save_component(dest,name,model,dict(layout_sha256=digest(dest/'layout.json'),feature_sha256=digest(fx),label_sha256=digest(yf),weight_sha256=digest(wf),row_order='original seven case then original known row order'))


def connection_tables(dest):
    from weld_pipeline.sjg_staged_connection_1155 import generate
    obj,ob=load_component(dest,'object');end,eb=load_component(dest,'endpoint');obj.verbose=0;end.verbose=0
    models=dict(object=obj,endpoint=end,K=12)
    def phase(name,**extra):print('STAGED_CONNECTION',name,extra,flush=True)
    total,bindings=generate(dest,models,phase)
    write(dest/'connection_tables.json',dict(status='COMPLETE_CONNECTION_TABLES_FROZEN',rows=total,cases=bindings,
        object_component_sha256=ob['component_sha256'],endpoint_component_sha256=eb['component_sha256'],
        features_sha256=digest(dest/'link_train.f32'),labels_sha256=digest(dest/'link_labels.i8'),GT_endpoint_overrides=0))


def fit_connection(dest):
    from sklearn.ensemble import ExtraTreesClassifier
    r=read(dest/'connection_tables.json');assert r['status']=='COMPLETE_CONNECTION_TABLES_FROZEN'
    assert digest(dest/'link_train.f32')==r['features_sha256'] and digest(dest/'link_labels.i8')==r['labels_sha256']
    x=np.memmap(dest/'link_train.f32',dtype='<f4',mode='r',shape=(r['rows'],32));y=np.memmap(dest/'link_labels.i8',dtype='i1',mode='r',shape=(r['rows'],))
    for first in range(0,r['rows'],8192):assert np.isfinite(x[first:first+8192]).all()
    assert np.any(y==0);model=ExtraTreesClassifier(**dict(SETTINGS,max_depth=16)).fit(x,y)
    x._mmap.close();y._mmap.close();del x,y;gc.collect();check_resources()
    save_component(dest,'connection',model,dict(connection_tables_sha256=digest(dest/'connection_tables.json'),
        object_component_sha256=r['object_component_sha256'],endpoint_component_sha256=r['endpoint_component_sha256']))


def package(dest):
    models={};states={};components={}
    for name in ('object','endpoint','connection'):
        models[name],r=load_component(dest,name);states[name]=r['numeric_state_sha256'];components[name]=r['component_sha256']
    links=read(dest/'connection_tables.json')
    assert links['object_component_sha256']==components['object'] and links['endpoint_component_sha256']==components['endpoint']
    models.update(K=12,feature_columns=152,feature_run='run329',feature_runs={key:feature_path(key).parent.name for key in (*KEYS,'64')},
        model_family='RELATIVE_COMPATIBLE_K12_FORESTS',training_sources=list(KEYS),supervision_run='run318',compatible_run='run317',input_request_run='run312')
    atomic_binary_write(dest/'model.pkl',lambda f:pickle.dump(models,f,protocol=5))
    write(dest/'package_saved.json',dict(status='PACKAGE_SAVED_PENDING_VERIFY',model_sha256=digest(dest/'model.pkl'),component_state_sha256=states,
        component_file_sha256=components,cases=links['cases'],forest_fit_calls=3,G=363,N=1134182,E=13533,K=12,connection_rows=links['rows']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True);args=ap.parse_args()
    dest=OUT/args.run;start=time.perf_counter()
    try:
        check_resources();stage=args.stage
        if stage=='layout':layout(dest)
        elif stage=='software_control':software_control(dest)
        elif stage in ('object','endpoint'):fit_component(dest,stage)
        elif stage=='connection_tables':connection_tables(dest)
        elif stage=='connection':fit_connection(dest)
        elif stage.startswith('verify_') and stage!='verify_package':
            name=stage[7:];model,r=load_component(dest,name,verified=False)
            write(dest/(name+'_verified.json'),dict(r,status='COMPONENT_VERIFIED',verification_process_independent=True))
        elif stage=='package':package(dest)
        elif stage=='verify_package':
            r=read(dest/'package_saved.json');assert digest(dest/'model.pkl')==r['model_sha256']
            with (dest/'model.pkl').open('rb') as f:models=pickle.load(f)
            assert r['component_state_sha256']=={k:forest_state_sha256(models[k]) for k in r['component_state_sha256']}
            write(dest/'package_verified.json',dict(r,status='PACKAGE_VERIFIED'))
        else:raise ValueError(stage)
        write(dest/('stage_'+stage+'_complete.json'),dict(status='STAGE_COMPLETE',stage=stage,seconds=time.perf_counter()-start,utc=utc()))
    except BaseException:
        write(dest/('stage_'+args.stage+'_failure.json'),dict(error=traceback.format_exc(),stage=args.stage,seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
