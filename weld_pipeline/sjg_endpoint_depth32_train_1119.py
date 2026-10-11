"""Pro301: one depth32 endpoint refit; fixed object and connection forests."""
import argparse,hashlib,importlib.util,pickle,sys,time,traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run302');args=ap.parse_args()
    dest=OUT/args.run;start=time.perf_counter()
    def phase(name,**extra):
        write(dest/'worker_phase.json',dict(phase=name,seconds=time.perf_counter()-start,utc=utc(),**extra))
        print('ENDPOINT_DEPTH32_PHASE',name,'SECONDS',round(time.perf_counter()-start,3),extra,flush=True)
    try:
        phase('IMPORTING_CPU_LIBRARIES')
        import numpy as np
        from sklearn.ensemble import ExtraTreesRegressor
        from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
        receipt=read(OUT/'run298/receipt.json');base=OUT/'run298/model.pkl'
        assert digest(base)==receipt['model_sha256']
        with base.open('rb') as f:payload=pickle.load(f)
        frozen={k:forest_state_sha256(payload[k]) for k in ('object','connection')}
        old=payload['endpoint'].get_params(deep=False);new=dict(old,max_depth=32,warm_start=False)
        delta={k:[old[k],new[k]] for k in old if old[k]!=new[k]};assert delta=={'max_depth':[16,32]}
        loader=OUT/'run289/source_snapshot/sjg_joint_forest_train_1109.py'
        assert digest(loader)==read(OUT/'run289/registered_protocol.json')['code_bindings'][loader.name]
        spec=importlib.util.spec_from_file_location('_original_endpoint_loader',loader)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cases=[];xs=[];ys=[]
        for key in ('21','BM79A_native_local','10'):
            phase('LOADING_ORIGINAL_POSITIVE_ROWS',source=key)
            case=module.load_case(key,supervision=True);prior=next(c for c in receipt['cases'] if c['source']==key)
            for field in ('feature_receipt_sha256','alias_map_sha256','geometry_sha256','compiled_target_sha256'):
                assert case[field]==prior[field]
            positive=case['y']!=0;xs.append(case['x'][positive]);ys.append(case['q'][positive]);cases.append(case)
            check_resources()
        x=np.concatenate(xs);q=np.concatenate(ys);assert x.shape==(782,128) and q.shape==(782,2)
        phase('FIT_ONE_ENDPOINT_FOREST',positive_rows=782,trees=64,depth=32)
        payload['endpoint']=ExtraTreesRegressor(**new).fit(x,q)
        payload.update(base_model_run='run298',endpoint_settings=new,frozen_component_sha256=frozen,
            single_changed_parameter={'endpoint.max_depth':[16,32]})
        assert {k:forest_state_sha256(payload[k]) for k in frozen}==frozen
        check_resources();phase('ATOMIC_SAVE_AND_NUMERIC_RELOAD_VERIFY')
        atomic_binary_write(dest/'model.pkl',lambda f:pickle.dump(payload,f,protocol=5))
        with (dest/'model.pkl').open('rb') as f:restored=pickle.load(f)
        assert {k:forest_state_sha256(restored[k]) for k in frozen}==frozen
        assert forest_state_sha256(restored['endpoint'])==forest_state_sha256(payload['endpoint'])
        assert digest(base)==receipt['model_sha256'] and 'torch' not in sys.modules
        write(dest/'worker_receipt.json',dict(status='ONE_ENDPOINT_DEPTH32_FOREST_FIT_OTHERS_FROZEN',
            model_sha256=digest(dest/'model.pkl'),base_model_run='run298',base_model_sha256=receipt['model_sha256'],
            frozen_component_sha256=frozen,frozen_component_hash_format='EXACT_FOREST_PREDICTIVE_STATE_v1',
            frozen_components_verified_after_save_reload=True,forest_fit_calls=1,object_fit_calls=0,connection_fit_calls=0,
            endpoint_fit_calls=1,endpoint_settings=new,original_loader_sha256=digest(loader),
            positive_feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),positive_ratio_label_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
            positive_rows=782,G=213,cases=receipt['cases'],fixed_presence_threshold=.5,all_positive_nodes_fixed_selected=True,
            seconds=time.perf_counter()-start,Torch_imported=False,extraction_correct_rate=None,extraction_precision_mm=None,utc=utc()))
        phase('COMPLETE')
    except BaseException:
        write(dest/'worker_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start,utc=utc()));raise


if __name__=='__main__':main()
