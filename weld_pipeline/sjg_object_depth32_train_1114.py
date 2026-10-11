"""Pro291: fit only a new depth-32 object forest; freeze the other two."""
import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
import traceback
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc,check_resources


def forest_sha(model):
    from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256
    return forest_state_sha256(model)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run294');args=ap.parse_args()
    dest=OUT/args.run;start=time.perf_counter()
    def phase(name,**extra):
        write(dest/'worker_phase.json',dict(phase=name,seconds=time.perf_counter()-start,utc=utc(),**extra))
        print('CPU_DEPTH32_PHASE',name,'SECONDS',round(time.perf_counter()-start,3),extra,flush=True)
    try:
        phase('IMPORTING_CPU_LIBRARIES')
        import faulthandler
        faulthandler.enable();faulthandler.dump_traceback_later(90,repeat=True)
        import numpy as np
        from sklearn.ensemble import ExtraTreesClassifier
        from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write
        assert 'torch' not in sys.modules
        original_receipt=read(OUT/'run289/receipt.json');original_path=OUT/'run289/model.pkl'
        assert digest(original_path)==original_receipt['model_sha256']
        with original_path.open('rb') as f:payload=pickle.load(f)
        frozen={name:forest_sha(payload[name]) for name in ('endpoint','connection')}
        parameters=payload['object'].get_params(deep=False);assert parameters['max_depth']==16 and not parameters['warm_start']
        new_parameters=dict(parameters,max_depth=32,warm_start=False)
        differences={k:[parameters[k],new_parameters[k]] for k in parameters if parameters[k]!=new_parameters[k]}
        assert differences=={'max_depth':[16,32]}
        # Use the original archived loader, preserving row order and supervision
        # construction; this avoids a changed DEV route touching TRAIN tables.
        loader_path=OUT/'run289/source_snapshot/sjg_joint_forest_train_1109.py'
        protocol=read(OUT/'run289/registered_protocol.json')
        assert digest(loader_path)==protocol['code_bindings'][loader_path.name]
        spec=importlib.util.spec_from_file_location('_run289_frozen_training_loader',loader_path)
        old_loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_loader)
        phase('LOADING_ORIGINAL_COMPLETE_OBJECT_TABLES');tables=[];labels=[];bindings=[]
        for key in ('21','BM79A_native_local','10'):
            case=old_loader.load_case(key,supervision=True)
            previous=next(c for c in original_receipt['cases'] if c['source']==key)
            for field in ('feature_receipt_sha256','alias_map_sha256','geometry_sha256','compiled_target_sha256'):
                assert case[field]==previous[field]
            tables.append(case['x']);labels.append(case['y']);bindings.append(dict(previous,
                complete_object_rows=len(case['x']),positive_responsibilities=len(case['teacher'])))
            print('ORIGINAL_OBJECT_TABLE',key,len(case['x']),len(case['teacher']),flush=True)
            del case;check_resources()
        x=np.concatenate(tables);y=np.concatenate(labels);del tables,labels
        assert x.shape==(336344,128) and int((y!=0).sum())==782
        xsha=hashlib.sha256(x.tobytes()).hexdigest();ysha=hashlib.sha256(y.tobytes()).hexdigest()
        phase('FIT_ONE_NEW_OBJECT_FOREST_DEPTH32',rows=len(x),trees=64)
        new_object=ExtraTreesClassifier(**new_parameters).fit(x,y)
        payload['object']=new_object;payload['object_settings']=new_parameters
        payload['frozen_component_sha256']=frozen;payload['base_model_run']='run289'
        payload['single_changed_parameter']=differences;payload['bindings']=bindings
        assert {name:forest_sha(payload[name]) for name in frozen}==frozen
        check_resources();phase('ATOMIC_MODEL_SAVE')
        atomic_binary_write(dest/'model.pkl',lambda f:pickle.dump(payload,f,protocol=5))
        with (dest/'model.pkl').open('rb') as f:restored=pickle.load(f)
        assert {name:forest_sha(restored[name]) for name in frozen}==frozen
        assert digest(original_path)==original_receipt['model_sha256']
        assert 'torch' not in sys.modules
        write(dest/'worker_receipt.json',dict(status='ONE_DEPTH32_OBJECT_FOREST_FIT_OTHER_TWO_FROZEN',
            model_sha256=digest(dest/'model.pkl'),base_model_run='run289',base_model_sha256=original_receipt['model_sha256'],
            object_settings=new_parameters,single_changed_parameter=differences,frozen_component_sha256=frozen,
            frozen_component_hash_format='EXACT_FOREST_PREDICTIVE_STATE_v1',
            frozen_components_verified_after_save_reload=True,forest_fit_calls=1,endpoint_fit_calls=0,connection_fit_calls=0,
            original_loader_sha256=digest(loader_path),object_table_sha256=xsha,object_labels_sha256=ysha,
            complete_object_rows=len(x),positive_responsibilities=int((y!=0).sum()),G=213,cases=bindings,
            fixed_presence_threshold=.5,all_positive_nodes_fixed_selected=True,Torch_imported=False,
            seconds=time.perf_counter()-start,extraction_correct_rate=None,extraction_precision_mm=None,utc=utc()))
        phase('COMPLETE');faulthandler.cancel_dump_traceback_later()
    except BaseException:
        write(dest/'worker_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start,utc=utc()));raise


if __name__=='__main__':main()
