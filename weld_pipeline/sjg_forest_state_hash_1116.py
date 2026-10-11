"""Stable exact predictive-state digest; structured dtype padding is excluded.

Pickled sklearn Tree node arrays contain C-structure padding, which is not a
model weight and need not survive deserialization. Every named field, numeric
bit pattern, shape, estimator parameter and class mapping is hashed instead.
"""
import hashlib
import json
import numpy as np


def forest_state_manifest(model):
    def primitive(value):
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,np.ndarray):return array(value)
        if isinstance(value,dict):return {str(k):primitive(v) for k,v in sorted(value.items())}
        if isinstance(value,(list,tuple)):return [primitive(v) for v in value]
        if value is None or isinstance(value,(str,bool,int,float)):return value
        raise TypeError('Unregistered predictive-state value: '+str(type(value)))
    def array(value):
        value=np.asarray(value)
        if value.dtype.names:
            return dict(shape=list(value.shape),fields={name:array(value[name]) for name in value.dtype.names})
        if value.dtype.hasobject:raise TypeError('Object dtype needs an explicit semantic encoding')
        return dict(shape=list(value.shape),dtype=value.dtype.str,
                    bytes_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
    def estimator_fields(estimator):
        result=dict(type=type(estimator).__module__+'.'+type(estimator).__name__,parameters=primitive(estimator.get_params(deep=False)))
        for name in ('n_features_in_','n_outputs_','n_classes_','classes_','max_features_','feature_names_in_'):
            if hasattr(estimator,name):result[name]=primitive(getattr(estimator,name))
        return result
    result=estimator_fields(model);result['version']='EXACT_FOREST_PREDICTIVE_STATE_v1'
    result['estimators']=[]
    for estimator in model.estimators_:
        state=estimator_fields(estimator);tree=estimator.tree_
        state['tree_state']=primitive(tree.__getstate__())
        state['tree_dimensions']=dict(n_features=int(tree.n_features),n_outputs=int(tree.n_outputs),n_classes=array(tree.n_classes))
        result['estimators'].append(state)
    return result


def forest_state_sha256(model):
    state=forest_state_manifest(model)
    return hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
