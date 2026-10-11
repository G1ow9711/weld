"""Pro364: necessary nearest pooling and full frozen-prefix migration checks."""
import gc
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources
from weld_pipeline.sjg_parent_set_scorer_1159 import ParentSetScorer,AdamState,SHAPES
from weld_pipeline.sjg_selector_resume_state_1178 import restore,reuse_columns
from weld_pipeline.sjg_nearest_parent_scorer_1185 import migrate,expand,old_coordinates
from weld_pipeline.sjg_nearest_parent_features_1184 import NearestParentRelations
from weld_pipeline.sjg_relation_column_cache_1173 import RelationColumnCache,CachedParentRelations
from weld_pipeline.sjg_selector_trial_worker_1174 import bindings
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS


def initial(source):
    training=read(source/'selector_training.json');assert training['updates']==224
    assert digest(source/'selector.npz')==training['selector_sha256']
    assert digest(source/'selector_normalization.npz')==training['normalization_sha256']
    with np.load(source/'selector_normalization.npz') as z:mean=z['mean'].copy();scale=z['scale'].copy()
    old=ParentSetScorer(1031);adam=AdamState(old)
    restored=restore(source,old,adam,mean,scale,training['bindings'],allow_completed_stage=True)
    loaded=ParentSetScorer.load(source/'selector.npz')
    assert all(np.array_equal(old.parameters[k],loaded.parameters[k]) for k in SHAPES)
    model,optimizer=migrate(old,adam)
    for key in SHAPES:
        for a,b in ((old.parameters,model.parameters),(adam.first,optimizer.first),(adam.second,optimizer.second)):
            value=old_coordinates(b[key]) if key=='W1' else b[key]
            assert np.array_equal(a[key],value)
            if key=='W1':assert np.count_nonzero(b[key][192:208])==0
    assert optimizer.step_count==224
    return old,adam,model,optimizer,mean,scale,restored


def pooling_control():
    context=NearestParentRelations.__new__(NearestParentRelations)
    context.parents=[None]*4;context.selected=[]
    context.total=np.zeros((4,16));context.maximum=np.zeros((4,16))
    context.nearest=np.zeros((4,16));context.nearest_distance=np.full(4,np.inf)
    context.nearest_index=np.full(4,-1,dtype=np.int64)
    assert np.array_equal(context.features(),np.zeros((4,48)))
    columns={}
    for j,triplet in ((3,[1,4,2]),(2,[3,2,3]),(1,[2,3,3])):
        col=np.full((4,16),float(j));col[:,:3]=triplet;columns[j]=col
        context.add_column(j,col)
        for i in set(range(4))-set(context.selected):
            stable=sorted(context.selected)
            winner=min(stable,key=lambda index:(columns[index][i,:3].max(),index))
            assert context.nearest_index[i]==winner
            assert np.array_equal(context.nearest[i],columns[winner][i])
    assert context.nearest_index[0]==1
    assert not np.array_equal(context.nearest[0],np.minimum.reduce([v[0] for v in columns.values()]))
    bad=np.zeros((4,16));bad[0,0]=np.nan
    try:context.add_column(0,bad)
    except ValueError:pass
    else:raise AssertionError('NaN accepted')
    assert len(context.parents)==4 and context.selected==[3,2,1]
    return dict(empty_single_tie_and_coherent_whole_row=True,original_index_tie_break=True,
        nonfinite_rejected=True,candidate_count_unchanged=True)


def control(dest):
    source=OUT/'run364'
    old,adam,model,optimizer,mean,scale,restored=initial(source)
    pooling=pooling_control()
    cache=RelationColumnCache(dest/'control_relation_cache',bindings(dest,KEYS))
    reuse=reuse_columns(source/'training_relation_cache',cache);records=[]
    try:
        for key in ('21','BM79A_native_local','10'):
            folder=source/f'test_{key}';frozen=read(folder/'final_prediction.json')['selected_candidate_indices']
            u=(np.load(folder/'unary.npy')-mean)/scale
            parents,points=load_parents(folder);metadata=read(folder/'parent_metadata.json')
            a=CachedParentRelations(parents,metadata,key,cache)
            b=NearestParentRelations(parents,metadata,key,cache)
            for step in range(len(frozen)+1):
                check_resources();ra=a.features();rb=b.features()
                assert np.array_equal(ra,rb[:,:32])
                sa,_=old.forward(u,ra,a.selected);sb,forward=model.forward(u,rb,b.selected)
                assert np.array_equal(sa,sb), 'Initial score parity must be bitwise'
                ca=old.choose_next(sa,a.selected);cb=model.choose_next(sb,b.selected)
                assert ca==cb==(frozen[step] if step<len(frozen) else None)
                if ca is None:break
                a.add(ca);b.add(cb)
            records.append(dict(source=key,prefixes=len(frozen)+1,all_logits_bitwise_equal=True,
                all_argmax_and_STOP_exact=True,prediction_sha256=digest(folder/'final_prediction.json')))
            del a,b,parents;points._mmap.close();gc.collect()
        gradients={k:np.linspace(-.2,.3,int(np.prod(shape))).reshape(shape) for k,shape in SHAPES.items()}
        expanded={k:expand(v) if k=='W1' else v.copy() for k,v in gradients.items()}
        adam.update(old,gradients);optimizer.update(model,expanded)
        for key in SHAPES:
            for a,b in ((old.parameters,model.parameters),(adam.first,optimizer.first),(adam.second,optimizer.second)):
                assert np.array_equal(a[key],old_coordinates(b[key]) if key=='W1' else b[key])
        assert adam.step_count==optimizer.step_count==225
        # Artificial nonempty input tests derivative access to newly added rows.
        unary=np.zeros((2,160));relation=np.ones((2,48))
        _,state=model.forward(unary,relation,[1])
        gradient=model.backward(state,np.array([1.,0.,0.]))['W1'][192:208]
        assert np.isfinite(gradient).all() and np.any(gradient!=0)
        row,col=np.unravel_index(np.argmax(np.abs(gradient)),gradient.shape)
        index=(192+row,col);value=model.parameters['W1'][index]
        model.parameters['W1'][index]=value+1e-6
        plus=model.forward(unary,relation,[1])[0][0]
        model.parameters['W1'][index]=value-1e-6
        minus=model.forward(unary,relation,[1])[0][0]
        model.parameters['W1'][index]=value
        derivative_error=float(abs((plus-minus)/2e-6-gradient[row,col]))
        assert derivative_error<1e-7
        write(dest/'nearest_migration_control.json',dict(status='NEAREST_AND_COMPLETE_FROZEN_MIGRATION_PASSED',
            original_coordinates_and_Adam_step_exact=True,synthetic_next_Adam_bitwise_equal=True,
            actual_training_updates=0,synthetic_models_discarded=True,new_rows_receive_finite_gradient=True,
            new_row_finite_difference_error=derivative_error,
            pooling=pooling,records=records,cache_reuse=reuse,source_checkpoint_sha256=restored['checkpoint_sha256']))
        print('NEAREST_MIGRATION_CONTROL_PASSED',records,flush=True)
    finally:write(dest/'control_relation_cache_receipt.json',cache.receipt())
