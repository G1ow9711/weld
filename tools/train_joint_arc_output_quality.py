"""Fixed-budget reference-export quality head; family-held-out downstream tests."""
from pathlib import Path
import time
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_geometry_d3_409 import score_source
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def decoded(records,indices):
    parents=[records[i]['parent'] for i in indices if 'parent' in records[i]];invalid=[records[i]['invalid'] for i in indices if 'invalid' in records[i]]
    return dict(parents=parents,invalid_predictions=invalid,total_prediction_children=sum(p['child_count'] for p in parents)+sum(r['counted_false_children'] for r in invalid))


def main():
    started=time.perf_counter();source=BASE/'joint_arc_output_quality_data_v1';r=read(source/'receipt.json');assert digest(source/'input_receipt.json')==r['input_receipt_sha256'];assert digest(source/'labels.json')==r['labels_sha256'];inp=read(source/'input_receipt.json');assert digest(source/'features.npy')==inp['feature_sha256'];assert digest(source/'outputs.json')==inp['outputs_sha256'];x=np.load(source/'features.npy');records=read(source/'outputs.json')['rows'];labels=read(source/'labels.json');y=np.array([row['quality'] for row in labels['rows']]);families=np.array([row['family'] for row in records]);truth={}
    for family,key in [('first','new_ac2008a03607'),('second','new_fcaf2af1abee')]:
        path=BASE/'labels'/key/'targets.json';assert digest(path)==labels['target_sha256'][family];truth[family]=native_targets(read(path));baseline=score_source(decoded(records,np.flatnonzero(families==family)),truth[family],[False]*len(truth[family]['parents']));expected=read(BASE/('joint_arc_pointer_'+family+'_all_candidates_v1')/'geometry_diagnostic.json')['score'];assert baseline['categories']==expected['categories']
    out=BASE/'joint_arc_output_quality_head_v1';out.mkdir(exist_ok=False);config=dict(n_estimators=96,max_depth=18,min_samples_leaf=2,max_features=.75,random_state=1031,n_jobs=2)
    write(out/'protocol.json',dict(data_receipt_sha256=digest(source/'receipt.json'),config=config,threshold=.5,threshold_selection='Fixed before any quality-head fit; no sweep',sample_weight='Equal TRAIN family, half mass on quality>0 and half on zero reference-quality rows within family when both present',scope='Predict quality of exported reference geometry, not physical weld/contact existence. All output rows included. Quality head leaveout only; upstream pointer trained on both families. Invalid and duplicate outputs retain original scoring semantics. No independent end-to-end generalization claim.',fit_calls=3,DEV_reads=0))
    summary=[]
    for name,train_families,test_families in [('joint',['first','second'],['first','second']),('leave_first',['second'],['first']),('leave_second',['first'],['second'])]:
        train=np.flatnonzero(np.isin(families,train_families));weights=np.zeros(len(train))
        for family in train_families:
            local=families[train]==family;positive=local&(y[train]>0);negative=local&(y[train]==0);groups=[g for g in [positive,negative] if g.any()]
            for group in groups:weights[group]=1/len(train_families)/len(groups)/group.sum()
        model=ExtraTreesRegressor(**config);model.fit(x[train],y[train],sample_weight=weights);joblib.dump(model,out/(name+'.pkl'));pred=model.predict(x);assert np.isfinite(pred).all();np.save(out/(name+'_scores.npy'),pred)
        for family in test_families:
            selected=np.flatnonzero((families==family)&(pred>=.5));score=score_source(decoded(records,selected),truth[family],[False]*len(truth[family]['parents']));item=dict(split=name,family=family,selected_output_rows=len(selected),score=score,model_sha256=digest(out/(name+'.pkl')),scores_sha256=digest(out/(name+'_scores.npy')));summary.append(item);print(name,family,score['categories']['overall'],flush=True)
    write(out/'evaluation.json',dict(rows=summary,scope='Reference-export quality diagnostic, fixed-H and branch parent assumptions. Joint TRAIN fit and conditional downstream family leaveout only.',no_automatic_promotion=True));write(out/'receipt.json',dict(status='QUALITY_HEAD_EVALUATED_NOT_PROMOTED',protocol_sha256=digest(out/'protocol.json'),evaluation_sha256=digest(out/'evaluation.json'),fit_calls=3,seconds=time.perf_counter()-started,DEV_reads=0))

if __name__=='__main__':main()
