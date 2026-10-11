"""Matched child-quality heads; keep original parent only if all children pass."""
from pathlib import Path
import time
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from tools.train_joint_arc_output_quality import decoded
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_geometry_d3_409 import score_source
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main(feature_set='full'):
    started=time.perf_counter();source=BASE/'joint_arc_child_quality_data_v1';sr=read(source/'receipt.json');assert digest(source/'input_receipt.json')==sr['input_receipt_sha256'];assert digest(source/'labels.json')==sr['labels_sha256'];inp=read(source/'input_receipt.json');assert digest(source/'features.npy')==inp['feature_sha256'];assert digest(source/'rows.json')==inp['rows_sha256'];parentsource=BASE/'joint_arc_output_quality_data_v1';assert digest(parentsource/'receipt.json')==inp['parent_data_receipt_sha256'];parentinp=read(parentsource/'input_receipt.json');assert digest(parentsource/'outputs.json')==parentinp['outputs_sha256'];parents=read(parentsource/'outputs.json')['rows'];rows=read(source/'rows.json')['rows'];x=np.load(source/'features.npy');columns=np.arange(423) if feature_set=='full' else np.asarray([block*83+i for block in range(5) for i in range(69)]);x=x[:,columns];labels=read(source/'labels.json');y=np.array([r['quality'] for r in labels['rows']]);families=np.array([r['family'] for r in rows]);parent_ids=np.array([r['parent_row'] for r in rows]);parent_families=np.array([r['family'] for r in parents]);truth={}
    for family,key in [('first','new_ac2008a03607'),('second','new_fcaf2af1abee')]:
        path=BASE/'labels'/key/'targets.json';assert digest(path)==labels['target_sha256'][family];truth[family]=native_targets(read(path));assert set(parent_ids[families==family])==set(np.flatnonzero(parent_families==family))
    previous=read(BASE/'joint_arc_output_quality_head_v1/protocol.json');config=previous['config'];threshold=previous['threshold'];out=BASE/('joint_arc_child_quality_head_v1' if feature_set=='full' else 'joint_arc_child_quality_local_patch_v1');out.mkdir(exist_ok=False);write(out/'protocol.json',dict(feature_set=feature_set,selected_columns=columns.tolist(),data_receipt_sha256=digest(source/'receipt.json'),config=config,threshold=threshold,previous_parent_protocol_sha256=digest(BASE/'joint_arc_output_quality_head_v1/protocol.json'),aggregation='Keep original parent iff min child quality >= fixed0.5; no splitting/reordering/correction of original parent',sample_weight=previous['sample_weight'],scope='Same classifier budget, child instead of whole-parent quality units. Reference geometry quality only, not physical contact/existence. Shared upstream saw both families; leaveout is quality-head-only.',fit_calls=3,DEV_reads=0));summary=[]
    for name,train_families,test_families in [('joint',['first','second'],['first','second']),('leave_first',['second'],['first']),('leave_second',['first'],['second'])]:
        train=np.flatnonzero(np.isin(families,train_families));weights=np.zeros(len(train))
        for family in train_families:
            local=families[train]==family;groups=[g for g in [local&(y[train]>0),local&(y[train]==0)] if g.any()]
            for group in groups:weights[group]=1/len(train_families)/len(groups)/group.sum()
        model=ExtraTreesRegressor(**config);model.fit(x[train],y[train],sample_weight=weights);joblib.dump(model,out/(name+'.pkl'));pred=model.predict(x);parent_scores=np.ones(len(parents));np.minimum.at(parent_scores,parent_ids,pred);np.savez(out/(name+'_scores.npz'),child_scores=pred,parent_scores=parent_scores)
        for family in test_families:
            selected=np.flatnonzero((parent_families==family)&(parent_scores>=threshold));score=score_source(decoded(parents,selected),truth[family],[False]*len(truth[family]['parents']));summary.append(dict(split=name,family=family,selected_output_rows=len(selected),score=score,model_sha256=digest(out/(name+'.pkl')),scores_sha256=digest(out/(name+'_scores.npz'))));print(name,family,score['categories']['overall'],flush=True)
    write(out/'evaluation.json',dict(rows=summary,no_automatic_promotion=True));write(out/'receipt.json',dict(protocol_sha256=digest(out/'protocol.json'),evaluation_sha256=digest(out/'evaluation.json'),fit_calls=3,seconds=time.perf_counter()-started,DEV_reads=0))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--feature-set',choices=['full','local_patch'],default='full');main(p.parse_args().feature_set)
