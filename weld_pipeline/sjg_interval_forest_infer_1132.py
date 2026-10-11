"""Frozen pure-point-input inference for count/ordinal interval forests."""
import argparse
import pickle
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_joint_forest_train_1109 import load_case
from weld_pipeline.sjg_interval_forest_runtime_1129 import predict_instances,link_blocks
from weld_pipeline.sjg_joint_link_runtime_1107 import select_paths
from weld_pipeline.sjg_request_competition_1125 import competition_resources
from weld_pipeline.sjg_joint_forest_infer_1111 import decode_paths
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--model-run',default='run319');ap.add_argument('--source',required=True);args=ap.parse_args()
    start=time.perf_counter();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    try:
        training=read(OUT/args.model_run/'receipt.json')
        assert training['status']=='THREE_INTERVAL_FORESTS_TRAINING_COMPLETE'
        path=OUT/args.model_run/'model.pkl';assert digest(path)==training['model_sha256']
        with path.open('rb') as f:models=pickle.load(f)
        assert {name:forest_state_sha256(models[name]) for name in training['component_state_sha256']}==training['component_state_sha256']
        case=load_case(args.source,False);check_resources()
        screen=predict_instances(case,models);nodes=screen['nodes']
        edges=[];masks=[];probabilities=[]
        for link,features,legal in link_blocks(nodes,case['graph']):
            raw=models['connection'].predict_proba(features);p=np.zeros((len(link),4),np.float64)
            p[:,models['connection'].classes_.astype(int)]=raw
            edges.append(link);masks.append(legal);probabilities.append(p)
        links=np.concatenate(edges) if edges else np.empty((0,2),np.int32)
        legal=np.concatenate(masks) if masks else np.empty((0,3),bool)
        p=np.concatenate(probabilities) if probabilities else np.empty((0,4),np.float64)
        del edges,masks,probabilities
        request_path=OUT/models['input_request_run']/f'source_{args.source}_input_requests.json'
        bound=next(r for r in read(OUT/models['input_request_run']/'all_input_requests_frozen_before_labels.json')['cases'] if r['source']==args.source)
        assert digest(request_path)==bound['sha256']
        resources=competition_resources(nodes,case['graph'],read(request_path))
        check_resources()
        selection=select_paths(nodes,links,p,legal,wall_seconds=10.,competition_resources=resources,allow_node_selection=True)
        decoded=decode_paths(nodes,selection,case['graph'],screen['invalid'])
        decoded.update(runtime_model='ORDERED_LOCAL_INTERVAL_FORESTS',
            connector_removed_positive_nodes=selection['unselected_explanations'],
            interval_capacity_K=models['K'],GT_reads=0)
        write(dest/'decoded.json',decoded)
        write(dest/'selected_graph.json',dict(nodes=nodes,selection=selection,competition_resources=resources))
        write(dest/'receipt.json',dict(status='CONDITIONAL_PURE_INPUT_PREDICTION_FROZEN',source=args.source,
            model_run=args.model_run,model_sha256=training['model_sha256'],decoded_sha256=digest(dest/'decoded.json'),
            input_feature_receipt_sha256=case['feature_receipt_sha256'],alias_map_sha256=case['alias_map_sha256'],
            objects_scored=screen['objects_scored'],positive_objects=screen['admitted_instances'],
            admitted_interval_instances=screen['admitted_instances'],unselected_explanations=selection['unselected_explanations'],
            maximum_probability=screen['maximum_probability'],K=models['K'],saturated_objects=len(screen['saturated_object_rows']),
            saturated_object_rows=screen['saturated_object_rows'],conditional_edges=len(links),
            graph_selection_seconds=selection['seconds'],cached_pipeline_seconds=time.perf_counter()-start,
            GT_reads=0,STEP_reads=0,raw_PCD_uncached_seconds=None,full_contract_pass=False,utc=utc()))
        print('FROZEN_INTERVAL_PREDICTION',args.source,'ADMITTED',screen['admitted_instances'],
              'UNSELECTED',selection['unselected_explanations'],'OUTPUT_CHILDREN',decoded['total_prediction_children'],
              'SECONDS',time.perf_counter()-start,flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start,GT_reads=0));raise


if __name__=='__main__':main()
