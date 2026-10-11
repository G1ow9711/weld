"""Freeze complete pure-input relative-forest prediction before independent scoring."""
import argparse
import pickle
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_relative_forest_runtime_1139 import load_relative_case,predict_nodes,relative_link_blocks
from weld_pipeline.sjg_joint_link_runtime_1107 import select_paths
from weld_pipeline.sjg_joint_forest_infer_1111 import decode_paths
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--model-run',default='run330')
    ap.add_argument('--source',required=True);ap.add_argument('--feature-run',default='run329');args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        training=read(OUT/args.model_run/'receipt.json');assert training['status']=='THREE_RELATIVE_FORESTS_TRAINING_COMPLETE'
        path=OUT/args.model_run/'model.pkl';assert digest(path)==training['model_sha256']
        with path.open('rb') as f:models=pickle.load(f)
        assert {k:forest_state_sha256(models[k]) for k in training['component_state_sha256']}==training['component_state_sha256']
        case=load_relative_case(args.source,args.feature_run,False);screen=predict_nodes(case,models['object'],models['endpoint'])
        nodes=screen['nodes'];all_links=[];all_masks=[];all_probs=[]
        for links,features,legal in relative_link_blocks(nodes,case):
            probabilities=np.zeros((len(links),4))
            probabilities[:,models['connection'].classes_.astype(int)]=models['connection'].predict_proba(features)
            all_links.append(links);all_masks.append(legal);all_probs.append(probabilities)
            check_resources()
        links=np.concatenate(all_links) if all_links else np.empty((0,2),np.int32)
        legal=np.concatenate(all_masks) if all_masks else np.empty((0,3),bool)
        probabilities=np.concatenate(all_probs) if all_probs else np.empty((0,4))
        selection=select_paths(nodes,links,probabilities,legal,wall_seconds=10.)
        decoded=decode_paths(nodes,selection,case['graph'],screen['invalid'])
        decoded['runtime_model']='RELATIVE_OBSERVED_FACE_152_THREE_FORESTS'
        write(dest/'decoded.json',decoded);write(dest/'selected_graph.json',dict(nodes=nodes,selection=selection))
        write(dest/'receipt.json',dict(status='CONDITIONAL_PURE_INPUT_PREDICTION_FROZEN',source=args.source,
            model_run=args.model_run,model_sha256=training['model_sha256'],decoded_sha256=digest(dest/'decoded.json'),
            input_feature_receipt_sha256=case['feature_receipt_sha256'],alias_map_sha256=case['alias_map_sha256'],
            feature_run=args.feature_run,objects_scored=screen['object_rows_scored'],positive_objects=screen['positive_count'],
            maximum_probability=screen['maximum_probability'],conditional_edges=len(links),graph_selection_seconds=selection['seconds'],
            cached_pipeline_seconds=time.perf_counter()-start,GT_reads=0,STEP_reads=0,
            raw_PCD_uncached_seconds=None,full_contract_pass=False,utc=utc()))
        print('RELATIVE_FROZEN_PREDICTION',args.source,'POSITIVE',screen['positive_count'],'OUTPUT',decoded['total_prediction_children'],flush=True)
    except BaseException:write(dest/'failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
