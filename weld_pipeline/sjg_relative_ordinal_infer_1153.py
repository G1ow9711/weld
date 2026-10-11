"""Freeze the complete predicted interval graph before independent scoring."""
import argparse,pickle,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,check_resources,utc
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case,predict_instances,relative_link_blocks
from weld_pipeline.sjg_joint_link_runtime_1107 import select_paths
from weld_pipeline.sjg_request_competition_1125 import competition_resources
from weld_pipeline.sjg_joint_forest_infer_1111 import decode_paths
from weld_pipeline.sjg_forest_state_hash_1116 import forest_state_sha256


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--source',required=True)
    ap.add_argument('--model-run',default='run345');args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False);started=time.perf_counter()
    try:
        receipt=read(OUT/args.model_run/'receipt.json');assert receipt['status']=='THREE_RELATIVE_ORDINAL_FORESTS_TRAINING_COMPLETE'
        modelpath=OUT/args.model_run/'model.pkl';assert digest(modelpath)==receipt['model_sha256']
        with modelpath.open('rb') as f:models=pickle.load(f)
        assert receipt['component_state_sha256']=={key:forest_state_sha256(models[key]) for key in receipt['component_state_sha256']}
        for name in ('object','endpoint','connection'):models[name].verbose=0
        case=load_case(args.source);screen=predict_instances(case,models);nodes=screen['nodes'];edges=[];masks=[];probs=[]
        for links,features,legal in relative_link_blocks(nodes,case):
            p=np.zeros((len(links),4));p[:,models['connection'].classes_.astype(int)]=models['connection'].predict_proba(features)
            edges.append(links);masks.append(legal);probs.append(p);check_resources()
        links=np.concatenate(edges) if edges else np.empty((0,2),np.int32)
        legal=np.concatenate(masks) if masks else np.empty((0,3),bool);p=np.concatenate(probs) if probs else np.empty((0,4))
        del edges,masks,probs
        request=OUT/models['input_request_run']/f'source_{args.source}_input_requests.json'
        bound=next(v for v in read(OUT/models['input_request_run']/'all_input_requests_frozen_before_labels.json')['cases'] if v['source']==args.source)
        assert digest(request)==bound['sha256'];resources=competition_resources(nodes,case['graph'],read(request))
        selected=select_paths(nodes,links,p,legal,wall_seconds=10.,competition_resources=resources,allow_node_selection=True)
        decoded=decode_paths(nodes,selected,case['graph'],screen['invalid'])
        decoded.update(runtime_model='RELATIVE_COMPATIBLE_K12_FORESTS',interval_capacity_K=12,
            connector_removed_positive_nodes=selected['unselected_explanations'],GT_reads=0)
        write(dest/'decoded.json',decoded);write(dest/'selected_graph.json',dict(nodes=nodes,selection=selected,competition_resources=resources))
        write(dest/'receipt.json',dict(status='CONDITIONAL_PURE_INPUT_PREDICTION_FROZEN',source=args.source,model_run=args.model_run,
            model_sha256=receipt['model_sha256'],decoded_sha256=digest(dest/'decoded.json'),objects_scored=screen['objects_scored'],
            positive_objects=screen['admitted_instances'],maximum_probability=screen['maximum_probability'],conditional_edges=len(links),
            graph_selection_seconds=selected['seconds'],cached_pipeline_seconds=time.perf_counter()-started,
            input_feature_receipt_sha256=case['feature_receipt_sha256'],alias_map_sha256=case['alias_map_sha256'],
            saturated_object_rows=screen['saturated_object_rows'],GT_reads=0,STEP_reads=0,raw_PCD_uncached_seconds=None,full_contract_pass=False,utc=utc()))
        print('RELATIVE_ORDINAL_PREDICTION',args.source,'ADMITTED',screen['admitted_instances'],'OUTPUT',decoded['total_prediction_children'],flush=True)
    except BaseException:write(dest/'failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-started));raise


if __name__=='__main__':main()
