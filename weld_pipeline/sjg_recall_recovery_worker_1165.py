"""Pro358 component-resident screen/graph stages; no forest fitting entrypoint."""
import argparse, gc, shutil, time, traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_staged_relative_worker_1154 import load_component
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case,predict_instances
from weld_pipeline.sjg_relative_forest_runtime_1139 import relative_link_blocks
from weld_pipeline.sjg_bounded_gap_runtime_1164 import GapResolver
from weld_pipeline.sjg_joint_link_runtime_1107 import select_paths
from weld_pipeline.sjg_request_competition_1125 import competition_resources
from weld_pipeline.sjg_joint_forest_infer_1111 import decode_paths
from weld_pipeline.sjg_parent_set_features_1160 import unary_features
from weld_pipeline.sjg_recall_set_worker_1162 import freeze_compact


def component(name):
    source=OUT/('run358' if name=='object' else 'run349')
    model,r=load_component(source,name);model.verbose=0
    binding=read(OUT/'run358/upstream_verified.json')
    assert r['numeric_state_sha256']==binding['component_state_sha256'][name]
    assert r['component_sha256']==binding['component_file_sha256'][name]
    return model


def screen(dest,key):
    case=load_case(key);models=dict(object=component('object'),endpoint=component('endpoint'),K=12)
    value=predict_instances(case,models)
    value.update(model_sha256=read(OUT/'run358/upstream_verified.json')['model_sha256'],
        feature_receipt_sha256=case['feature_receipt_sha256'],GT_reads=0,fit_calls=0,source=key)
    path=dest/f'screen_{key}.json';write(path,value)
    write(dest/f'screen_{key}_receipt.json',dict(screen_sha256=digest(path),nodes=len(value['nodes']),
        invalid=len(value['invalid']),model_sha256=value['model_sha256'],feature_receipt_sha256=value['feature_receipt_sha256']))
    print('SCREEN_FROZEN',key,'NODES',len(value['nodes']),'INVALID',len(value['invalid']),flush=True)


def request_resources(nodes,case,key):
    req=OUT/'run312'/f'source_{key}_input_requests.json'
    bound=next(v for v in read(OUT/'run312/all_input_requests_frozen_before_labels.json')['cases'] if v['source']==key)
    assert digest(req)==bound['sha256']
    return competition_resources(nodes,case['graph'],read(req))


def graph(dest,key,test=False):
    begin=time.perf_counter();folder=dest/(('test_' if test else 'source_')+key);folder.mkdir(exist_ok=False)
    frozen=read(dest/f'screen_{key}_receipt.json');path=dest/f'screen_{key}.json'
    assert digest(path)==frozen['screen_sha256'];value=read(path);nodes=value['nodes'];case=load_case(key)
    binding=read(OUT/'run358/upstream_verified.json')
    assert value['model_sha256']==binding['model_sha256']
    assert value['feature_receipt_sha256']==case['feature_receipt_sha256']
    connection=component('connection')
    resolver=GapResolver(case['graph'],[n['fragment'] for n in nodes],audit=(key=='10'))
    print('GRAPH_ONLY_CONNECTION_RESIDENT',key,'NODES',len(nodes),'CACHE_BYTES',resolver.stats['cache_bytes'],flush=True)
    edges=[];masks=[];probabilities=[];count=0
    for number,(links,features,legal) in enumerate(relative_link_blocks(nodes,case,resolver)):
        p=np.zeros((len(links),4));p[:,connection.classes_.astype(int)]=connection.predict_proba(features)
        edges.append(links);masks.append(legal);probabilities.append(p);count+=len(links)
        if number%100==0:
            print('GRAPH_ROWS',key,number,'/',len(nodes),'EDGES',count,'EXACT_GAP_CALLS',resolver.stats['exact_calls'],flush=True)
        check_resources()
    write(folder/'gap_execution.json',resolver.receipt())
    del connection,resolver;gc.collect()
    links=np.concatenate(edges) if edges else np.empty((0,2),np.int32)
    legal=np.concatenate(masks) if masks else np.empty((0,3),bool)
    p=np.concatenate(probabilities) if probabilities else np.empty((0,4))
    del edges,masks,probabilities;gc.collect()
    write(folder/'complete_edge_table_summary.json',dict(edges=count,source=key,all_legal_edges_complete=True,
        links_sha256=__import__('hashlib').sha256(links.tobytes()).hexdigest(),GT_reads=0))
    print('FREE_GRAPH_COMPLETE',key,'ALL_LEGAL_EDGES',count,flush=True)
    resources=request_resources(nodes,case,key)
    selected=select_paths(nodes,links,p,legal,wall_seconds=10.,competition_resources=resources,allow_node_selection=True)
    del links,p,legal;gc.collect()
    decoded=decode_paths(nodes,selected,case['graph'],value['invalid'])
    decoded.update(runtime_model='PRO355_FREE_UPSTREAM_CANDIDATES',GT_reads=0)
    point_count=sum(len(s['points']) for p in decoded['parents'] for s in p['segments'])
    estimated_write_bytes=point_count*256+len(nodes)*4096+4*1024*1024
    free=shutil.disk_usage(ROOT).free
    write(folder/'disk_write_reservation.json',dict(free_bytes=free,estimated_output_bytes=estimated_write_bytes,
        floor_bytes=5368709120,estimate_is_not_measured_file_size=True))
    if free-estimated_write_bytes<5368709120:raise RuntimeError('Insufficient space above original disk floor for complete outputs')
    write(folder/'decoded.json',decoded)
    write(folder/'selected_graph.json',dict(nodes=nodes,selection=selected,competition_resources=resources))
    u,metadata=unary_features(decoded,nodes,selected,case)
    np.save(folder/'unary.npy',u);write(folder/'parent_metadata.json',metadata);freeze_compact(folder,decoded)
    write(folder/'receipt.json',dict(status='PURE_INPUT_COMPLETE_PARENT_CANDIDATES_FROZEN',source=key,
        upstream_model_sha256=binding['model_sha256'],decoded_sha256=digest(folder/'decoded.json'),
        unary_sha256=digest(folder/'unary.npy'),input_feature_receipt_sha256=case['feature_receipt_sha256'],
        candidate_parents=len(decoded['parents']),candidate_children=decoded['total_prediction_children'],
        conditional_edges=count,graph_selection_seconds=selected['seconds'],cached_pipeline_seconds=time.perf_counter()-begin,
        screen_stage_separate=True,screen_sha256=frozen['screen_sha256'],GT_reads=0,STEP_reads=0,teacher_nodes_added=0))
    print('FREE_PARENT_OUTPUT',key,len(decoded['parents']),decoded['total_prediction_children'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True);ap.add_argument('--source',default='')
    args=ap.parse_args();dest=OUT/args.run;start=time.perf_counter()
    try:
        check_resources()
        if args.stage=='screen':screen(dest,args.source)
        elif args.stage in ('graph','test_graph'):graph(dest,args.source,args.stage=='test_graph')
        elif args.stage=='control':
            from weld_pipeline.sjg_gap_equivalence_control_1167 import control
            control(dest)
        else:raise ValueError('No forest fit is permitted in recovery')
        write(dest/f'stage_{args.stage}_{args.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-start,
            worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{args.stage}_{args.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
