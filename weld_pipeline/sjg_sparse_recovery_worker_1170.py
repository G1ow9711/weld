"""Pro359 streamed graph, separate sparse solve, and separate geometry decode."""
import argparse,gc,hashlib,json,os,shutil,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources,resources,utc


def phase(folder,name,**extra):
    value=dict(phase=name,utc=utc(),memory=resources(),**extra)
    write(folder/'phase.json',value)
    with (folder/'phase_history.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
    print('SPARSE_PHASE',name,extra,flush=True)


def budget_disk(size):
    if shutil.disk_usage(ROOT).free-int(size)<5368709120:raise RuntimeError('Complete artifact write would cross original 5GiB floor')


def load_edges(folder):
    r=read(folder/'edge_manifest.json');assert r['status']=='COMPLETE_GRAPH_AND_RESOURCES_FROZEN'
    for name,b in r['files'].items():assert digest(folder/name)==b['sha256']
    n=r['edge_count']
    if n:
        links=np.memmap(folder/'edges.i32',dtype='<i4',mode='r',shape=(n,2))
        legal=np.memmap(folder/'legal.u8',dtype='u1',mode='r',shape=(n,3)).view(bool)
        p=np.memmap(folder/'probabilities.f64',dtype='<f8',mode='r',shape=(n,4))
    else:links=np.empty((0,2),np.int32);legal=np.empty((0,3),bool);p=np.empty((0,4))
    res={name:np.load(folder/('resources_'+name+'.npy'),mmap_mode='r') for name in ('offsets','indices','bounds','group_ids')}
    res['keys']=read(folder/'resource_keys.json')
    return links,p,legal,res,r


def build_graph(dest,key,test=False):
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    from weld_pipeline.sjg_relative_forest_runtime_1139 import relative_link_blocks
    from weld_pipeline.sjg_bounded_gap_runtime_1164 import GapResolver
    from weld_pipeline.sjg_recall_recovery_worker_1165 import component
    from weld_pipeline.sjg_compact_resources_1169 import compact_resources,compare_legacy
    from weld_pipeline.sjg_request_competition_1125 import competition_resources
    folder=dest/(('test_' if test else 'source_')+key);folder.mkdir(exist_ok=False);begin=time.perf_counter()
    screen_path=dest/f'screen_{key}.json';sr=read(dest/f'screen_{key}_receipt.json');assert digest(screen_path)==sr['screen_sha256']
    screen=read(screen_path);nodes=screen['nodes'];case=load_case(key)
    binding=read(dest/'upstream_verified.json')
    assert screen['model_sha256']==binding['model_sha256'] and screen['feature_receipt_sha256']==case['feature_receipt_sha256']
    phase(folder,'graph_start',nodes=len(nodes));connection=component('connection')
    resolver=GapResolver(case['graph'],[n['fragment'] for n in nodes],audit=False)
    names=('edges.i32','legal.u8','probabilities.f64');hashes={name:hashlib.sha256() for name in names};features_hash=hashlib.sha256()
    streams={name:(folder/name).open('wb') for name in names};count=0
    try:
        for row,(links,features,legal) in enumerate(relative_link_blocks(nodes,case,resolver)):
            p=np.zeros((len(links),4));p[:,connection.classes_.astype(int)]=connection.predict_proba(features)
            budget_disk(links.nbytes+legal.nbytes+p.nbytes+1024*1024)
            for name,arr in zip(names,(links.astype('<i4',copy=False),legal.astype('u1',copy=False),p.astype('<f8',copy=False))):
                block=arr.tobytes();streams[name].write(block);hashes[name].update(block)
            features_hash.update(features.tobytes());count+=len(links)
            if row%200==0:print('STREAM_ALL_EDGES',key,row,'/',len(nodes),'EDGES',count,flush=True)
            check_resources()
        for stream in streams.values():stream.flush();os.fsync(stream.fileno())
    finally:
        for stream in streams.values():stream.close()
    write(folder/'gap_execution.json',resolver.receipt());del connection,resolver;gc.collect()
    files={name:dict(sha256=hashes[name].hexdigest(),bytes=(folder/name).stat().st_size) for name in names}
    write(folder/'all_edges_frozen.json',dict(edge_count=count,files=files,features_sha256=features_hash.hexdigest(),
        original_directed_order=True,legal_classes=['SAME_CHILD','SHARED','POSITIVE_GAP'],probability_columns=[0,1,2,3],GT_reads=0))
    phase(folder,'all_edges_frozen',edge_count=count,seconds=time.perf_counter()-begin)
    request_path=OUT/'run312'/f'source_{key}_input_requests.json'
    request_binding=next(v for v in read(OUT/'run312/all_input_requests_frozen_before_labels.json')['cases'] if v['source']==key)
    assert digest(request_path)==request_binding['sha256'];frozen=read(request_path)
    phase(folder,'resource_sweep_start');res=compact_resources(nodes,case['graph'],frozen,folder=folder)
    if key in ('21','BM79A_native_local'):compare_legacy(res,competition_resources(nodes,case['graph'],frozen))
    budget_disk(sum(v.nbytes for v in res.values() if isinstance(v,np.ndarray))+1024*1024)
    for name in ('offsets','indices','bounds','group_ids'):
        path=folder/('resources_'+name+'.npy');res[name].flush();files[path.name]=dict(sha256=digest(path),bytes=path.stat().st_size)
    write(folder/'resource_keys.json',res['keys']);files['resource_keys.json']=dict(sha256=digest(folder/'resource_keys.json'))
    write(folder/'edge_manifest.json',dict(status='COMPLETE_GRAPH_AND_RESOURCES_FROZEN',edge_count=count,files=files,
        source=key,features_sha256=features_hash.hexdigest(),screen_sha256=sr['screen_sha256'],model_sha256=binding['model_sha256'],
        input_feature_receipt_sha256=case['feature_receipt_sha256'],resource_rows=len(res['offsets'])-1,
        resource_members=len(res['indices']),seconds=time.perf_counter()-begin,forest_fit_calls=0,GT_reads=0,STEP_reads=0))
    phase(folder,'graph_process_complete',resource_rows=len(res['offsets'])-1,resource_members=len(res['indices']))


def solve(dest,key,test=False):
    from weld_pipeline.sjg_sparse_path_solver_1168 import select_paths
    folder=dest/(('test_' if test else 'source_')+key)
    links,p,legal,res,r=load_edges(folder);screen=read(dest/f'screen_{key}.json')
    assert digest(dest/f'screen_{key}.json')==r['screen_sha256']
    selection,mapping=select_paths(screen['nodes'],links,p,legal,res,wall_seconds=60.,allow_node_selection=True,
        phase=lambda name,**extra:phase(folder,name,**extra))
    budget_disk(sum(v.nbytes for v in mapping.values())+1024*1024)
    for name,value in mapping.items():np.save(folder/('solver_'+name+'.npy'),value)
    write(folder/'selection.json',selection)
    write(folder/'solve_receipt.json',dict(status='COMPLETE_SPARSE_PATH_SOLUTION',edge_manifest_sha256=digest(folder/'edge_manifest.json'),
        selection_sha256=digest(folder/'selection.json'),no_forests_loaded=True,no_geometry_loaded=True,
        map_files={name:digest(folder/('solver_'+name+'.npy')) for name in mapping},GT_reads=0))


def decode(dest,key,test=False):
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    from weld_pipeline.sjg_joint_forest_infer_1111 import decode_paths
    from weld_pipeline.sjg_parent_set_features_1160 import unary_features
    from weld_pipeline.sjg_recall_set_worker_1162 import freeze_compact
    folder=dest/(('test_' if test else 'source_')+key);r=read(folder/'solve_receipt.json')
    assert digest(folder/'selection.json')==r['selection_sha256']
    manifest=read(folder/'edge_manifest.json');assert digest(folder/'edge_manifest.json')==r['edge_manifest_sha256']
    screen=read(dest/f'screen_{key}.json');assert digest(dest/f'screen_{key}.json')==manifest['screen_sha256']
    case=load_case(key);nodes=screen['nodes'];selected=read(folder/'selection.json')
    decoded=decode_paths(nodes,selected,case['graph'],screen['invalid']);decoded.update(runtime_model='PRO359_EQUIVALENT_SPARSE_UPSTREAM',GT_reads=0)
    points=sum(len(s['points']) for p in decoded['parents'] for s in p['segments'])
    budget_disk(points*256+len(nodes)*4096+4*1024*1024)
    write(folder/'decoded.json',decoded)
    write(folder/'selected_graph.json',dict(nodes=nodes,selection=selected,competition_resources_format='member_CSR',
        edge_manifest_sha256=r['edge_manifest_sha256']))
    u,metadata=unary_features(decoded,nodes,selected,case);np.save(folder/'unary.npy',u)
    write(folder/'parent_metadata.json',metadata);freeze_compact(folder,decoded)
    write(folder/'receipt.json',dict(status='PURE_INPUT_COMPLETE_PARENT_CANDIDATES_FROZEN',source=key,
        upstream_model_sha256=manifest['model_sha256'],decoded_sha256=digest(folder/'decoded.json'),unary_sha256=digest(folder/'unary.npy'),
        input_feature_receipt_sha256=case['feature_receipt_sha256'],candidate_parents=len(decoded['parents']),
        candidate_children=decoded['total_prediction_children'],conditional_edges=manifest['edge_count'],
        graph_selection_seconds=selected['seconds'],GT_reads=0,STEP_reads=0,teacher_nodes_added=0,
        solver_revision='1170_PRO359_SPARSE_DOMINANCE',possible_tied_optimum_path_change=True))
    print('FROZEN_COMPLETE_PARENTS',key,len(decoded['parents']),decoded['total_prediction_children'],flush=True)


def path_control(dest,key):
    from weld_pipeline.sjg_sparse_path_solver_1168 import validate_original,objective_exact
    folder=dest/f'source_{key}';links,p,legal,res,r=load_edges(folder);screen=read(dest/f'screen_{key}.json')
    old=read(OUT/'run358'/f'source_{key}/selected_graph.json')['selection'];new=read(folder/'selection.json')
    validate_original(screen['nodes'],links,legal,res,old);validate_original(screen['nodes'],links,legal,res,new)
    old_value=objective_exact(screen['nodes'],links,p,old);new_value=objective_exact(screen['nodes'],links,p,new)
    assert old_value==new_value,'Objectives differ as exact rational representations of original float64 coefficients'
    identical=old['selected_nodes']==new['selected_nodes'] and old['selected_links']==new['selected_links']
    if identical:
        before=read(OUT/'run358'/f'source_{key}/decoded.json');after=read(folder/'decoded.json')
        assert before['parents']==after['parents'] and before['invalid_predictions']==after['invalid_predictions']
        assert before['total_prediction_children']==after['total_prediction_children']
    write(folder/'path_control.json',dict(status='PATHS_IDENTICAL' if identical else 'TIED_OPTIMUM_PATH_CHANGE',
        original_constraints_pass=True,objective_exact_numerator=str(new_value.numerator),objective_exact_denominator=str(new_value.denominator),
        old_paths_reused_for_selector=False,new_prediction_sha256=digest(folder/'decoded.json'),GT_reads=0))
    print('PATH_CONTROL',key,'IDENTICAL' if identical else 'TIED_OPTIMUM_PATH_CHANGE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True);ap.add_argument('--source',default='')
    args=ap.parse_args();dest=OUT/args.run;start=time.perf_counter();stage=args.stage
    try:
        check_resources()
        if stage=='mechanism':
            from weld_pipeline.sjg_sparse_solver_control_1172 import control
            control(dest)
        elif stage in ('graph','test_graph'):build_graph(dest,args.source,stage.startswith('test_'))
        elif stage in ('solve','test_solve'):solve(dest,args.source,stage.startswith('test_'))
        elif stage in ('decode','test_decode'):decode(dest,args.source,stage.startswith('test_'))
        elif stage=='path_control':path_control(dest,args.source)
        else:raise ValueError(stage)
        write(dest/f'stage_{stage}_{args.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-start,
            worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{stage}_{args.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start));raise


if __name__=='__main__':main()
