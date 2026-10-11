"""One execution-equivalence control; never a model-quality test."""
import hashlib, itertools, json, time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,check_resources
from weld_pipeline.sjg_bounded_gap_runtime_1164 import GapResolver
from weld_pipeline.sjg_joint_segment_graph_1104 import JointSegmentGraph
from weld_pipeline.sjg_recall_recovery_worker_1165 import component,request_resources
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_relative_forest_runtime_1139 import relative_link_blocks
from weld_pipeline.sjg_joint_link_runtime_1107 import select_paths
from weld_pipeline.sjg_joint_forest_infer_1111 import decode_paths


def boundary_control():
    results=[]
    for height,source,reverse,gaps in itertools.product(
        [np.nextafter(3.,0.),3.,np.nextafter(3.,np.inf),3.+1e-10], [0,1], [False,True], [[[5,15]],[[5,7],[13,15]]]):
        profiles=[]
        for k,y in enumerate((0.,height)):
            ends=[[0.,y,0.],[20.,y,0.]]
            if reverse:ends.reverse()
            profiles.append(dict(representation=dict(kind='line',endpoints_mm=ends),face_indices=[0,1],
                unobserved_gaps_mm=gaps if k==source else [],support_intervals_mm=[]))
        graph=JointSegmentGraph(profiles,[0,2,4],[[0,5,1,0],[15,20,1,0]]*2)
        compact=GapResolver(graph,range(4),audit=True)
        lru=GapResolver(graph,range(4),audit=True,matrix_limit=0,lru_limit=1)
        for a,b in [(0,3),(1,2),(0,3),(3,0),(2,1)]:
            old=graph.connection(a,b,'POSITIVE_GAP')
            assert compact.resolve(a,[b])[0]==old and lru.resolve(a,[b])[0]==old
            # The original scalar necessary condition also verifies batching.
            assert compact.possible[a,b]==compact._original_necessary(a,b)
        assert len(lru.lru)<=1
        if height<=3. and len(gaps)==1:assert lru.stats['exact_calls']>=3
        results.append(dict(height=float(height),source=source,reverse=reverse,gaps=gaps,stats=lru.receipt()))
    return results


def control(dest):
    started=time.perf_counter();boundary=boundary_control();connection=component('connection');cases=[]
    for key in ('21','BM79A_native_local'):
        folder=OUT/'run358'/f'source_{key}';saved=read(folder/'selected_graph.json');nodes=saved['nodes'];case=load_case(key)
        resolver=GapResolver(case['graph'],[n['fragment'] for n in nodes],audit=True)
        old=relative_link_blocks(nodes,case);new=relative_link_blocks(nodes,case,resolver)
        edges=[];masks=[];probs=[];h=hashlib.sha256();count=0
        for a,b in itertools.zip_longest(old,new):
            assert a is not None and b is not None
            assert all(np.array_equal(x,y) for x,y in zip(a,b)), 'Edge/feature/order mismatch'
            pa=connection.predict_proba(a[1]);pb=connection.predict_proba(b[1]);assert np.array_equal(pa,pb)
            full=np.zeros((len(a[0]),4));full[:,connection.classes_.astype(int)]=pb
            edges.append(b[0]);masks.append(b[2]);probs.append(full);count+=len(b[0])
            for x in (*a,pa):h.update(x.tobytes())
        links=np.concatenate(edges);legal=np.concatenate(masks);p=np.concatenate(probs)
        resources=request_resources(nodes,case,key)
        assert resources==saved['competition_resources']
        selected=select_paths(nodes,links,p,legal,wall_seconds=10.,competition_resources=resources,allow_node_selection=True)
        assert selected['selected_nodes']==saved['selection']['selected_nodes']
        assert [list(x) for x in selected['selected_links']]==saved['selection']['selected_links']
        decoded=decode_paths(nodes,selected,case['graph'],[])
        original=read(folder/'decoded.json')
        assert not original['invalid_predictions']
        assert json.dumps(decoded['parents'],sort_keys=True)==json.dumps(original['parents'],sort_keys=True)
        assert decoded['total_prediction_children']==original['total_prediction_children']
        cases.append(dict(source=key,edges=count,parents=len(decoded['parents']),all_features_and_probabilities_exact=True,
            complete_parent_paths_exact=True,values_sha256=h.hexdigest(),gap=resolver.receipt()))
        check_resources()
    write(dest/'equivalence_control.json',dict(status='EXECUTION_EQUIVALENCE_PASSED',cases=cases,
        synthetic_boundaries=boundary,seconds=time.perf_counter()-started,forest_fit_calls=0,GT_reads=0,
        new_extraction_result=False))
    print('EXECUTION_EQUIVALENCE_PASSED',[(x['source'],x['edges']) for x in cases],flush=True)
