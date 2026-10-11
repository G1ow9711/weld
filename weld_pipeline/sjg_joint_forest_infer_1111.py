"""Frozen pure-input conditional forest inference; no supervision reads."""
import argparse
import pickle
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc,check_resources
from weld_pipeline.sjg_joint_forest_train_1109 import load_case,predict_nodes,compact_connections
from weld_pipeline.sjg_joint_link_runtime_1107 import pair_features,select_paths
from weld_pipeline.sjg_joint_graph_capacity_1105 import ContinuousGraphBank
from weld_pipeline.sjg_parent_set_runtime_1081 import decode_choices


def decode_paths(nodes,selection,graph,initial_invalid):
    successors={a:(b,k) for a,b,k in selection['selected_links']}
    predecessors={b:a for a,b,k in selection['selected_links']}
    visited=set();paths=[];chosen=selection['selected_nodes'];choices=[];invalid=[]
    for i in chosen:
        if i in predecessors:continue
        chain=[];edges=[];j=i
        while j not in visited:
            visited.add(j);chain.append(j)
            if j not in successors:break
            j,kind=successors[j];edges.append(kind)
        paths.append((chain,edges,False))
    for i in chosen:
        if i in visited:continue
        chain=[];edges=[];j=i
        while j not in visited:
            visited.add(j);chain.append(j)
            if j not in successors:raise ValueError('Unvisited component is not a directed cycle')
            j,kind=successors[j];edges.append(kind)
        if j!=i:raise ValueError('Path joins an already occupied node')
        # A child boundary is the stable start of a mixed-child closed parent.
        boundary=next((k for k,e in enumerate(edges) if e!=1),None)
        if boundary is not None:
            shift=(boundary+1)%len(chain);chain=chain[shift:]+chain[:shift];edges=edges[shift:]+edges[:shift]
        paths.append((chain,edges,True))
    assert visited==set(chosen)
    for query,(chain,edges,cycle) in enumerate(paths):
        groups=[];connections=[]
        for pos,index in enumerate(chain):
            if pos==0 or edges[pos-1]!=1:
                groups.append([index])
                if pos:connections.append('SHARED' if edges[pos-1]==2 else 'POSITIVE_GAP')
            else:groups[-1].append(index)
        try:
            if cycle and edges[-1]==3:raise ValueError('Positive gap cannot close a shared closed-parent boundary')
            children=[]
            for ci,group in enumerate(groups):
                ns=[nodes[i] for i in group];first=ns[0];pid=graph.profile(first['fragment']);curve=graph.curve(pid)
                if any(graph.profile(n['fragment'])!=pid or n['side']!=first['side'] or n['forward']!=first['forward'] or n['weld_type']!=first['weld_type'] for n in ns):
                    raise ValueError('SAME_CHILD path changes curve, role, direction or weld category')
                closed=bool((cycle and len(groups)==1) or (len(group)==1 and first.get('closed')))
                if closed and (curve.kind!='arc' or not curve.arc.is_closed):raise ValueError('Closed child lacks a complete circular carrier')
                start=first['lo'] if first['forward'] else first['hi']
                end=ns[-1]['hi'] if first['forward'] else ns[-1]['lo']
                lo,hi=sorted((start,end));wrap=False
                if len(ns)>1 and curve.kind=='arc' and curve.arc.is_closed:
                    positions=[n['lo'] if n['forward'] else n['hi'] for n in ns]
                    differences=np.diff(positions)*(1 if first['forward'] else -1)
                    wrap=bool(np.any(differences<0))
                if closed:lo=hi=float(start)
                children.append(dict(profile=pid,intervals=[0,0],fractions=[lo/curve.length,hi/curve.length],
                    forward=first['forward'],closed=closed,wrap=wrap,weld_type=first['weld_type'],
                    side='UNKNOWN',connection='UNKNOWN' if ci==0 else connections[ci-1]))
            parent_closed=bool(cycle or (len(children)==1 and children[0]['closed']))
            choices.append(dict(query=query,probability=1.,child_count=len(children),children=children,closed=parent_closed))
        except ValueError as exc:
            invalid.append(dict(counted_false_children=len(groups),predicted_weld_types=[nodes[g[0]]['weld_type'] for g in groups],
                predicted_kinds=[1 if graph.curve(graph.profile(nodes[g[0]]['fragment'])).kind=='line' else 2 for g in groups],
                reason=str(exc),parent_query=query))
    decoded=decode_choices(choices,ContinuousGraphBank(graph))
    for entry in initial_invalid:
        node=entry['node'];curve=graph.curve(graph.profile(node['fragment']))
        invalid.append(dict(counted_false_children=1,predicted_weld_types=[node['weld_type']],
                            predicted_kinds=[1 if curve.kind=='line' else 2],reason=entry['reason']))
    decoded['invalid_predictions'].extend(invalid)
    decoded['total_prediction_children']+=sum(x['counted_false_children'] for x in invalid)
    decoded.update(runtime_model='CONDITIONAL_FINITE_JOINT_EXTRA_TREES',positive_fragments=len(chosen)+len(initial_invalid),
                   connector_removed_positive_nodes=0,GT_reads=0)
    return decoded


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--model-run',default='run289');ap.add_argument('--source',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        training=read(OUT/args.model_run/'receipt.json')
        assert training['status'] in ('THREE_CPU_FORESTS_TRAINING_COMPLETE','CONDITIONAL_MODEL_OBJECT_REFIT_COMPLETE',
                                      'CONDITIONAL_SAVED_MODEL_POSTHOC_VERIFIED','CONDITIONAL_MODEL_CONNECTION_REFIT_COMPLETE',
                                      'CONDITIONAL_MODEL_ENDPOINT_REFIT_COMPLETE')
        model_path=OUT/args.model_run/'model.pkl';assert digest(model_path)==training['model_sha256']
        check_resources()
        with model_path.open('rb') as f:models=pickle.load(f)
        if training['status'] in ('CONDITIONAL_MODEL_OBJECT_REFIT_COMPLETE','CONDITIONAL_SAVED_MODEL_POSTHOC_VERIFIED',
                                  'CONDITIONAL_MODEL_CONNECTION_REFIT_COMPLETE','CONDITIONAL_MODEL_ENDPOINT_REFIT_COMPLETE'):
            from weld_pipeline.sjg_object_depth32_train_1114 import forest_sha
            assert {name:forest_sha(models[name]) for name in training['frozen_component_sha256']}==training['frozen_component_sha256']
        case=load_case(args.source,supervision=False);screen=predict_nodes(case,models['object'],models['endpoint'])
        nodes=screen['nodes'];links,legal,geometry=compact_connections(nodes,case['graph'])
        probabilities=np.zeros((len(links),4),np.float64)
        for first in range(0,len(links),32768):
            last=min(len(links),first+32768);features=np.empty((last-first,32),np.float32)
            for row,e in enumerate(range(first,last)):
                a,b=map(int,links[e]);features[row],_=pair_features(nodes[a],nodes[b],case['graph'],(geometry[a],geometry[b]))
            values=models['connection'].predict_proba(features)
            probabilities[first:last,models['connection'].classes_.astype(int)]=values
            check_resources()
        selection=select_paths(nodes,links,probabilities,legal,wall_seconds=10.)
        decoded=decode_paths(nodes,selection,case['graph'],screen['invalid'])
        write(dest/'decoded.json',decoded);write(dest/'selected_graph.json',dict(nodes=nodes,selection=selection))
        write(dest/'receipt.json',dict(status='CONDITIONAL_PURE_INPUT_PREDICTION_FROZEN',source=args.source,
            model_run=args.model_run,model_sha256=training['model_sha256'],decoded_sha256=digest(dest/'decoded.json'),
            input_feature_receipt_sha256=case['feature_receipt_sha256'],alias_map_sha256=case['alias_map_sha256'],
            objects_scored=screen['object_rows_scored'],positive_objects=screen['positive_count'],
            maximum_probability=screen['maximum_probability'],conditional_edges=len(links),
            graph_selection_seconds=selection['seconds'],cached_pipeline_seconds=time.perf_counter()-start,
            original_training_worker_exit_code=training.get('original_worker_exit_code',0),
            original_training_protocol_pass=training.get('external_budget_pass',False),
            training_verification_repaired_without_refit=training['status']=='CONDITIONAL_SAVED_MODEL_POSTHOC_VERIFIED',
            GT_reads=0,STEP_reads=0,raw_PCD_uncached_seconds=None,full_contract_pass=False,utc=utc()))
        print('FROZEN_CONDITIONAL_PREDICTION',args.source,'POSITIVE',screen['positive_count'],
              'OUTPUT_CHILDREN',decoded['total_prediction_children'],'SECONDS',time.perf_counter()-start,flush=True)
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-start,utc=utc()));raise


if __name__=='__main__':main()
