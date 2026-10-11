"""Compile certified complete-parent explanations, without global re-solving.

Layered reachability proposes one complete witness through each alternative.
Every proposed whole parent is then checked with the original D3 scorer and
input competition constraints. An unproved alternative stays unresolved.
"""
import gc
import json
import time
from collections import defaultdict
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, check_resources
from weld_pipeline.sjg_joint_forest_train_1109 import load_case, category, attributes
from weld_pipeline.sjg_joint_link_runtime_1107 import pair_features
from weld_pipeline.sjg_joint_graph_capacity_1105 import decode_parent
from weld_pipeline.sjg_request_competition_1125 import competition_resources
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents

KEYS=('21','4','10','25','36','42','BM79A_native_local')


def layered_path(sizes, edges, forced, closed_edges=None):
    """Reachability with predecessor pointers; never enumerate path products."""
    starts=range(sizes[0]) if closed_edges is not None else [None]
    for start in starts:
        allowed=[np.ones(n,bool) for n in sizes]
        for layer,alternative in forced.items():
            allowed[layer][:]=False;allowed[layer][alternative]=True
        if start is not None:
            allowed[0] &= np.arange(sizes[0])==start
        reachable=allowed[0].copy();back=[]
        for layer, matrix in enumerate(edges):
            candidates=matrix & reachable[:,None] & allowed[layer+1][None,:]
            reachable=candidates.any(0)
            back.append(np.argmax(candidates,axis=0))
        if closed_edges is not None:
            reachable &= closed_edges[:,start]
        if reachable.any():
            chosen=[int(np.flatnonzero(reachable)[0])]
            for predecessor in back[::-1]:chosen.append(int(predecessor[chosen[-1]]))
            return chosen[::-1]
    return None


def alternative_nodes(case, alt, child, side, rep, index):
    graph=case['graph'];curve=graph.curve(alt['profile']);fragments=graph.witness_fragments(alt)
    nodes=[];ratios=[]
    for fragment in fragments:
        canonical=int(rep[fragment]);slot=index[canonical]*2+side
        target_curve=graph.curve(graph.profile(canonical));lo,hi=graph.spans[canonical,:2]
        projected,_=target_curve.project(curve.at(np.asarray([alt['lo'],alt['hi']])))
        ca,cb=sorted(projected.tolist())
        if alt['closed']:left,right=float(lo),float(hi)
        elif alt['wrap']:
            if hi<=ca:left,right=float(lo),min(float(hi),ca)
            elif lo>=cb:left,right=max(float(lo),cb),float(hi)
            else:raise ValueError('Wrapped alternative cannot occupy a fragment once')
        else:left,right=max(float(lo),ca),min(float(hi),cb)
        if not right>left:raise ValueError('No positive finite interval after canonical projection')
        value=category(child['weld_type'],alt['forward'],bool(alt['closed'] and len(fragments)==1),False)
        nodes.append(dict(fragment=canonical,side=side,probability=1.,lo=left,hi=right,
                          object_row=slot,**attributes(value)))
        ratios.append([(left-lo)/(hi-lo),(right-lo)/(hi-lo)])
    return nodes,ratios


def compile_case(key,dest):
    case=load_case(key,supervision=False);graph=case['graph']
    target_file=OUT/f'run264/source_{key}_targets.json'
    registered=next(r for r in read(OUT/'run264/receipt.json')['cases'] if r['source']==key)
    assert digest(target_file)==registered['compiled_sha256']
    target=read(target_file)
    old=read(OUT/f'run285/source_{key}_joint_witnesses.json')['witnesses']
    capacity=next(r for r in read(OUT/'run285/receipt.json')['cases'] if r['source']==key)
    rejected={(r['parent'],r['child'],r['alternative']) for r in capacity['rejected_witnesses']}
    cohort=next(r for r in read(OUT/'run98/local_cohort.json')['cases'] if r['source_key']==key)
    assert digest(OUT/cohort['target_file'])==cohort['target_sha256']
    native=read(OUT/cohort['target_file'])
    requests=read(OUT/f'run312/source_{key}_input_requests.json')
    with np.load(OUT/f'run288/source_{key}_alias_map.npz') as z:rep=z['fragment_representative']
    index={int(f):i for i,f in enumerate(case['canonical'])}
    events=[];unresolved_rows=set();path_records=[];edges_true=defaultdict(set)
    parent_receipts=[];failures=[]
    for pi,parent in enumerate(target['parents']):
        layers=[];old_parent=old[pi];assert old_parent['parent']==pi
        for ci,child in enumerate(parent['children']):
            side=int(old_parent['children'][ci]['side_hypothesis']);layer=[]
            for ai,alt in enumerate(child['alternatives']):
                if (pi,ci,ai) in rejected:continue
                try:
                    nodes,ratios=alternative_nodes(case,alt,child,side,rep,index)
                    if any(not pair_features(a,b,graph)[1][0] for a,b in zip(nodes[:-1],nodes[1:])):
                        raise ValueError('Alternative internal SAME_CHILD is illegal')
                    layer.append(dict(ai=ai,alt=alt,nodes=nodes,ratios=ratios))
                except ValueError as error:
                    # Preserve every original compatible fragment as unresolved.
                    for f in graph.witness_fragments(alt):unresolved_rows.add(index[int(rep[f])]*2+side)
                    failures.append(dict(parent=pi,child=ci,alternative=ai,reason=str(error)))
            layers.append(layer)
        if any(not layer for layer in layers):
            raise ValueError('Original parent lost all alternatives: '+str((key,pi)))
        original_path=[]
        for ci,layer in enumerate(layers):
            matches=[i for i,item in enumerate(layer) if item['ai']==old_parent['children'][ci]['alternative']]
            if len(matches)!=1:raise ValueError('Original certified witness absent from layer')
            original_path.append(matches[0])
        matrices=[]
        for ci in range(len(layers)-1):
            kind={'SHARED':1,'POSITIVE_GAP':2}[parent['gaps'][ci]]
            matrix=np.zeros((len(layers[ci]),len(layers[ci+1])),bool)
            for i,a in enumerate(layers[ci]):
                for j,b in enumerate(layers[ci+1]):
                    matrix[i,j]=pair_features(a['nodes'][-1],b['nodes'][0],graph)[1][kind]
            matrices.append(matrix)
        closure=None
        if parent['closed'] and len(layers)>1:
            closure=np.zeros((len(layers[-1]),len(layers[0])),bool)
            for i,a in enumerate(layers[-1]):
                for j,b in enumerate(layers[0]):
                    closure[i,j]=pair_features(a['nodes'][-1],b['nodes'][0],graph)[1][1]
        certified={};valid=[set() for _ in layers];no_path=0;unproved=0
        for ci,layer in enumerate(layers):
            for ai,alternative in enumerate(layer):
                # Always retain the already certified complete original witness.
                # A first DP path can fail nonlocal whole-parent constraints;
                # that is not evidence that the original child has no path.
                path=original_path.copy() if original_path[ci]==ai else layered_path([len(v) for v in layers],matrices,{ci:ai},closure)
                if path is None:
                    no_path+=1;unresolved_rows.update(n['object_row'] for n in alternative['nodes']);continue
                token=tuple(path)
                if token not in certified:
                    selected=[layers[c][a] for c,a in enumerate(path)]
                    nodes=[n for a in selected for n in a['nodes']]
                    collisions=competition_resources(nodes,graph,requests)
                    duplicate_rows=len({n['object_row'] for n in nodes})!=len(nodes)
                    dec=decode_parent([a['alt'] for a in selected],parent,graph)
                    single=dict(native,parents=[native['parents'][pi]],original_children=parent['child_count'])
                    score=score_native_parents(dec,single,[False])['categories']['overall']
                    passed=not collisions and score['TP']==parent['child_count'] and score['FP']==0
                    certified[token]=passed
                    path_records.append(dict(parent=pi,path=path,original_alternatives=[a['ai'] for a in selected],
                                             passed=passed,competition_collisions=len(collisions),
                                             repeated_object_rows_allowed_for_disjoint_instances=duplicate_rows,score=score,
                                             instance_chain=[dict(object_row=n['object_row'],owner=[pi,c],lo=n['lo'],hi=n['hi'])
                                                             for c,a in enumerate(selected) for n in a['nodes']] if passed else []))
                    if passed:
                        chain=[];child_indices=[]
                        for c,a in enumerate(selected):
                            valid[c].add(path[c]);chain.extend(n['object_row'] for n in a['nodes']);child_indices.extend([c]*len(a['nodes']))
                        for k,(a,b) in enumerate(zip(chain[:-1],chain[1:])):
                            kind=1 if child_indices[k]==child_indices[k+1] else {'SHARED':2,'POSITIVE_GAP':3}[parent['gaps'][child_indices[k]]]
                            edges_true[a,b].add(kind)
                        if parent['closed'] and len(chain)>1:edges_true[chain[-1],chain[0]].add(1 if len(layers)==1 else 2)
                if not certified[token]:
                    unproved+=1;unresolved_rows.update(n['object_row'] for n in alternative['nodes'])
        for ci,indices in enumerate(valid):
            if not indices:raise ValueError('No certified complete witness for original child: '+str((key,pi,ci)))
            for ai in sorted(indices):
                item=layers[ci][ai];weight=1./len(indices)/len(item['nodes'])
                for node,q in zip(item['nodes'],item['ratios']):
                    value=category(node['weld_type'],node['forward'],node['closed'],node['wrap'])
                    events.append(dict(row=node['object_row'],parent=pi,child=ci,alternative=item['ai'],
                                       label=value,ratio=q,weight=weight))
        parent_receipts.append(dict(parent=pi,children=parent['child_count'],
            complete_witnesses_checked=len(certified),certified_witnesses=sum(certified.values()),
            certified_alternatives=sum(len(v) for v in valid),no_path_alternatives=no_path,unproved_alternatives=unproved))
        if pi%20==0:print('COMPATIBLE_PARENT_SUPERVISION',key,pi+1,len(target['parents']),'EVENTS',len(events),flush=True)
    # One original child's alternatives jointly have exactly unit weight.
    sums=defaultdict(float)
    for e in events:sums[e['parent'],e['child']]+=e['weight']
    assert len(sums)==target['G'] and all(abs(v-1.)<1e-10 for v in sums.values())
    by_row=defaultdict(list)
    for event in events:by_row[event['row']].append(event)
    y=np.zeros(len(case['x']),np.int16);weights=np.ones(len(y),np.float64)
    for row in unresolved_rows:y[row]=-1;weights[row]=0.
    conflicts=[];feature_groups=defaultdict(list)
    for row,values in by_row.items():
        labels={v['label'] for v in values}
        if len(labels)>1:conflicts.append(dict(type='ROW_CATEGORY_MULTIMODE',row=row,labels=sorted(labels)))
        y[row]=min(labels);weights[row]=sum(v['weight'] for v in values)
        feature_groups[case['x'][row].tobytes()].append((row,values))
    for group in feature_groups.values():
        labels={v['label'] for _,values in group for v in values}
        if len(labels)>1:conflicts.append(dict(type='IDENTICAL_INPUT_CATEGORY_MULTIMODE',rows=[r for r,_ in group],labels=sorted(labels)))
        # Each endpoint mode is measured in mm on its own finite fragment.
        # A spread exceeding the 3 mm contract gate cannot be hidden by averaging.
        all_ratios=np.asarray([v['ratio'] for _,values in group for v in values])
        widths=[float(np.diff(graph.spans[int(case['canonical'][r//2]),:2])[0]) for r,_ in group]
        span=float(np.ptp(all_ratios,axis=0).max()*max(widths))
        if span>3.:conflicts.append(dict(type='IDENTICAL_INPUT_ENDPOINT_MULTIMODE',rows=[r for r,_ in group],maximum_endpoint_mode_spread_mm=span))
    # Every identical input row is examined, not only already selected positives.
    for row,vector in enumerate(case['x']):
        if y[row]==0 and vector.tobytes() in feature_groups:
            conflicts.append(dict(type='IDENTICAL_INPUT_COMPATIBLE_NEGATIVE_CONFLICT',row=row))
    link_conflicts=[dict(rows=list(k),labels=sorted(v)) for k,v in edges_true.items() if len(v)>1]
    np.savez(dest/f'source_{key}_supervision.npz',labels=y,object_weights=weights,
        endpoint_rows=np.asarray([e['row'] for e in events],np.int64),
        endpoint_ratios=np.asarray([e['ratio'] for e in events],np.float64),
        endpoint_weights=np.asarray([e['weight'] for e in events],np.float64))
    write(dest/f'source_{key}_compatible_events.json',dict(events=events,
        certified_positive_edges=[dict(rows=list(k),labels=sorted(v)) for k,v in edges_true.items()],
        unresolved_rows=sorted(unresolved_rows),side_truth_verified=False))
    write(dest/f'source_{key}_complete_path_proofs.json',path_records)
    receipt=dict(source=key,G=target['G'],parents=len(target['parents']),object_rows=len(y),
        compatible_positive_rows=int((y>0).sum()),known_negative_rows=int((y==0).sum()),
        unresolved_rows=int((y<0).sum()),endpoint_events=len(events),original_child_weight_sum=float(sum(sums.values())),
        conflicts=conflicts,link_class_conflicts=link_conflicts,parent_receipts=parent_receipts,
        rejected_alternative_diagnostics=failures,new_global_capacity_searches=0,new_fit_calls=0,
        status='COMPATIBLE_SUPERVISION_CONSISTENT' if not conflicts and not link_conflicts else 'COMPATIBLE_SUPERVISION_MULTIMODE_OR_CONFLICT')
    write(dest/f'source_{key}_receipt.json',receipt)
    print('COMPATIBLE_SUPERVISION_CASE_COMPLETE',key,'POSITIVE',receipt['compatible_positive_rows'],'CONFLICTS',len(conflicts),flush=True)
    return receipt


def main():
    import argparse,traceback
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run313');args=ap.parse_args()
    start=time.perf_counter();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    assert read(OUT/'run312/receipt.json')['status']=='INPUT_COMPETITION_AND_ORIGINAL_WITNESSES_PASS'
    write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),sources=list(KEYS),new_fit_calls=0,
        no_global_capacity_search=True,all_positive_requires_complete_parent_witness=True,
        alternative_weight_sum_per_original_child=1.,DEV_labels_read=0))
    rows=[]
    try:
        for key in KEYS:
            check_resources();row=compile_case(key,dest);rows.append(row)
            write(dest/'progress.json',dict(cases=rows));gc.collect()
    except BaseException:
        write(dest/'failure.json',dict(error=traceback.format_exc(),completed_sources=[r['source'] for r in rows],
              new_fit_calls=0,seconds=time.perf_counter()-start));raise
    passed=all(r['status']=='COMPATIBLE_SUPERVISION_CONSISTENT' for r in rows)
    write(dest/'receipt.json',dict(status='COMPATIBLE_SUPERVISION_READY' if passed else 'COMPATIBLE_SUPERVISION_GATE_FAILED',
        cases=rows,G=sum(r['G'] for r in rows),parents=sum(r['parents'] for r in rows),
        new_fit_calls=0,new_global_capacity_searches=0,DEV_labels_read=0,
        extraction_correct_rate=None,extraction_precision_mm=None,seconds=time.perf_counter()-start))
    print('COMPATIBLE_SUPERVISION_GATE_COMPLETE',passed,time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
