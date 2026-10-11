"""Pro314: exact local count and ordinal-conditioned interval supervision.

Only complete known local sets supervise count. Original child identity is a
TRAIN association, never a feature. Alternative witnesses of the same child do
not add count. No incompatible modes are averaged or silently masked.
"""
from collections import defaultdict
import gc
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, check_resources
from weld_pipeline.sjg_joint_forest_train_1109 import load_case
from weld_pipeline.sjg_request_competition_1125 import interval_projection

KEYS=('21','4','10','25','36','42','BM79A_native_local')


def count_category(category, count):
    if not 1 <= category <= 16 or count < 1:
        raise ValueError('Invalid positive category/count')
    return (int(count)-1)*16+int(category)


def decode_count_category(value):
    if int(value) < 1:
        return 0, 0
    return (int(value)-1)%16+1, (int(value)-1)//16+1


def compile_sets(key, dest,compatible_run='run314'):
    data=read(OUT/f'{compatible_run}/source_{key}_compatible_events.json')
    case=load_case(key,supervision=False)
    graph=case['graph']
    by_row=defaultdict(lambda: defaultdict(list))
    for event in data['events']:
        by_row[event['row']][event['parent'],event['child']].append(event)
    raw_unresolved=set(data['unresolved_rows'])
    # Count uncertainty is about additional true instances, not additional
    # explanations of an instance already certified on this same input row.
    compiled=read(OUT/f'run264/source_{key}_targets.json')
    original_witnesses=read(OUT/f'run285/source_{key}_joint_witnesses.json')['witnesses']
    prior=next(r for r in read(OUT/'run285/receipt.json')['cases'] if r['source']==key)
    rejected={(r['parent'],r['child'],r['alternative']) for r in prior['rejected_witnesses']}
    with np.load(OUT/f'run288/source_{key}_alias_map.npz') as z:representative=z['fragment_representative']
    inverse={int(f):i for i,f in enumerate(case['canonical'])}
    possible_owners=defaultdict(set)
    for pi,parent in enumerate(compiled['parents']):
        for ci,child in enumerate(parent['children']):
            side=int(original_witnesses[pi]['children'][ci]['side_hypothesis'])
            for ai,alt in enumerate(child['alternatives']):
                if (pi,ci,ai) in rejected:continue
                for fragment in graph.witness_fragments(alt):
                    row=inverse[int(representative[fragment])]*2+side
                    possible_owners[row].add((pi,ci))
    unresolved_owners={row:owners-set(by_row.get(row,{})) for row,owners in possible_owners.items()
                       if owners-set(by_row.get(row,{}))}
    count_unresolved=set(unresolved_owners)
    # The previously unresolved row can be resolved only if every possible
    # original owner has a certified whole-parent explanation on that row.
    reconciled=[]
    for row in sorted(raw_unresolved-count_unresolved):
        assert possible_owners[row] and possible_owners[row] <= set(by_row.get(row,{}))
        reconciled.append(dict(row=row,possible_owners=[list(v) for v in sorted(possible_owners[row])],
                               all_possible_owners_already_certified=True))
    with np.load(OUT/f'{compatible_run}/source_{key}_supervision.npz') as z:
        old_labels=z['labels']
    labels=np.zeros(len(case['x']),np.int32)
    labels[list(count_unresolved)]=-1
    instances=[]; conflicts=[]; sets=[]; owner_totals=defaultdict(float)
    for row,owners in sorted(by_row.items()):
        row_intervals=[]
        width=float(np.diff(graph.spans[int(case['canonical'][row//2]),:2])[0])
        for owner,alternatives in owners.items():
            # Same GT instance, possibly several certified equivalent witnesses.
            # Select one intact witness target by fixed alternative order, never
            # an average of different parents or endpoint modes.
            values=sorted(alternatives,key=lambda e:(e['alternative'],e['ratio'][0],e['ratio'][1]))
            modes=np.asarray([e['ratio'] for e in values])
            if float(np.ptp(modes,axis=0).max()*width)>3.:
                conflicts.append(dict(type='ONE_INSTANCE_INCOMPATIBLE_WITNESSES',row=row,owner=list(owner)))
            chosen=values[0]
            row_intervals.append(dict(row=row,owner=list(owner),label=chosen['label'],
                ratio=chosen['ratio'],chosen_alternative=chosen['alternative'],
                original_explanation_weight=sum(e['weight'] for e in values),
                alternative_witnesses=len(values),width_mm=width))
        row_intervals.sort(key=lambda e:(e['ratio'][0],e['ratio'][1],e['label']))
        classes={e['label'] for e in row_intervals}
        if len(classes)!=1:
            conflicts.append(dict(type='LOCAL_SET_MIXED_CATEGORY',row=row,classes=sorted(classes)))
        # Coexistence: different interval instances must not claim a positive
        # common span on the same local carrier and side. Boundary uncertainty
        # is explicitly assessed, not hidden by fitting an average.
        overlap=[]
        for a,b in zip(row_intervals[:-1],row_intervals[1:]):
            fragment=int(case['canonical'][row//2]);curve=graph.curve(graph.profile(fragment))
            domain_lo=float(graph.spans[fragment,0])
            a_end=domain_lo+a['ratio'][1]*width;b_start=domain_lo+b['ratio'][0]*width
            length=a_end-b_start
            if curve.kind=='line':
                a_range=interval_projection(curve,a_end,curve.ends[0],curve.direction)
                b_range=interval_projection(curve,b_start,curve.ends[0],curve.direction)
                certified_overlap=a_range[0]-b_range[1]
            else:
                certified_overlap=length-64*np.finfo(float).eps*max(1.,curve.length)
            if certified_overlap>0:
                overlap.append(dict(owners=[a['owner'],b['owner']],overlap_mm=length,
                                    certified_inner_overlap_mm=certified_overlap))
        if overlap:
            conflicts.append(dict(type='LOCAL_INTERVAL_SET_COEXISTENCE_UNPROVED',row=row,overlaps=overlap))
        known=row not in count_unresolved
        if known:
            labels[row]=count_category(min(classes),len(row_intervals))
        for k,event in enumerate(row_intervals,1):
            event.update(k=k,count=len(row_intervals),count_is_known=known)
            if known:owner_totals[tuple(event['owner'])]+=event['original_explanation_weight']
            instances.append(event)
        sets.append(dict(row=row,count=len(row_intervals),count_is_known=known,
                         owners=[e['owner'] for e in row_intervals]))
    # Unresolved local sets never become zero-count observations.
    assert all(labels[row]==-1 for row in count_unresolved)
    known_owners={tuple(e['owner']) for e in instances if e['count_is_known']}
    all_owners={tuple(e['owner']) for e in instances}
    missing=sorted(all_owners-known_owners)
    if missing:conflicts.append(dict(type='ORIGINAL_CHILD_HAS_NO_COMPLETE_COUNT_SUPERVISION',owners=[list(o) for o in missing]))
    for event in instances:
        event['weight']=event['original_explanation_weight']/owner_totals[tuple(event['owner'])] if event['count_is_known'] else 0.
    groups=defaultdict(list)
    for event in instances:
        if event['count_is_known']:
            groups[case['x'][event['row']].tobytes()].append(event)
    for values in groups.values():
        counts={e['count'] for e in values}
        if len(counts)!=1:
            conflicts.append(dict(type='IDENTICAL_INPUT_COUNT_CONFLICT',rows=sorted({e['row'] for e in values}),counts=sorted(counts)))
        slots=defaultdict(list)
        for event in values:slots[event['k']].append(event)
        for k,items in slots.items():
            modes=np.asarray([e['ratio'] for e in items]);spread=float(np.ptp(modes,axis=0).max()*max(e['width_mm'] for e in items))
            if spread>3. or len({e['label'] for e in items})!=1:
                conflicts.append(dict(type='IDENTICAL_INPUT_ORDINAL_CONFLICT',k=k,rows=sorted({e['row'] for e in items}),spread_mm=spread))
    for row,vector in enumerate(case['x']):
        if labels[row]==0 and vector.tobytes() in groups:
            conflicts.append(dict(type='IDENTICAL_INPUT_ZERO_POSITIVE_COUNT_CONFLICT',row=row))
    assert all(labels[row]<0 for row in np.flatnonzero(old_labels<0) if row in count_unresolved)
    weights=np.ones(len(labels),np.float64);weights[labels<0]=0.
    for row in by_row:
        if labels[row]>0:weights[row]=0.
    for event in instances:
        if event['count_is_known']:weights[event['row']]+=event['weight']
    np.savez(dest/f'source_{key}_count_targets.npz',labels=labels,weights=weights)
    write(dest/f'source_{key}_interval_instances.json',dict(instances=instances,sets=sets,
        runtime_GT_identifiers_forbidden=True,unresolved_rows=sorted(count_unresolved),
        reconciled_alternative_only_uncertainty=reconciled,
        unresolved_original_owners={str(row):[list(o) for o in sorted(owners)] for row,owners in unresolved_owners.items()}))
    row=dict(source=key,known_positive_rows=int((labels>0).sum()),known_negative_rows=int((labels==0).sum()),
        unresolved_rows=int((labels<0).sum()),all_interval_instances=len(instances),
        endpoint_training_instances=sum(e['count_is_known'] for e in instances),
        maximum_known_count=max((e['count'] for e in instances if e['count_is_known']),default=0),
        maximum_unresolved_lower_bound=max((e['count'] for e in instances if not e['count_is_known']),default=0),
        G=len(all_owners),original_children_with_known_supervision=len(known_owners),
        reconciled_alternative_only_uncertainty=len(reconciled),
        total_endpoint_weight=sum(e['weight'] for e in instances),conflicts=conflicts)
    write(dest/f'source_{key}_receipt.json',row)
    print('ORDERED_INTERVAL_SET_AUDIT',key,'KNOWN',row['known_positive_rows'],'K',row['maximum_known_count'],
          'INSTANCES',row['endpoint_training_instances'],'CONFLICTS',len(conflicts),flush=True)
    return row


def reindex_original_witnesses(key,dest,K):
    data=read(dest/f'source_{key}_interval_instances.json')
    mapping={(e['row'],tuple(e['owner'])):e for e in data['instances']}
    case=load_case(key,supervision=True)
    index={int(f):i for i,f in enumerate(case['canonical'])}
    with np.load(OUT/f'run288/source_{key}_alias_map.npz') as z:rep=z['fragment_representative']
    rows=[];collisions=[];occupied=set();unknown=[]
    for parent in read(OUT/f'run285/source_{key}_joint_witnesses.json')['witnesses']:
        chain=[]
        for child in parent['children']:
            owner=(parent['parent'],child['child'])
            for fragment in child['fragments']:
                row=index[int(rep[fragment])]*2+child['side_hypothesis']
                event=mapping.get((row,owner))
                if event is None:raise ValueError('Original witness instance missing after set compilation')
                instance=(row,event['k'])
                if instance in occupied:collisions.append(dict(instance=list(instance),owner=list(owner)))
                occupied.add(instance)
                if not event['count_is_known']:unknown.append(dict(instance=list(instance),owner=list(owner)))
                assert 1<=event['k']<=K,'TRAIN original witness exceeds registered capacity'
                chain.append(dict(object_row=row,k=event['k'],owner=list(owner),original_node=case['teacher'][row]))
        rows.append(dict(parent=parent['parent'],instances=chain,gaps=parent['gaps'],closed=parent['closed']))
    result=dict(source=key,parents=len(rows),G=case['G'],witnesses=rows,collisions=collisions,
                count_unresolved_original_instances=unknown,original_geometry_unchanged=True,
                original_sides_and_gaps_unchanged=True,new_capacity_searches=0)
    write(dest/f'source_{key}_original_instance_witnesses.json',result)
    return dict(source=key,G=case['G'],parents=len(rows),collisions=len(collisions),
                original_instances_with_unresolved_count=len(unknown))


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run315');ap.add_argument('--compatible-run',default='run314');args=ap.parse_args()
    start=time.perf_counter();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    assert read(OUT/'run312/receipt.json')['status']=='INPUT_COMPETITION_AND_ORIGINAL_WITNESSES_PASS'
    assert read(OUT/args.compatible_run/'receipt.json')['G']==363
    write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),sources=list(KEYS),new_fit_calls=0,
        K_rule='maximum complete known TRAIN local count; unresolved lower bound checked, never silently clipped',
        endpoint_input_columns=129,DEV_labels_read=0,no_endpoint_average=True,compatible_supervision_run=args.compatible_run))
    rows=[]
    for key in KEYS:
        check_resources();rows.append(compile_sets(key,dest,args.compatible_run));gc.collect()
        write(dest/'progress.json',dict(cases=rows))
    K=max(r['maximum_known_count'] for r in rows)
    over=[r['source'] for r in rows if r['maximum_unresolved_lower_bound']>K]
    if over:
        witnesses=[]
    else:
        witnesses=[reindex_original_witnesses(key,dest,K) for key in KEYS]
    passed=not over and all(not r['conflicts'] for r in rows) and all(not r['collisions'] for r in witnesses)
    write(dest/'receipt.json',dict(status='ORDERED_INTERVAL_SUPERVISION_READY' if passed else 'ORDERED_INTERVAL_SUPERVISION_GATE_FAILED',
        cases=rows,witness_reindex=witnesses,K=K,unresolved_capacity_exceeds_K=over,
        E=sum(r['endpoint_training_instances'] for r in rows),G=sum(r['G'] for r in rows),
        new_fit_calls=0,DEV_labels_read=0,new_global_capacity_searches=0,
        full_contract_pass=False,extraction_correct_rate=None,extraction_precision_mm=None,
        seconds=time.perf_counter()-start))
    print('ORDERED_INTERVAL_SUPERVISION_GATE_COMPLETE',passed,'K',K,time.perf_counter()-start,flush=True)


if __name__=='__main__':main()
