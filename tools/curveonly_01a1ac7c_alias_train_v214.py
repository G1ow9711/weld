"""Same paired learner; certify literal source aliases and keep eachGT case mass."""
import types
from collections import Counter
import numpy as np
from tools import curveonly_01a1ac7c_experiment as e
from tools import curveonly_01a1ac7c_event_mass_train_v201 as previous
m=previous.previous.m
SETTINGS=dict(previous.SETTINGS,supervision='ORIGINAL_PLUS_LITERAL_FINITE_SOURCE_ALIASES',
              instance_mass='EACH_TRUTH_SPLIT_EQUALLY_ACROSS_ITS_CERTIFIED_REPRESENTATIONS')
configure=previous.configure
class_weights=previous.previous.class_weights


def assemble(cases):
    all_rows,details=[],[]
    masses=dict(classification=0.,offset=0.,order=0.,full=0.)
    for case in sorted(cases,key=lambda q:q['source_id']):
        assert case['split']=='TRAIN'
        for key in ('features','geometry','local_targets'):
            assert e.sha(case[key]['path'])==case[key]['sha256']
        packet=e.read(case['geometry']['path'])
        rows=m.candidate_rows(np.load(case['features']['path']),packet)
        choices=case['chosen'];representations=Counter(q['truth'] for q in choices)
        count=len(representations);canonical={q['candidate'] for q in choices}
        prepared,occupancy_weights,choice_rows,full_groups=[],[],[],[]
        nonfull=Counter();full=Counter();order_keys=Counter()
        for row in rows:
            chosen=[q for q in choices if q['candidate']==row['candidate']]
            choice_rows.append(chosen)
            target=m.targets(row,chosen);prepared.append(target)
            n=len(row['grid'])
            row.update(targets=target['fields'].astype(np.float32),class_weight=np.zeros((3,n),np.float32),
                offset_weight=np.zeros((2,n),np.float32),order_weight=np.zeros(n,np.float32),
                full_target=target['full'],full_weight=0.,source_id=case['source_id'])
            pos=np.zeros(n,float);station=np.arange(n)*row['grid_step_mm']
            for choice,code in zip(chosen,target['codes']):
                (full if code['full_cycle'] else nonfull)[choice['truth']]+=1
                take=np.ones(n,bool) if code['full_cycle'] else np.zeros(n,bool)
                if not code['full_cycle']:
                    for turn in ((-1,0,1) if row['closed'] else (0,)):
                        s=station+turn*row['length_mm']
                        take|=(s>=choice['start_mm'])&(s<=choice['end_mm'])
                if not take.any():take[[code['start_index'],code['end_index']]]=True
                pos[take]+=1/(count*representations[choice['truth']]*int(take.sum()))
            occupancy_weights.append(pos)
            keys={}
            for i in np.flatnonzero(target['order_known']):
                left=chosen[int(target['start_instance'][i])]['truth']
                right=chosen[int(target['end_instance'][i])]['truth']
                key=(left,right);order_keys[key]+=1;keys[int(i)]=key
            row['_order_keys']=keys
            full_groups.append(0 if target['full'] else 1 if row['candidate'] in canonical else 2)
        assert not (set(full)&set(nonfull)), 'SameGT full-cycle semantics differ between representations'
        assert abs(sum(float(x.sum()) for x in occupancy_weights)-(1. if count else 0.))<1e-8
        groups=[class_weights(rows,canonical,0,occupancy_weights)]
        for head in (1,2):
            positive=[]
            for row,target,chosen in zip(rows,prepared,choice_rows):
                w=np.zeros(len(row['grid']),float)
                for choice,code in zip(chosen,target['codes']):
                    if not code['full_cycle']:
                        index=code['start_index'] if head==1 else code['end_index']
                        w[index]=1/(len(nonfull)*nonfull[choice['truth']])
                positive.append(w)
            groups.append(class_weights(rows,canonical,head,positive))
        for row,target,chosen in zip(rows,prepared,choice_rows):
            for choice,code in zip(chosen,target['codes']):
                if not code['full_cycle']:
                    for head,index in [(0,code['start_index']),(1,code['end_index'])]:
                        row['offset_weight'][head,index]=1/(2*len(nonfull)*nonfull[choice['truth']])
            for index,key in row.pop('_order_keys').items():
                row['order_weight'][index]=1/(len(order_keys)*order_keys[key])
        closed_counts=np.array([sum(row['closed'] and group==k for row,group in zip(rows,full_groups)) for k in range(3)])
        present=int((closed_counts>0).sum())
        if present:
            for row,group,chosen in zip(rows,full_groups,choice_rows):
                if not row['closed']:continue
                if group==0:
                    assert len(chosen)==1
                    row['full_weight']=1/(present*len(full)*full[chosen[0]['truth']])
                else:row['full_weight']=1/(present*closed_counts[group])
            assert abs(sum(q['full_weight'] for q in rows)-1)<1e-6
        masses['classification']+=1
        masses['offset']+=bool(nonfull)
        masses['order']+=bool(order_keys)
        masses['full']+=present>0
        all_rows.extend(rows)
        details.append(dict(source_id=case['source_id'],candidates=len(rows),complete_intervals=count,
            represented_intervals=len(choices),nonfull_intervals=len(nonfull),full_cycles=len(full),
            foreground_case_mass=float(sum(x.sum() for x in occupancy_weights)),
            class_group_counts=groups,closed_curve_group_counts=closed_counts.tolist(),
            each_true_instance_equal_mass_before_class_group_balance=True,
            original_GT_and_unknown_alternatives_retained=True))
    mean,scale=m.codec.previous.fit_range_normalizer(all_rows)
    assert max(float(np.abs((q['grid']-mean)/scale).max()) for q in all_rows)<=1.00001
    return all_rows,mean,scale,masses,details


namespace=dict(previous.namespace,SETTINGS=SETTINGS,assemble=assemble)
assembly_control=types.FunctionType(previous.assembly_control.__code__,namespace)
fit=types.FunctionType(previous.fit.__code__,namespace)
mathematical_controls=previous.mathematical_controls


def parity_control(cases,expanded,destination):
    old=previous.assemble(cases)
    fresh=assemble(cases)
    assert np.array_equal(old[1],fresh[1]) and np.array_equal(old[2],fresh[2]) and old[3]==fresh[3]
    maximum=0.
    for a,b in zip(old[0],fresh[0]):
        for key in ('grid','targets','class_weight','offset_weight','order_weight'):
            maximum=max(maximum,float(np.abs(a[key]-b[key]).max(initial=0.)))
        maximum=max(maximum,abs(a['full_weight']-b['full_weight']))
    assert maximum<=1e-7
    # Drop copies before assembling the expanded population.
    del old,fresh
    rows,mean,scale,masses,details=assemble(expanded)
    grouped={}
    for row in rows:grouped.setdefault(row['source_id'],[]).append(row)
    for group in grouped.values():
        assert all(abs(sum(float(q['class_weight'][h].sum()) for q in group)-1)<1e-5 for h in range(3))
    e.write(__import__('pathlib').Path(destination)/'ASSEMBLY_PARITY.json',dict(passed=True,
        original_canonical_targets_and_normalizer_unchanged=True,original_weight_max_delta=maximum,
        all_case_classification_masses_equal1=True,masses=masses,details=details,
        TRAIN_inputs=len(cases),no_optimizer_or_model_weight_reads=True))
