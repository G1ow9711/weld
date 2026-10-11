"""Retain original child identities in TRAIN interval-set supervision."""
import argparse
from collections import defaultdict
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_whole_joint_targets_1229 import TRAIN


def gold_set(groups,scores,n):
    """Best exactly-one-per-original-child ordered nonoverlapping explanation."""
    if not groups:return [],0.
    values=[];links=[];ends=[];selected_categories=[]
    for k,group in enumerate(groups):
        pairs=np.asarray(group['pairs'],dtype=np.int64);starts=pairs//n;current_ends=pairs%n
        categories=group.get('allowed_categories',[group['category']])
        alternatives=np.stack([scores[starts,current_ends,category-1] for category in categories],axis=1)
        best_category=alternatives.argmax(1)
        local=alternatives[np.arange(len(pairs)),best_category]
        selected_categories.append(np.asarray(categories)[best_category])
        if k==0:
            v=local.copy();previous=np.full(len(pairs),-1,int)
        else:
            order=np.argsort(ends[-1],kind='stable');old=values[-1][order]
            maximum=np.maximum.accumulate(old)
            best=np.empty(len(old),int);at=0
            for j in range(len(old)):
                if old[j]>old[at]:at=j
                best[j]=order[at]
            available=np.searchsorted(ends[-1][order],starts,side='right')-1
            previous=np.where(available>=0,best[np.maximum(available,0)],-1)
            v=local+np.where(available>=0,maximum[np.maximum(available,0)],-np.inf)
        values.append(v);links.append(previous);ends.append(current_ends)
    index=int(np.argmax(values[-1]));value=float(values[-1][index])
    if not np.isfinite(value):raise ValueError('No legal complete ordered gold interval set')
    chosen=[]
    for k in range(len(groups)-1,-1,-1):
        group=groups[k];pair=int(group['pairs'][index])
        category=int(selected_categories[k][index])
        chosen.append((pair//n,pair%n,category,float(scores[pair//n,pair%n,category-1])))
        index=int(links[k][index])
    return chosen[::-1],value


def prepare(dest,input_run='run396',pair_run='run398'):
    if pair_run in ('run398','run403') and (OUT/'run404/confirmed_supervision_conflicts.json').exists():
        raise RuntimeError('Inherited pair supervision has confirmed native-D3 positive omissions; do not rebuild identical incomplete set labels')
    from weld_pipeline.sjg_endpoint_role_train_1246 import inputs
    from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
    from weld_pipeline.sjg_recall_set_worker_1162 import target_for
    start=time.perf_counter();summaries=[]
    for key in TRAIN:
        sparse_path=OUT/f'{pair_run}/source_{key}_pair_targets.json';sparse=read(sparse_path)
        binding=next(c for c in read(OUT/f'{pair_run}/receipt.json')['cases'] if c['source']==key)
        assert digest(sparse_path)==binding['targets_sha256']
        x,m,receipt=inputs(key,input_run);x._mmap.close();case=load_case(key);g=case['graph'];case['x']._mmap.close()
        target,tbind=target_for(key);sr=read(OUT/f'run380/source_{key}/joint_supervision_receipt.json')
        source=OUT/f'run380/source_{key}/joint_interval_instances.json'
        assert digest(source)==sr['instances_sha256'];events=defaultdict(list)
        for event in read(source)['instances']:
            if event['count_is_known']:events[event['row']].append(event)
        records=[];masses=defaultdict(float);covered=set();maximum_k=0
        for raw in sparse['records']:
            r={k:v for k,v in raw.items() if not k.startswith('positive_')}
            a=r['q_offset'];n=r['n'];q=m['q_mm'][a:a+n];curve=g.curve(r['row']//2);xyz=curve.at(q);groups=[]
            for event in sorted(events[r['row']],key=lambda v:(v['ratio'][0],v['ratio'][1])):
                owner=tuple(event['owner']);pi,ci=owner
                truth=np.asarray(target['parents'][pi]['children'][ci]['original_points_mm'])[[0,-1]]
                native=curve.at(np.asarray(event['ratio'])*event['length_mm'])
                if np.linalg.norm(native-truth[::-1],axis=1).sum()<np.linalg.norm(native-truth,axis=1).sum():truth=truth[::-1]
                starts=np.flatnonzero(np.linalg.norm(xyz-truth[0],axis=1)<=3)
                ends=np.flatnonzero(np.linalg.norm(xyz-truth[1],axis=1)<=3)
                pairs=sorted({int(i*n+j) for i in starts for j in ends if i<j})
                assert pairs,(key,r['row'],owner,'original child has no legal pair')
                # This establishes that matching costs cannot count two selected
                # nonoverlapping predictions as the same original child.
                pa=np.asarray(pairs)
                assert int((pa//n).max())<int((pa%n).min()),(key,r['row'],owner,'tolerance variants can be disjoint')
                groups.append(dict(owner=list(owner),category=int(event['category']),pairs=pairs,
                    original_weight=float(event['weight'])))
                masses[owner]+=event['weight'];covered.add(owner)
            bypair={}
            for i,group in enumerate(groups):
                for pair in group['pairs']:
                    assert pair not in bypair,(key,r['row'],'same interval can match different original children')
                    bypair[pair]=i
            assert set(bypair)==set(raw['positive_indices'])
            gold_set(groups,np.zeros((n,n,4),np.float32),n)
            r['gold_groups']=groups;maximum_k=max(maximum_k,len(groups));records.append(r)
        assert len(covered)==sparse['original_children']
        negative_rows=sum(not r['gold_groups'] for r in records)
        fn_mass=defaultdict(float)
        for r in records:
            groups=r['gold_groups']
            if groups:
                costs=[v['original_weight']/masses[tuple(v['owner'])]/(6*len(covered)) for v in groups]
                r['row_weight']=sum(costs);r['false_positive_cost']=1/len(groups)
                for v,cost in zip(groups,costs):
                    v['miss_cost']=cost/r['row_weight'];fn_mass[tuple(v['owner'])]+=cost
            else:
                r['row_weight']=1/(6*negative_rows);r['false_positive_cost']=1.
        assert all(abs(v-1/(6*len(covered)))<1e-10 for v in fn_mass.values())
        output=dict(source=key,records=records,original_G=len(covered),covered_G=len(covered),
            all_pairs=sparse['all_pairs'],feature_sha256=sparse['feature_sha256'],map_sha256=sparse['map_sha256'],
            source_supervision_sha256=digest(sparse_path),target_sha256=tbind['target_sha256'],
            positive_row_weight=sum(r['row_weight'] for r in records if r['gold_groups']),
            negative_row_weight=sum(r['row_weight'] for r in records if not r['gold_groups']),
            original_child_missing_cost=1/(6*len(covered)),maximum_gold_count=maximum_k,
            full_gold_sets_feasible=True,no_disjoint_same_owner_alternatives=True,no_pair_multi_owner=True)
        path=dest/f'source_{key}_set_targets.json';write(path,output)
        summaries.append(dict(source=key,G=len(covered),rows=len(records),positive_rows=len(records)-negative_rows,
            negative_rows=negative_rows,maximum_gold_count=maximum_k,target_sha256=digest(path)))
        print('SET_TARGETS_READY',summaries[-1],flush=True)
    write(dest/'receipt.json',dict(status='COMPLETE_ORDERED_TRAIN_INTERVAL_SETS_READY',cases=summaries,G=213,
        fit_calls=0,DEV_reads=0,STEP_reads=0,all_original_children_preserved=True,
        full_contract_pass=False,seconds=time.perf_counter()-start))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--input-run',default='run396');p.add_argument('--pair-run',default='run398')
    a=p.parse_args();d=OUT/a.run;d.mkdir(exist_ok=False)
    write(d/'registered_protocol.json',dict(source_sha256=digest(__file__),TRAIN=list(TRAIN),
        input_run=a.input_run,pair_run=a.pair_run,DEV_reads=0,fit_calls=0))
    prepare(d,a.input_run,a.pair_run)


if __name__=='__main__':main()
