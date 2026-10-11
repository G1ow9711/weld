"""Aggregate certified TRAIN fragment instances into whole-carrier intervals.

Retain per-instance categories, original owner identities for supervision only,
and unknown complete-set counts. No DEV target access or oracle candidates.
"""
import argparse
from collections import defaultdict
import gc
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_whole_joint_features_1228 import SCHEMA

TRAIN=('21','BM79A_native_local','10')


def prepare(key,dest):
    if key not in TRAIN:raise ValueError('This pilot may only read the registered three TRAIN sources')
    begin=time.perf_counter();case=load_case(key);graph=case['graph'];n=len(graph.profiles)
    source=OUT/'run318';label_path=source/f'source_{key}_count_targets.npz';instance_path=source/f'source_{key}_interval_instances.json'
    bindings=next(c for c in read(OUT/'run342/receipt.json')['cases'] if c['source']==key)
    assert digest(label_path)==bindings['labels_sha256'] and digest(instance_path)==bindings['instances_sha256']
    with np.load(label_path) as z:old_labels=z['labels']
    raw=read(instance_path);canonical=np.asarray(case['canonical'])
    pids=np.searchsorted(graph.offsets,canonical,side='right')-1
    # A complete whole-carrier label requires every original fragment on that
    # carrier to have a canonical, complete side label. Aliased missing pieces
    # remain unknown; they are never silently converted into NO_WELD examples.
    known_count=np.zeros(n*2,int);canonical_count=np.bincount(pids,minlength=n)
    for side in (0,1):np.add.at(known_count,pids*2+side,old_labels[side::2]>=0)
    required=np.repeat(np.diff(graph.offsets),2)
    complete=(known_count==required)&(required>0)
    owners=defaultdict(lambda:defaultdict(list))
    for event in raw['instances']:
        row=int(event['row']);fragment=int(canonical[row//2]);pid=int(pids[row//2]);side=row%2
        lo,hi=graph.spans[fragment,:2];q=lo+np.array(event['ratio'])*(hi-lo)
        owners[pid*2+side][tuple(event['owner'])].append(dict(lo=float(q[0]),hi=float(q[1]),
            category=int(event['label']),weight=float(event['original_explanation_weight']),original_row=row,
            count_is_known=bool(event['count_is_known'])))
    labels=np.full(n*2,-1,np.int32);labels[complete]=0;instances=[];conflicts=[]
    for row,mapping in sorted(owners.items()):
        pid=row//2;length=graph.curve(pid).length;items=[]
        for owner,pieces in mapping.items():
            categories=sorted({p['category'] for p in pieces})
            if len(categories)!=1:
                conflicts.append(dict(reason='OWNER_CATEGORY_CONFLICT',row=row,owner=list(owner),categories=categories))
                complete[row]=False;labels[row]=-1;continue
            lo=min(p['lo'] for p in pieces);hi=max(p['hi'] for p in pieces)
            assert 0<=lo<hi<=length+1e-8
            # Fixed numerical roundoff normalization only, not endpoint repair.
            q=np.array([lo,hi])/length
            assert q.min()>=-1e-12 and q.max()<=1+1e-12
            items.append(dict(row=row,owner=list(owner),category=categories[0],ratio=q.tolist(),
                length_mm=length,fragment_explanations=len(pieces),source_rows=sorted({p['original_row'] for p in pieces}),
                original_explanation_weight=sum(p['weight'] for p in pieces)))
        items.sort(key=lambda x:(x['ratio'][0],x['ratio'][1],x['category'],x['owner']))
        if complete[row]:labels[row]=len(items)
        for k,item in enumerate(items,1):item.update(k=k,count=len(items),count_is_known=bool(complete[row]));instances.append(item)
    # Normalize endpoint explanations per original child, so many fragments or
    # alternative carriers do not silently multiply that child's supervision.
    totals=defaultdict(float)
    for event in instances:
        if event['count_is_known']:totals[tuple(event['owner'])]+=event['original_explanation_weight']
    for event in instances:
        event['weight']=event['original_explanation_weight']/totals[tuple(event['owner'])] if event['count_is_known'] else 0.
    case['x']._mmap.close()
    features=np.load(dest/f'source_{key}/joint_features.npy',mmap_mode='r');assert features.shape==(n*2,160)
    assert read(dest/f'source_{key}/receipt.json')['schema']==SCHEMA
    # Exact feature/label contradictions are explicit evidence, not a reason
    # to discard rows or modify customer labels.
    exact={};collisions=[]
    for row in np.flatnonzero(labels>=0):
        signature=features[row].tobytes();previous=exact.get(signature)
        if previous is None:exact[signature]=(int(row),int(labels[row]))
        elif previous[1]!=labels[row]:collisions.append(dict(first_row=previous[0],other_row=int(row),counts=[previous[1],int(labels[row])]))
    features._mmap.close()
    directory=dest/f'source_{key}';np.savez(directory/'joint_count_targets.npz',labels=labels)
    write(directory/'joint_interval_instances.json',dict(instances=instances,conflicts=conflicts,GT_identifiers_are_supervision_only=True))
    old_owners={tuple(e['owner']) for e in raw['instances']};known_owners={tuple(e['owner']) for e in instances if e['count_is_known']}
    summary=dict(source=key,profiles=n,rows=n*2,known_positive_rows=int(np.sum(labels>0)),known_negative_rows=int(np.sum(labels==0)),
        unknown_rows=int(np.sum(labels<0)),original_children=len(old_owners),children_with_complete_carrier_supervision=len(known_owners),
        missing_complete_count_owners=[list(o) for o in sorted(old_owners-known_owners)],
        maximum_known_count=int(labels.max(initial=0)),interval_instances=len(instances),
        known_interval_instances=sum(e['count_is_known'] for e in instances),category_conflicts=conflicts,
        exact_feature_count_conflicts=collisions,labels_sha256=digest(directory/'joint_count_targets.npz'),
        instances_sha256=digest(directory/'joint_interval_instances.json'),source_labels_sha256=bindings['labels_sha256'],
        source_instances_sha256=bindings['instances_sha256'],DEV_target_reads=0,STEP_reads=0,fit_calls=0,seconds=time.perf_counter()-begin)
    write(directory/'joint_supervision_receipt.json',summary)
    print('WHOLE_JOINT_SUPERVISION',key,{k:summary[k] for k in ('known_positive_rows','known_negative_rows','unknown_rows','original_children','children_with_complete_carrier_supervision','maximum_known_count')},
          'CATEGORY_CONFLICTS',len(conflicts),'INPUT_CONFLICTS',len(collisions),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run
    assert read(dest/'receipt.json')['status']=='WHOLE_JOINT_INPUTS_COMPLETE'
    result=[]
    for key in TRAIN:result.append(prepare(key,dest));gc.collect()
    write(dest/'joint_supervision_receipt.json',dict(status='WHOLE_JOINT_SUPERVISION_AUDITED',cases=result,TRAIN=list(TRAIN),
        original_G=sum(c['original_children'] for c in result),DEV_target_reads=0,fit_calls=0,full_contract_pass=False))


if __name__=='__main__':main()
