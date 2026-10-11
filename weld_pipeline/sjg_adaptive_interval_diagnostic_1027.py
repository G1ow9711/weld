"""TRAIN-only cost/representation diagnostic for input-midpoint refinement.

This is an oracle diagnostic, not a neural decoder or an extraction score.
Split positions are input interval midpoints. Targets choose only the oracle
state and continuous endpoint supervision. Runtime must predict those states.
"""
import argparse
import math
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,utc


def adaptive_tree(lo,hi,segments,leaf_width=8.,max_depth=48):
    stack=[(float(lo),float(hi),0)];nodes=0;mixed=0;leaves=[];query_positions=set()
    while stack:
        left,right,depth=stack.pop();nodes+=1
        # Endpoints and the parent's midpoint recur in child cells. Cache
        # exactly equal parameters; no tolerance-based point merging.
        query_positions.update((left,(left+right)/2,right))
        relevant=[s for s in segments if min(right,s[1])-max(left,s[0])>1e-9]
        if not relevant:continue
        inside=len(relevant)==1 and relevant[0][0]<=left and relevant[0][1]>=right
        if inside:
            leaves.append((left,right,relevant[0][2]));continue
        mixed+=1
        if right-left<=leaf_width and len(relevant)==1:
            s=relevant[0];leaves.append((max(left,s[0]),min(right,s[1]),s[2]));continue
        if depth>=max_depth:raise RuntimeError('Unresolved short/multiple child cell; no silent deletion')
        mid=(left+right)/2
        if not left<mid<right:raise RuntimeError('Numerically indivisible cell')
        stack.extend([(mid,right,depth+1),(left,mid,depth+1)])
    return dict(nodes=nodes,mixed_nodes=mixed,leaves=leaves,query_positions=query_positions)


def controls():
    cases=[[(20,20.1,'short')],[(0,50,'a'),(50,100,'b')],[(0,40,'a'),(60,100,'b')],
           [(1,1.1,'a'),(1.11,1.2,'b'),(1.21,1.3,'c')]]
    for segments in cases:
        tree=adaptive_tree(0,100,segments)
        for lo,hi,identity in segments:
            rows=sorted((a,b) for a,b,i in tree['leaves'] if i==identity)
            assert rows and abs(rows[0][0]-lo)<1e-9 and abs(rows[-1][1]-hi)<1e-9
            assert all(abs(a[1]-b[0])<1e-9 for a,b in zip(rows,rows[1:]))
            assert abs(sum(b-a for a,b in rows)-(hi-lo))<1e-9
    return dict(short_segments_below_8mm_representable=True,shared_endpoint_child_identities_preserved=True,
                positive_gaps_preserved=True,multiple_children_in_one_coarse_cell_refined=True,
                oracle_labels_not_runtime_decisions=True,no_extraction_accuracy_claim=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    write(dest/'registered_protocol.json',dict(stage='TRAIN_ADAPTIVE_COST_ORACLE_ONLY',source_domains='run201',
        maximum_leaf_width_mm=8.,split_position='input interval midpoint',TRAIN_G=363,
        heldout_labels_not_read=True,model_predictions=False,optimizer_updates=0,start_utc=utc()))
    write(dest/'controls.json',controls());rows=[]
    for case in read(OUT/'run186/training_manifest.json')['cases']:
        key=case['source_key'];domain=read(OUT/f'run201/source_{key}_joint_domain.json')
        sup=read(OUT/f'run186/source_{key}_supervision.json')
        with np.load(OUT/f'run201/source_{key}_joint_mapping.npz',allow_pickle=False) as z:
            mapping=z['profile_family'];transforms=z['local_to_canonical']
        by_family={}
        for profile,events in sup['by_profile'].items():
            p=int(profile);fid=int(mapping[p]);start,sign,length=transforms[p];family=domain['families'][fid]
            children=by_family.setdefault(fid,{})
            for event in events:
                child=event['original_child_id'];error=event['maximum_bidirectional_sample_error_mm']
                if child in children and children[child][0]<=error:continue
                pieces=[]
                for a,b in event['intervals']:
                    lo,hi=sorted((float(start+sign*a),float(start+sign*b)))
                    circumference=family['circumference_mm']
                    if circumference:
                        if hi-lo>=circumference-1e-8:pieces.append((0.,circumference,child));continue
                        span=hi-lo;lo%=circumference;hi=lo+span
                        if hi>circumference:pieces.extend([(lo,circumference,child),(0.,hi-circumference,child)]);continue
                    pieces.append((lo,hi,child))
                children[child]=(error,pieces)
        nodes=0;leaves=0;overlap_conflicts=0;represented=set();unique_queries=0
        for fid,children in by_family.items():
            family_queries=set()
            segments=[s for _,pieces in children.values() for s in pieces]
            # Different GT child hypotheses can overlap after geometric
            # projection. Keep them separate; this exposes ambiguity rather
            # than silently applying an exclusivity constraint.
            conflict=any(a[2]!=b[2] and min(a[1],b[1])-max(a[0],b[0])>1e-7 for i,a in enumerate(segments) for b in segments[i+1:])
            if conflict:
                overlap_conflicts+=1
                batches=[[s] for s in segments]
            else:batches=[segments]
            for batch in batches:
                for lo,hi in domain['families'][fid]['domains_mm']:
                    result=adaptive_tree(lo,hi,batch);nodes+=result['nodes'];leaves+=len(result['leaves'])
                    family_queries.update(result['query_positions'])
                    represented.update(s[2] for s in result['leaves'])
            unique_queries+=len(family_queries)
        row=dict(source_key=key,G=case['original_G'],represented_children=len(represented),compatible_families=len(by_family),
            oracle_nodes=nodes,three_queries_per_node_upper=3*nodes,leaves=leaves,
            exact_cached_query_positions=unique_queries,
            oracle_local_query_budget_pass=unique_queries<=100000,
            conflicting_projection_families=overlap_conflicts,oracle_classification_assumed=True,
            original_G_retained=True)
        assert len(represented)==case['original_G'];rows.append(row);print('ADAPTIVE_ORACLE_COST',row,flush=True)
    write(dest/'receipt.json',dict(status='ADAPTIVE_ORACLE_DIAGNOSTIC_COMPLETE',cases=rows,TRAIN_G=363,
        represented_TRAIN=sum(r['represented_children'] for r in rows),model_accuracy_not_measured=True,
        relationship_and_family_selection_cost_not_included=True,runtime_budget_not_proven=True,
        full_contract_pass=False,optimizer_updates=0,end_utc=utc()))


if __name__=='__main__':main()
