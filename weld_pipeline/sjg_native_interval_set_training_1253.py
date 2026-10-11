"""Train local interval sets from independently checked native compatibility.

Geometry directions are alternatives under the current D3 whole-parent reversal
rule. Parent parity metadata remains explicit for subsequent full-parent assembly.
"""
import argparse
from collections import defaultdict
import gc
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_endpoint_role_train_1246 import inputs
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_interval_set_targets_1249 import gold_set

TRAIN=('21','BM79A_native_local','10')


class ZeroScores:
    def __getitem__(self,index):return np.zeros(np.broadcast_arrays(index[0],index[1])[0].shape)


def prepare(dest,train=TRAIN,native_run='run406',input_run='run402',expected_g=213):
    audit=read(OUT/native_run/'independent_audit_summary.json')
    assert audit['status']=='PASS' and audit['original_G']==expected_g
    assert {c['source'] for c in audit['cases']}==set(train)
    balance=2*len(train)
    cases=[]
    for key in train:
        src=OUT/f'{native_run}/source_{key}';sr=read(src/'receipt.json')
        assert digest(src/'compatibility.json')==sr['compatibility_sha256'] and digest(src/'pairs.npz')==sr['pairs_sha256']
        relation=read(src/'compatibility.json')
        with np.load(src/'pairs.npz') as z:pair_indices=z['indices']
        eligible={r['row'] for r in relation['rows'] if r['complete_background_supervision_allowed']}
        x,m,ir=inputs(key,input_run);x._mmap.close()
        assert ir['feature_sha256']==sr['feature_sha256'] and ir['map_sha256']==sr['map_sha256']
        case=load_case(key);g=case['graph'];case['x']._mmap.close()
        profiles={p['profile']:p for p in relation['profiles']};records=[];owner_rows=defaultdict(int)
        for i,row in enumerate(m['rows']):
            row=int(row)
            if row not in eligible:continue
            a,b=map(int,m['offsets'][i:i+2]);n=b-a;q=m['q_mm'][a:b];curve=g.curve(row//2)
            by_owner=defaultdict(lambda:defaultdict(set));parities=defaultdict(lambda:defaultdict(set))
            for entry in profiles[row//2]['groups']:
                assert not entry['implicit_all_pairs'],'Full-circle implicit labels require a 17-class successor; no child may be silently dropped'
                pairs=pair_indices[entry['pair_start']:entry['pair_stop']]
                oi=entry['owner_index'];native=entry['native_category'];reverse=entry['reversed_parent_category']
                by_owner[oi][native].update(map(int,pairs));by_owner[oi][reverse].update(map(int,pairs))
                parities[oi][native].add(False);parities[oi][reverse].add(True)
            groups=[]
            for oi,categories in by_owner.items():
                union=set().union(*categories.values())
                assert all(pairs==union for pairs in categories.values()),'Pair-specific orientation masks must be represented, not discarded'
                allowed=sorted(categories);assert set(allowed)<=set(range(1,5))
                pairs=np.asarray(sorted(union),np.int64);assert (pairs//n).max()<(pairs%n).min()
                owner=relation['owners'][oi];owner_key=(owner['parent'],owner['child'])
                groups.append(dict(owner=list(owner_key),category=allowed[0],allowed_categories=allowed,pairs=pairs.tolist(),
                    category_parent_parities={str(c):sorted(v) for c,v in parities[oi].items()},
                    sort_key=[float(np.median(q[pairs//n])),float(np.median(q[pairs%n]))]))
                owner_rows[owner_key]+=1
            groups.sort(key=lambda v:v.pop('sort_key'))
            gold_set(groups,ZeroScores(),n)
            records.append(dict(source=key,row=row,q_offset=a,feature_offset=2*a,n=n,
                curve_length_mm=curve.length,is_closed=bool(curve.kind=='arc' and curve.arc.is_closed),
                pair_count=n*(n-1)//2,gold_groups=groups))
        assert len(owner_rows)==sr['G']
        negative=sum(not r['gold_groups'] for r in records);mass=defaultdict(float)
        for r in records:
            groups=r['gold_groups']
            if groups:
                costs=[1/(balance*sr['G']*owner_rows[tuple(v['owner'])]) for v in groups]
                r['row_weight']=sum(costs);r['false_positive_cost']=1/len(groups)
                for group,cost in zip(groups,costs):
                    group['miss_cost']=cost/r['row_weight'];mass[tuple(group['owner'])]+=cost
            else:r['row_weight']=1/(balance*negative);r['false_positive_cost']=1.
        assert all(abs(value-1/(balance*sr['G']))<1e-10 for value in mass.values())
        output=dict(source=key,records=records,original_G=sr['G'],covered_G=len(owner_rows),
            all_pairs=sum(v['pair_count'] for v in records),feature_sha256=ir['feature_sha256'],map_sha256=ir['map_sha256'],
            native_compatibility_sha256=sr['compatibility_sha256'],native_pairs_sha256=sr['pairs_sha256'],
            original_target_sha256=sr['target_sha256'],full_gold_sets_feasible=True,
            no_disjoint_same_owner_alternatives=True,no_pair_multi_owner=sr['multiple_owner_geometry_pairs']==0,
            original_child_miss_cost=1/(balance*sr['G']),unknown_background_rows_not_used=True,
            geometry_orientation_alternatives=True,whole_parent_parity_still_required=True,
            final_parent_assembly_not_supervised_here=True,
            positive_row_weight=sum(v['row_weight'] for v in records if v['gold_groups']),
            negative_row_weight=sum(v['row_weight'] for v in records if not v['gold_groups']))
        path=dest/f'source_{key}_set_targets.json';write(path,output)
        record=dict(source=key,G=sr['G'],rows=len(records),positive_rows=len(records)-negative,negative_rows=negative,
            maximum_gold_count=max(len(v['gold_groups']) for v in records),target_sha256=digest(path))
        cases.append(record);print('NATIVE_INTERVAL_SETS_READY',record,flush=True);gc.collect()
    assert sum(v['G'] for v in cases)==expected_g
    write(dest/'receipt.json',dict(status='COMPLETE_ORDERED_TRAIN_INTERVAL_SETS_READY',cases=cases,G=expected_g,
        input_run=input_run,native_run=native_run,TRAIN=list(train),
        source='independently audited original native D3 compatibility',fit_calls=0,DEV_reads=0,STEP_reads=0,
        whole_parent_parity_preserved_as_metadata=True,local_geometry_labels_not_full_parent_acceptance=True,
        full_contract_pass=False))


def alias_controls(dest):
    import torch
    from weld_pipeline.sjg_structured_interval_set_1251 import structured_choices
    rng=np.random.default_rng(987);n=5
    groups=[dict(pairs=[1,2],category=1,allowed_categories=[1,3],miss_cost=.6),
        dict(pairs=[2*n+4,3*n+4],category=2,allowed_categories=[2,4],miss_cost=.4)]
    record=dict(gold_groups=groups,false_positive_cost=.5)
    lookup={(pair//n,pair%n,c):k for k,g in enumerate(groups) for pair in g['pairs'] for c in g['allowed_categories']}
    def enumerate_sets(start):
        yield []
        for i in range(start,n-1):
            for j in range(i+1,n):
                for c in (1,2,3,4):
                    for tail in enumerate_sets(j):yield [(i,j,c)]+tail
    sets=list(enumerate_sets(0));maximum=0.
    for _ in range(8):
        scores=rng.normal(size=(n,n,4)).astype(np.float32)
        _,_,hinge=structured_choices(scores,record);best_aug=-np.inf;best_gold=-np.inf
        for edges in sets:
            owners=[lookup[e] for e in edges if e in lookup];assert len(owners)==len(set(owners))
            value=sum(float(scores[i,j,c-1]) for i,j,c in edges)
            cost=.5*sum(e not in lookup for e in edges)+sum(g['miss_cost'] for k,g in enumerate(groups) if k not in owners)
            best_aug=max(best_aug,value+cost)
            if len(edges)==2 and set(owners)=={0,1}:best_gold=max(best_gold,value)
        delta=abs(hinge-(best_aug-best_gold));maximum=max(maximum,delta);assert delta<2e-5
    # A high-scoring reversed singleton geometry is a valid alternative,
    # rather than a false positive solely because of its traversal direction.
    scores=np.full((3,3,4),-10.,np.float32);scores[0,2,2]=5
    p,g,value=structured_choices(scores,dict(gold_groups=[dict(pairs=[2],category=1,allowed_categories=[1,3],miss_cost=1.)],false_positive_cost=1.))
    assert p[0][2]==g[0][2]==3 and value==0
    write(dest/'native_alias_controls.json',dict(exhaustive_geometry_alias_hinge_max_abs=maximum,
        accepted_reversed_singleton_not_penalized=True,does_not_certify_independent_flips_in_full_parent=True,fit_calls=0))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',choices=['prepare','train','infer'],required=True)
    ap.add_argument('--source');ap.add_argument('--resume',action='store_true');a=ap.parse_args();dest=OUT/a.run
    if not a.resume and a.stage in ('prepare','train'):
        dest.mkdir(exist_ok=False)
        write(dest/'registered_protocol.json',dict(source_sha256=digest(__file__),stage=a.stage,TRAIN=list(TRAIN),DEV=['64'],
            input_run='run402',native_compatibility_run='run406',set_targets_run='run407',initial_model_run='run399',
            epochs=12,batch=64,optimizer='fresh Adam',learning_rate=.0003,seed=1031,
            original_G=213,final_epoch_only=True,preserve_best379=True,external_consultation_paused=True,
            geometry_orientation_aliases_do_not_remove_parent_parity_requirement=True,full_contract_pass=False))
    try:
        if a.stage=='prepare':prepare(dest);alias_controls(dest)
        else:
            from weld_pipeline import sjg_structured_interval_set_1251 as trainer
            if a.stage=='train':
                alias_controls(dest)
                if not a.resume:
                    protocol=read(dest/'registered_protocol.json');protocol.update(
                        trainer_source_sha256=digest(trainer.__file__),model_source_sha256=digest(trainer.base.__file__),
                        gold_decoder_source_sha256=digest(__import__('weld_pipeline.sjg_interval_set_targets_1249',fromlist=['']).__file__))
                    write(dest/'registered_protocol.json',protocol)
                trainer.train(dest,a.resume,supervision_run='run407')
            else:trainer.base.infer(dest,a.source)
    except BaseException:
        write(dest/f'{a.stage}_{a.source or "all"}_failure.json',dict(error=traceback.format_exc()));raise


if __name__=='__main__':main()
