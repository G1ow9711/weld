"""Original D3 geometry checks for expanded TRAIN compatibility.

Check three boundary/interior pairs in every positive group, and sampled
unrestricted pairs against every original child. Report sampling honestly.
"""
import argparse
from collections import defaultdict
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_native_whole_presence_1259 import TRAIN
from weld_pipeline.sjg_recall_set_worker_1162 import target_for
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_parent_set_runtime_1081 import child_geometry
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_curve_geometry import sample_arc
from weld_pipeline.sjg_interval_set_targets_1249 import gold_set
from weld_pipeline.sjg_native_interval_set_training_1253 import ZeroScores


def audit(key):
    assert key in TRAIN
    dest=OUT/'run422';folder=dest/f'source_{key}';r=read(folder/'receipt.json')
    assert digest(folder/'pairs.npz')==r['pairs_sha256'] and digest(folder/'compatibility.json')==r['compatibility_sha256']
    rel=read(folder/'compatibility.json')
    with np.load(folder/'pairs.npz') as z:indices=z['indices']
    mp_path=OUT/f'run421/endpoints/source_{key}/role_map.npz'
    assert digest(mp_path)==r['map_sha256']
    with np.load(mp_path) as z:mp={k:z[k] for k in z.files}
    lookup={int(row)//2:i for i,row in enumerate(mp['rows'])}
    target,binding=target_for(key);assert binding['target_sha256']==r['target_sha256']
    native=native_targets(target)
    children=[c for p in native['parents'] for c in p['children']]
    refs=[reference_segment(c) for c in children]
    closed=[bool(p['is_closed'] and len(p['children'])==1) for p in native['parents'] for c in p['children']]
    ends=np.asarray([np.asarray(ref['points'])[[0,-1]] for ref in refs])
    case=load_case(key);g=case['graph'];case['x']._mmap.close()
    checked_owners=set();positive_tests=0;random_tests=0;infeasible=[];short=[];rng=np.random.default_rng(1263)
    sampled=set(rng.choice(len(rel['profiles']),size=min(64,len(rel['profiles'])),replace=False).tolist())
    for pi,profile in enumerate(rel['profiles']):
        pid=profile['profile'];i=lookup[pid];a,b=mp['offsets'][i:i+2];q=mp['q_mm'][a:b];n=len(q);curve=g.curve(pid)
        unions=defaultdict(set);edge_owners=defaultdict(set)
        def segment(pair,category):
            lo,hi=divmod(int(pair),n);code=category-1
            points,arc=child_geometry(curve,q[lo],q[hi],bool(code&2),bool(code&4),bool(code&8))
            value=line_segment(points[0],points[-1]) if arc is None else sample_arc(arc)
            value['weld_type']='vertical' if code&1 else 'horizontal'
            return value
        for group in profile['groups']:
            assert not group['implicit_all_pairs'],'Implicit closed geometry requires dedicated exact audit'
            values=indices[group['pair_start']:group['pair_stop']]
            assert len(values)>0 and np.all(values//n<values%n) and np.all(np.diff(values)>0)
            oi=group['owner_index'];unions[oi].update(map(int,values))
            for category in (group['native_category'],group['reversed_parent_category']):
                for pair in values:edge_owners[int(pair),category].add(oi)
            for pair in np.unique(values[[0,len(values)//2,-1]]):
                pred=segment(pair,group['native_category'])
                assert child_error(pred,children[oi],refs[oi],closed[oi],False) is not None,(key,pid,oi,int(pair))
                positive_tests+=1;checked_owners.add(oi)
        groups=[]
        for oi,values in unions.items():
            values=np.asarray(sorted(values));lo=values//n;hi=values%n
            if lo.max()>=hi.min():short.append(dict(profile=pid,owner_index=oi))
            groups.append(dict(owner_index=oi,pairs=values,category=1,
                sort_key=(float(np.median(q[lo])),float(np.median(q[hi])))))
        groups.sort(key=lambda v:v.pop('sort_key'))
        try:gold_set(groups,ZeroScores(),n)
        except ValueError:infeasible.append(pid)
        if pi in sampled and n>=2:
            # Unrestricted pairs may be positive or negative; no hand selection.
            for _ in range(4):
                lo,hi=sorted(map(int,rng.choice(n,2,replace=False)));pair=lo*n+hi
                for category in range(1,5):
                    pred=segment(pair,category);ep=np.asarray(pred['points'])[[0,-1]]
                    distance=np.minimum(np.linalg.norm(ends-ep,axis=2).max(1),np.linalg.norm(ends-ep[::-1],axis=2).max(1))
                    expected=set()
                    for oi in np.flatnonzero((distance<=3.+1e-8)|np.asarray(closed)):
                        truth=children[oi]
                        if truth['original_kind']!=pred['kind'] or truth['weld_type']!=pred['weld_type']:continue
                        if child_error(pred,truth,refs[oi],closed[oi],False) is not None:expected.add(int(oi))
                        else:
                            # Native relation allows whole-parent reversed geometry;
                            # individual-child acceptance does not certify assembly parity.
                            from weld_pipeline.contract_geometry_d3_409 import reverse_segment
                            rev=dict(pred,**reverse_segment(pred))
                            if child_error(rev,truth,refs[oi],closed[oi],False) is not None:expected.add(int(oi))
                    rebuilt=edge_owners.get((pair,category),set())
                    assert expected==rebuilt,(key,pid,pair,category,expected,rebuilt)
                    random_tests+=1
    assert len(checked_owners)==r['covered_G']
    known=read(folder/'known_training_coverage.json')
    result=dict(source=key,G=len(children),covered_G=r['covered_G'],known_training_covered_G=known['covered_G'],
        original_owners_checked=len(checked_owners),positive_group_edge_samples_checked=positive_tests,
        unrestricted_pair_category_tests=random_tests,sampled_profiles=len(sampled),random_seed=1263,
        exhaustive_all_pairs_audit=False,nonoverlap_infeasible_profiles=infeasible,short_owner_core_failures=short,
        status='PASS' if r['covered_G']==len(children)==known['covered_G'] and not infeasible and not short else 'FAIL_CAPACITY_OR_SET_FEASIBILITY',
        source_sha256=digest(__file__),compatibility_sha256=r['compatibility_sha256'],pairs_sha256=r['pairs_sha256'],
        fit_calls=0,DEV_reads=0,STEP_reads=0,full_contract_pass=False)
    write(folder/'independent_D3_audit.json',result);print('EXPANDED_D3_AUDIT',result,flush=True)
    if all((dest/f'source_{k}/independent_D3_audit.json').exists() for k in TRAIN):
        cases=[read(dest/f'source_{k}/independent_D3_audit.json') for k in TRAIN]
        write(dest/'independent_audit_summary.json',dict(status='PASS' if all(c['status']=='PASS' for c in cases) else 'FAIL',
            cases=cases,original_G=sum(c['G'] for c in cases),fit_calls=0,DEV_reads=0,full_contract_pass=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);audit(p.parse_args().source)
