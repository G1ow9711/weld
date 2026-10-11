"""Native D3 compatibility on frozen candidate anchors, not inherited owners.

This is a relation: aliases and multiple compatible GT identities are retained.
It does not demand that every carrier independently output every compatible GT.
"""
import argparse
from collections import defaultdict
import gc
import time
import traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest
from weld_pipeline.sjg_endpoint_role_train_1246 import inputs
from weld_pipeline.sjg_relative_ordinal_runtime_1150 import load_case
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.sjg_recall_set_worker_1162 import target_for
from weld_pipeline.sjg_parent_set_runtime_1081 import child_geometry
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.contract_curve_geometry import sample_arc

TRAIN=('21','BM79A_native_local','10')


def controls(dest):
    rng=np.random.default_rng(419);maximum=0.
    truth=dict(original_kind='line',weld_type='horizontal')
    ref=line_segment([0.,0.,0.],[50.,10.,4.])
    for radius in (0.,1.,2.999,3.001):
        for _ in range(4):
            delta=rng.normal(size=(2,3));delta*=radius/np.linalg.norm(delta,axis=1)[:,None]
            points=np.asarray(ref['points'])[[0,-1]]+delta
            pred=dict(line_segment(points[0],points[1]),weld_type='horizontal')
            error=float(np.linalg.norm(points-np.asarray(ref['points'])[[0,-1]],axis=1).max())
            accepted=child_error(pred,truth,ref,False,False)
            assert (accepted is not None)==(error<=3.)
            if accepted:maximum=max(maximum,accepted['bidirectional_sampled_maximum_mm'])
    exact_ref=line_segment([0.,0.,0.],[50.,0.,0.])
    exact_pred=dict(line_segment([0.,3.,0.],[50.,3.,0.]),weld_type='horizontal')
    assert child_error(exact_pred,truth,exact_ref,False,False) is not None
    write(dest/'line_geometry_controls.json',dict(endpoint_convex_correspondence_matches_D3=True,
        maximum_accepted_sampled_error_mm=maximum,above_3mm_rejected=True,GT_reads=0))


def arc_matches(curve,q,pairs,truth,reference,closed):
    """Native-oriented arc matches, with whole-parent reversal recorded later."""
    accepted=defaultdict(set);n=len(q);wt=int(truth['weld_type']=='vertical')
    for pair in pairs:
        i,j=divmod(int(pair),n)
        for wrap in ((False,True) if curve.arc.is_closed and not closed else (False,)):
            for forward in (False,True):
                try:
                    _,arc=child_geometry(curve,q[i],q[j],forward,closed,wrap)
                    pred=dict(sample_arc(arc),weld_type=truth['weld_type'])
                    if child_error(pred,truth,reference,closed,False) is not None:
                        category=1+wt+2*forward+4*closed+8*wrap
                        accepted[category].add(int(pair))
                except (ValueError,ArithmeticError):
                    continue
    return accepted


def prepare(key,dest,input_run='run402',previous_set_run='run404',known_background_file=None,require_full_coverage=True):
    start=time.perf_counter();x,m,ir=inputs(key,input_run);x._mmap.close()
    case=load_case(key);g=case['graph'];case['x']._mmap.close()
    raw_target,bind=target_for(key);target=native_targets(raw_target)
    old_records=({r['row']:r for r in read(OUT/f'{previous_set_run}/source_{key}_set_targets.json')['records']}
                 if previous_set_run is not None else {})
    known_rows=(set(old_records) if known_background_file is None else
        {r['row'] for r in read(OUT/known_background_file)['rows'] if r['complete_background_supervision_allowed']})
    owners=[];references=[];children=[];closed_flags=[]
    for pi,parent in enumerate(target['parents']):
        for ci,child in enumerate(parent['children']):
            assert child['weld_type'] in ('horizontal','vertical')
            owners.append(dict(parent=pi,child=ci,original_child_id=child['original_child_id'],
                kind=child['original_kind'],weld_type=child['weld_type']))
            references.append(reference_segment(child));children.append(child)
            closed_flags.append(bool(parent['is_closed'] and len(parent['children'])==1))
    assert len(owners)==raw_target['original_children']
    by_profile={};row_metadata=[]
    for i,row in enumerate(m['rows']):
        row=int(row);a,b=map(int,m['offsets'][i:i+2]);q=m['q_mm'][a:b];pid=row//2
        if pid in by_profile:assert np.array_equal(by_profile[pid],q)
        else:by_profile[pid]=q
        row_metadata.append(dict(row=row,profile=pid,n=len(q),
            complete_background_supervision_allowed=row in known_rows,
            previous_owner_mapping_used_for_compatibility=False))
    endpoints=np.asarray([np.asarray(ref['points'])[[0,-1]] for ref in references])
    chunks=[];cursor=0;profiles=[];covered=set();missing_old_owner_relations=0;new_only_pairs=0;old_only_pairs=0
    multi_owner_pairs=0;overlap_profiles=0
    for pid,q in sorted(by_profile.items()):
        curve=g.curve(pid);n=len(q);xyz=curve.at(q);groups=[];potential_spans=[]
        projected,distances=curve.project(endpoints.reshape(-1,3));distances=distances.reshape(-1,2)
        for oi,(truth,reference,closed) in enumerate(zip(children,references,closed_flags)):
            if truth['original_kind']!=curve.kind:continue
            if closed:
                # Closed circles ignore arbitrary anchor endpoints. A compact
                # implicit range retains all i<j representations of a full ring.
                if curve.kind!='arc' or not curve.arc.is_closed:continue
                accepted=arc_matches(curve,q,[n-1],truth,reference,True)
                for category in accepted:
                    groups.append(dict(owner_index=oi,native_category=int(category),
                        reversed_parent_category=1+((int(category)-1)^2),
                        implicit_all_pairs=True,pair_count=n*(n-1)//2))
                    covered.add(oi)
                continue
            if max(distances[oi])>3.:continue
            ds=np.linalg.norm(xyz-endpoints[oi,0],axis=1);de=np.linalg.norm(xyz-endpoints[oi,1],axis=1)
            left=np.flatnonzero(ds<=3.);right=np.flatnonzero(de<=3.)
            if not len(left) or not len(right):continue
            if curve.kind=='line':
                accepted=defaultdict(set);wt=int(truth['weld_type']=='vertical')
                for i in left:
                    for j in right:
                        if i==j:continue
                        # Resolve floating-point boundary cases with the exact
                        # scorer instead of assuming a real-arithmetic bound.
                        if max(ds[i],de[j])>=3.-1e-8:
                            pred=dict(line_segment(xyz[i],xyz[j]),weld_type=truth['weld_type'])
                            if child_error(pred,truth,reference,False,False) is None:continue
                        lo,hi=sorted((int(i),int(j)));category=1+wt+2*(i<j)
                        accepted[int(category)].add(lo*n+hi)
            else:
                candidates=sorted({min(int(i),int(j))*n+max(int(i),int(j)) for i in left for j in right if i!=j})
                accepted=arc_matches(curve,q,candidates,truth,reference,False)
            for category,indices in sorted(accepted.items()):
                pairs=np.asarray(sorted(indices),np.int64)
                if not len(pairs):continue
                chunks.append(pairs);stop=cursor+len(pairs)
                groups.append(dict(owner_index=oi,native_category=int(category),
                    reversed_parent_category=1+((int(category)-1)^2),implicit_all_pairs=False,
                    pair_start=cursor,pair_stop=stop,pair_count=len(pairs)))
                potential_spans.append((oi,float(q[(pairs//n).min()]),float(q[(pairs%n).max()])))
                cursor=stop;covered.add(oi)
        # Summaries concern geometric aliases, never resolve them into fake negatives.
        owner_sets=defaultdict(set)
        # Groups are contiguous append-only views; collect the matching local chunks.
        local_chunks=chunks[-sum(not v['implicit_all_pairs'] for v in groups):] if any(not v['implicit_all_pairs'] for v in groups) else []
        position=0
        for group in groups:
            if group['implicit_all_pairs']:continue
            pairs=local_chunks[position];position+=1;wt=(group['native_category']-1)%2;oi=group['owner_index']
            for pair in pairs:
                code=int(pair)*2+wt;owner_sets[code].add(oi)
        multi_owner_pairs+=sum(len(v)>1 for v in owner_sets.values())
        overlap=any(a!=b and max(lo,lo2)<min(hi,hi2)-1e-6 for i,(a,lo,hi) in enumerate(potential_spans)
            for b,lo2,hi2 in potential_spans[i+1:])
        overlap_profiles+=int(overlap)
        native_pairs=set(owner_sets)
        for row in (pid*2,pid*2+1):
            if row not in old_records:continue
            previous=old_records[row];old_pairs={int(pair)*2+(v['category']-1)%2 for v in previous['gold_groups'] for pair in v['pairs']}
            old_owners={tuple(v['owner']) for v in previous['gold_groups']}
            native_owners={(owners[v['owner_index']]['parent'],owners[v['owner_index']]['child']) for v in groups}
            missing_old_owner_relations+=len(native_owners-old_owners)
            new_only_pairs+=len(native_pairs-old_pairs);old_only_pairs+=len(old_pairs-native_pairs)
        profiles.append(dict(profile=pid,n=n,groups=groups,geometry_alias_pair_count=sum(len(v)>1 for v in owner_sets.values()),
            overlapping_compatible_owner_extents=overlap))
        if len(profiles)%256==0:print('NATIVE_COMPATIBILITY_PROGRESS',key,len(profiles),'/',len(by_profile),flush=True)
    missing=[owners[i] for i in range(len(owners)) if i not in covered]
    if require_full_coverage:assert not missing,(key,'original children not covered',missing)
    folder=dest/f'source_{key}';folder.mkdir(exist_ok=False)
    np.savez_compressed(folder/'pairs.npz',indices=np.concatenate(chunks) if chunks else np.empty(0,np.int64))
    write(folder/'compatibility.json',dict(source=key,owners=owners,rows=row_metadata,profiles=profiles,
        parents=[dict(original_child_count=p['original_child_count'],is_closed=p['is_closed'],gaps=p['gaps']) for p in target['parents']],
        whole_parent_reversal_parity_must_be_shared=True,children_cannot_flip_independently=True,
        relation_not_mandatory_output_per_carrier=True,geometry_only_not_attribute_certification=True))
    result=dict(source=key,G=len(owners),covered_G=len(covered),profiles=len(profiles),rows=len(row_metadata),
        known_background_rows=sum(r['complete_background_supervision_allowed'] for r in row_metadata),
        unknown_background_rows=sum(not r['complete_background_supervision_allowed'] for r in row_metadata),
        indexed_native_oriented_pairs=cursor,missing_old_owner_relations=missing_old_owner_relations,
        geometry_positive_pairs_missing_old=new_only_pairs,old_positive_pairs_failing_native=old_only_pairs,
        multiple_owner_geometry_pairs=multi_owner_pairs,profiles_with_overlapping_owner_extents=overlap_profiles,
        target_sha256=bind['target_sha256'],feature_sha256=ir['feature_sha256'],map_sha256=ir['map_sha256'],
        input_run=input_run,previous_owner_comparison_available=previous_set_run is not None,
        known_background_sha256=digest(OUT/known_background_file) if known_background_file else None,
        missing_original_children=missing,require_full_coverage=require_full_coverage,
        compatibility_sha256=digest(folder/'compatibility.json'),pairs_sha256=digest(folder/'pairs.npz'),
        GT_used_for_supervision_only=True,GT_endpoint_insertion=False,DEV_reads=0,STEP_reads=0,fit_calls=0,
        full_contract_pass=False,seconds=time.perf_counter()-start)
    write(folder/'receipt.json',result);print('NATIVE_COMPATIBILITY_READY',result,flush=True)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();d=OUT/a.run;d.mkdir(exist_ok=False)
    write(d/'registered_protocol.json',dict(source_sha256=digest(__file__),TRAIN=list(TRAIN),input_run='run402',
        target_source='original native target_for, not inherited candidate owners',fit_calls=0,DEV_reads=0,
        D3_endpoint_and_whole_curve_mm=3.,whole_parent_parity_preserved=True,original_G=213))
    try:
        controls(d);cases=[]
        for key in TRAIN:cases.append(prepare(key,d));gc.collect()
        write(d/'receipt.json',dict(status='NATIVE_D3_COMPATIBILITY_REBUILT_PENDING_INDEPENDENT_AUDIT',cases=cases,
            G=sum(v['G'] for v in cases),fit_calls=0,DEV_reads=0,STEP_reads=0,full_contract_pass=False))
    except BaseException:write(d/'failure.json',dict(error=traceback.format_exc(),fit_calls=0));raise


if __name__=='__main__':main()
