"""Exact-cache control, original B training, and unchanged-denominator reporting."""
import argparse,time,traceback
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,check_resources
from weld_pipeline.sjg_recall_set_worker_1162 import load_parents,train_selector,target_for,final_score
from weld_pipeline.sjg_parent_set_features_1160 import ParentRelations
from weld_pipeline.sjg_relation_column_cache_1173 import RelationColumnCache,CachedParentRelations
from weld_pipeline.sjg_seven_relative_prepare_1146 import KEYS


def bindings(dest,keys):
    result={}
    for key in keys:
        folder=dest/f'source_{key}';r=read(folder/'receipt.json')
        assert digest(folder/'decoded.json')==r['decoded_sha256']
        result[key]=dict(prediction_sha256=r['decoded_sha256'],metadata_sha256=digest(folder/'parent_metadata.json'),
            compact_parents_sha256=digest(folder/'compact_parents.json'),points_sha256=digest(folder/'parent_points.npy'),
            feature_code_sha256=digest(ROOT/'weld_pipeline/sjg_parent_set_features_1160.py'),
            column_semantics='pair(candidate,selected_candidate) float64 full column; no targets or weights')
    return result


def cache_control(dest):
    folder=dest/'source_10';parents,points=load_parents(folder);metadata=read(folder/'parent_metadata.json')
    cache=RelationColumnCache(dest/'cache_control',bindings(dest,['10']))
    choices=[0,len(parents)//2,len(parents)-1]
    for order in (choices,choices[::-1]):
        reference=ParentRelations(parents,metadata);cached=CachedParentRelations(parents,metadata,'10',cache)
        for index in order:
            reference.add(index);cached.add(index)
            assert reference.selected==cached.selected and np.array_equal(reference.features(),cached.features())
        del reference,cached
    assert cache.stats['computed_columns']==3 and cache.stats['memory_hits']==3
    # Exercise disk reload after releasing the bounded in-memory entries.
    cache.memory.clear();cache.memory_bytes=0
    reference=ParentRelations(parents,metadata);cached=CachedParentRelations(parents,metadata,'10',cache)
    reference.add(choices[0]);cached.add(choices[0]);assert np.array_equal(reference.features(),cached.features())
    assert cache.stats['disk_hits']==1
    write(dest/'cache_equivalence.json',dict(status='EXACT_RELATION_COLUMN_CACHE_PASSED',full_features_bitwise_equal=True,
        distinct_prefix_orders=2,disk_reload_checked=True,cache=cache.receipt(),real_fit_calls=0,GT_reads=0))
    print('EXACT_RELATION_COLUMN_CACHE_PASSED',cache.receipt(),flush=True)


def train(dest):
    assert read(dest/'cache_equivalence.json')['status']=='EXACT_RELATION_COLUMN_CACHE_PASSED'
    cache=RelationColumnCache(dest/'training_relation_cache',bindings(dest,KEYS))
    try:train_selector(dest,relation_factory=lambda parents,metadata,key:CachedParentRelations(parents,metadata,key,cache))
    finally:write(dest/'training_relation_cache_receipt.json',cache.receipt())


def coverage_report(dest):
    from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
    from weld_pipeline.world_set_evaluation_285 import category_keys,CATEGORIES
    from weld_pipeline.sjg_parent_set_oracle_1158 import ParentSetOracle
    cases=[]
    for key in KEYS:
        c=read(dest/f'source_{key}/compatibility.json');target,_=target_for(key);truth=native_targets(target)['parents']
        categories={k:dict(G=0,maximum_coverable_children=0) for k in CATEGORIES}
        for parent in truth:
            for child in parent['children']:
                for category in category_keys(child['weld_type'],child['original_kind']):categories[category]['G']+=1
        # Optimize each category's covered child count under whole-parent compatibility.
        # Parent-count maximum alone need not maximize every child-weighted category.
        from scipy.optimize import linear_sum_assignment
        for category,value in categories.items():
            if not value['G']:continue
            weights=np.array([sum(category in category_keys(ch['weld_type'],ch['original_kind']) for ch in p['children']) for p in truth])
            cost=np.zeros((len(c['compatible_targets']),len(truth)),np.int64)
            for i,targets in enumerate(c['compatible_targets']):cost[i,targets]=-weights[targets]
            if cost.size:
                a,b=linear_sum_assignment(cost);value['maximum_coverable_children']=int(-cost[a,b].sum())
            value['upper_percent']=100*value['maximum_coverable_children']/value['G']
        cases.append(dict(source=key,G=c['original_children'],parents=c['original_parents'],
            maximum_parent_matching=c['maximum_matched_parents'],candidate_path_miss=c['uncovered_target_parents'],
            categories=categories,not_a_model_score=True))
    assert sum(c['G'] for c in cases)==363
    write(dest/'all_train_candidate_coverage.json',dict(cases=cases,G=363,all_original_denominators_retained=True,
        category_upper_bounds_may_require_different_matchings=True,
        missing_parents_status='CANDIDATE_PATH_MISS',is_extraction_result=False))
    print('ALL_SEVEN_CANDIDATE_DENOMINATORS_RETAINED',[(c['source'],c['G'],len(c['candidate_path_miss'])) for c in cases],flush=True)


def aggregate(dest):
    results=[read(dest/f'test_{k}/evaluation.json') for k in ('21','BM79A_native_local','10')]
    requirements={'overall':(213,203,10),'Hline':(84,80,4),'Vline':(126,120,6),'HArc':(3,3,0)}
    groups={};passed=True
    for name,(g,tp_min,fp_max) in requirements.items():
        values=[v['score']['categories'][name] for v in results]
        total={k:sum(v[k] for v in values) for k in ('G','TP','FP','predicted')};assert total['G']==g
        total.update(FN=g-total['TP'],TP_min=tp_min,FP_max=fp_max,
            extraction_correct_rate_percent=100*total['TP']/g,FP_over_G_percent=100*total['FP']/g,
            passed=total['TP']>=tp_min and total['FP']<=fp_max)
        groups[name]=total;passed &= total['passed']
    maxima=[r['report']['matched_maximum_geometry_error_mm'] for r in results if r['report']['matched_maximum_geometry_error_mm'] is not None]
    passed &= bool(maxima) and max(maxima)<=3.
    report=dict(status='THREE_CASE_GEOMETRY_CONTINUATION_EVALUATED',groups=groups,geometric_continuation_pass=bool(passed),
        reports=[r['report'] for r in results],matched_maximum_geometry_error_mm=max(maxima) if maxima else None,
        fully_matched_parents=sum(r['report']['fully_matched_parents'] for r in results),original_parents=126,
        baseline_TP=213,baseline_FP=0,regression_TP_delta=groups['overall']['TP']-213,
        geometry_gate_is_not_full_contract=True,full_contract_pass=False)
    write(dest/'three_case_final_summary.json',report);print('FINAL_THREE_CASE_METRICS',report,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--stage',required=True);ap.add_argument('--source',default='')
    args=ap.parse_args();dest=OUT/args.run;begin=time.perf_counter()
    try:
        check_resources()
        if args.stage=='certificate_control':
            from weld_pipeline.sjg_solver_certificate_1176 import regression_control
            value=regression_control(read(OUT/'run361/solver_assertion_diagnosis.json'))
            write(dest/'certificate_control.json',value);print('NUMERICAL_CERTIFICATE_CONTROL',value,flush=True)
        elif args.stage=='cache_control':cache_control(dest)
        elif args.stage=='train_selector':train(dest)
        elif args.stage=='coverage':coverage_report(dest)
        elif args.stage=='aggregate':aggregate(dest)
        elif args.stage=='score':final_score(dest/f'test_{args.source}',args.source,
            candidate_compatibility_path=dest/f'source_{args.source}/compatibility.json' if args.source in KEYS else None)
        else:raise ValueError(args.stage)
        write(dest/f'stage_{args.stage}_{args.source}_complete.json',dict(status='STAGE_COMPLETE',seconds=time.perf_counter()-begin,
            worker_source_sha256=digest(__file__)))
    except BaseException:
        write(dest/f'stage_{args.stage}_{args.source}_failure.json',dict(error=traceback.format_exc(),seconds=time.perf_counter()-begin));raise


if __name__=='__main__':main()
