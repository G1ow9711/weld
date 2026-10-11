"""Independent geometric child diagnosis after all free outputs are frozen."""
import argparse
import traceback
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import ROOT,OUT,read,write,digest,utc
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_curve_supervision_915 import supervise


def geometric_match(predictions,target,tolerance=3.):
    children=[child for parent in target['parents'] for child in parent['children']]
    curves=[CurveCarrier(p['representation']) for p in predictions]
    ends=np.asarray([c.at(np.array([0.,c.length])) for c in curves]).reshape(-1,2,3)
    kinds=np.asarray([c.kind for c in curves]);errors={};weight=np.zeros((len(children),len(curves)+len(children)))
    for i,child in enumerate(children):
        gt=CurveCarrier(child['representation']);gt_ends=gt.at(np.array([0.,gt.length]));points=gt.at(gt.grid(1.));tree=cKDTree(points)
        # Directed endpoint test. No independent per-child reversal is granted.
        endpoint=np.linalg.norm(ends-gt_ends,axis=2)
        possible=np.flatnonzero((kinds==gt.kind)&(endpoint.max(1)<=tolerance))
        for j in possible:
            j=int(j);pred=curves[j];p=pred.at(pred.grid(1.))
            d4=max(float(tree.query(p)[0].max()),float(cKDTree(p).query(points)[0].max()))
            if d4>tolerance:continue
            errors[i,j]=dict(endpoint_errors_mm=endpoint[j].tolist(),bidirectional_sampled_maximum_mm=d4,
                              original_child_id=child['original_child_id'],kind=gt.kind)
            weight[i,j]=1e6-d4
    ri,ci=linear_sum_assignment(weight,maximize=True);matches=[]
    for i,j in zip(ri,ci):
        if (int(i),int(j)) in errors:matches.append(dict(target=int(i),prediction=int(j),**errors[int(i),int(j)]))
    G=len(children);TP=len(matches);FP=len(curves)-TP
    return dict(G=G,TP=TP,FP=FP,FN=G-TP,predicted=len(curves),TP_over_G=TP/G if G else None,
                FP_over_G=FP/G if G else None,matches=matches,not_full_contract_accuracy=True,
                parent_count_order_gap_closure_and_weld_type_not_assessed=True,unknown_E16_not_counted_as_pass=True)


def controls():
    rep=dict(kind='line',endpoints_mm=[[0,0,0],[10,0,0]])
    target=dict(parents=[dict(children=[dict(original_child_id='control:0',representation=rep)])])
    score=geometric_match([dict(representation=rep),dict(representation=rep)],target)
    assert score['G']==1 and score['TP']==1 and score['FP']==1
    reverse=dict(kind='line',endpoints_mm=[[10,0,0],[0,0,0]])
    assert geometric_match([dict(representation=reverse)],target)['TP']==0
    assert geometric_match([],target)['FN']==1
    return dict(one_to_one_duplicate_FP=True,directed_endpoints_enforced=True,missing_prediction_keeps_denominator=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--prediction-run',default='run118');ap.add_argument('--data-run',default='run116');ap.add_argument('--proposal-run',default='run115')
    args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    source=OUT/args.prediction_run;write(dest/'registered_protocol.json',dict(stage='FROZEN_LOCAL_EVENT_GEOMETRY_DIAGNOSIS',
        source_sha256=digest(__file__),prediction_run=args.prediction_run,data_run=args.data_run,proposal_run=args.proposal_run,original_G=496,original_G_kept=True,curve_step_mm=1.,
        tolerance_mm=3.,global_one_to_one=True,directed_endpoints=True,no_per_child_reversal=True,
        no_posthoc_output_deletion=True,full_contract_pass=False,start_utc=utc()))
    try:
        write(dest/'controls.json',controls());assert read(source/'receipt.json')['status']=='LOCAL_TYPED_FREE_PREDICTIONS_FROZEN'
        manifest=read(source/'all_predictions_frozen_before_labels.json');bindings={r['source_key']:r for r in manifest['sources']}
        cases=read(OUT/'run98/local_cohort.json')['cases'];assert set(bindings)=={c['source_key'] for c in cases}
        # Verify the entire prediction set before the first target read.
        for key,row in bindings.items():
            for file,sha in row['files'].items():assert digest(source/('source_'+key)/file)==sha
        scores=[];coverage=[]
        train_data={r['source_key']:r for r in read(OUT/args.data_run/'training_manifest.json')['cases']}
        for case in cases:
            key=case['source_key'];assert digest(OUT/case['target_file'])==case['target_sha256']
            target=read(OUT/case['target_file']);predictions=read(source/('source_'+key)/'predictions.json')
            score=geometric_match(predictions,target);assert score['G']==case['original_G']
            if key in train_data:
                row=train_data[key];assert row['target_sha256']==case['target_sha256']
                available=dict(source_key=key,G=row['original_G'],geometrically_supported=row['covered_G'],missing=row['missing_G'])
            else:
                folder=OUT/args.proposal_run/('source_'+key);binding=read(folder/'input_freeze.json')
                assert digest(folder/'profiles.json')==binding['files']['profiles.json']
                support=supervise(read(folder/'profiles.json'),target)
                write(dest/('source_'+key+'_post_prediction_carrier_diagnosis.json'),support)
                available=dict(source_key=key,G=support['original_G'],geometrically_supported=support['covered_children'],missing=len(support['missing']))
            coverage.append(available)
            write(dest/('source_'+key+'_geometry.json'),score);scores.append(dict(source_key=key,role=case['role'],**score))
            print('LOCAL_TYPED_GEOMETRY',key,'TP',score['TP'],'/',score['G'],'FP',score['FP'],'CONTRACT_NOT_PASSED',flush=True)
        groups={}
        for name,keys in [('original_eight',[c['source_key'] for c in cases if c['source_key']!='BM79A_native_local']),
                          ('original_train',[c['source_key'] for c in cases if c['role']=='TRAIN']),
                          ('development',['64']),('arc_diagnostic',['24']),('additional_TRAIN',['BM79A_native_local'])]:
            totals={field:sum(row[field] for row in scores if row['source_key'] in keys) for field in ('G','TP','FP','FN','predicted')}
            totals.update(TP_over_G=totals['TP']/totals['G'],FP_over_G=totals['FP']/totals['G'])
            groups[name]=totals
        assert groups['original_eight']['G']==496
        write(dest/'receipt.json',dict(status='LOCAL_EVENT_GEOMETRY_DIAGNOSED',groups=groups,carrier_coverage_diagnosis=coverage,
            full_contract_pass=False,full_contract_accuracy_not_established=True,
            unmet_or_not_run=['parent grouping/count/order/gap/closure','weld type','all E16 fields including thickness and directed normals',
                             'formal frozen independent split','uncached three-run end-to-end timing gate'],
            geometry_results_are_auxiliary=True,external_consultation_not_sent=True,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise

if __name__=='__main__':main()
