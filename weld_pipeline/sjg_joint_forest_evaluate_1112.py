"""Independent label reader after the conditional prediction has been frozen."""
import argparse
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents


def bm_object_diagnostic(model_run):
    import pickle
    import numpy as np
    from weld_pipeline.sjg_joint_forest_train_1109 import load_case
    path=OUT/model_run/'model.pkl';receipt=read(OUT/model_run/'receipt.json')
    assert digest(path)==receipt['model_sha256']
    with path.open('rb') as f:models=pickle.load(f)
    if models.get('feature_columns')==152:
        from weld_pipeline.sjg_relative_forest_runtime_1139 import load_relative_case
        case=load_relative_case('BM79A_native_local',models['feature_run'],supervision=True)
    else:case=load_case('BM79A_native_local',supervision=True)
    ids=sorted(case['teacher']);x=case['x'][ids]
    model=models['object'];probability=model.predict_proba(x)[:,model.classes_!=0].sum(1)
    depths=[];samples=[]
    for estimator in model.estimators_:
        tree=estimator.tree_;depth=np.zeros(tree.node_count,np.int32);stack=[0]
        while stack:
            i=stack.pop()
            for child in (tree.children_left[i],tree.children_right[i]):
                if child>=0:depth[child]=depth[i]+1;stack.append(child)
        leaves=estimator.apply(x);depths.append(depth[leaves]);samples.append(tree.n_node_samples[leaves])
    depths=np.asarray(depths);samples=np.asarray(samples);rows=[]
    for j,i in enumerate(ids):
        node=case['teacher'][i];curve=case['graph'].curve(case['graph'].profile(node['fragment']))
        rows.append(dict(kind=curve.kind,object_row=i,positive_probability=float(probability[j]),
            passed_fixed_half=bool(probability[j]>=.5),trees_at_max_depth=int((depths[:,j]==model.max_depth).sum()),
            registered_max_depth=model.max_depth,leaf_samples_median=float(np.median(samples[:,j])),
            leaf_samples_min=int(samples[:,j].min()),leaf_samples_max=int(samples[:,j].max())))
    return dict(rows=rows,new_fit_calls=0,posthoc_TRAIN_diagnostic_attached_to_actual_result=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;pred=read(dest/'receipt.json')
    assert pred['status']=='CONDITIONAL_PURE_INPUT_PREDICTION_FROZEN'
    assert digest(dest/'decoded.json')==pred['decoded_sha256']
    key=pred['source'];case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/case['target_file'])==case['target_sha256']
    target=read(OUT/case['target_file']);decoded=read(dest/'decoded.json')
    training=read(OUT/pred['model_run']/'receipt.json')
    scope='TRAIN' if key in [r['source'] for r in training['cases']] else case['role']
    score=score_native_parents(decoded,target,[False]*len(target['parents']))
    value=score['categories']['overall'];tp=value['TP'];g=value['G'];fp=value['FP'];fn=g-tp
    errors=[]
    for match in score['matches']:
        errors.extend(match['ordered_endpoint_errors_after_allowed_whole_correspondence_mm'] or [])
        errors.append(match['bidirectional_sampled_maximum_mm'])
    full_parents=0
    for ti,t in enumerate(target['parents']):
        full_parents+=sum(m['original_parent_index']==ti for m in score['matches'])==len(t['children'])
    gate=({'21':tp==1 and fp==0,
           'BM79A_native_local':tp==4 and fp==0 and len(decoded['parents'])==1 and full_parents==1,
           '10':tp/g>=.8 and fp/g<=.2,'64':tp/g>=.5 and fp/g<=.5}).get(key,False)
    if training['status'] in ('THREE_RELATIVE_FORESTS_TRAINING_COMPLETE','THREE_RELATIVE_ORDINAL_FORESTS_TRAINING_COMPLETE') and scope=='TRAIN':
        gate=(tp==g and fp==0 and full_parents==len(target['parents']) and
              len(decoded['parents'])==len(target['parents']) and not decoded['invalid_predictions'] and
              bool(errors) and max(errors)<=3.)
    report=dict(source=key,scope=scope,legacy_cohort_role=case['role'],TP=tp,G=g,FP=fp,FN=fn,
        extraction_correct_rate_percent=100*tp/g,
        output_precision_percent=None if tp+fp==0 else 100*tp/(tp+fp),
        matched_maximum_geometry_error_mm=max(errors) if errors else None,
        emitted_children=value['predicted'],predicted_parents=len(decoded['parents']),
        original_parents=len(target['parents']),fully_matched_parents=full_parents,
        invalid_predicted_parents=len(decoded['invalid_predictions']),development_continuation_gate_pass=bool(gate),
        cached_inference_seconds=pred['cached_pipeline_seconds'],graph_selection_seconds=pred['graph_selection_seconds'],
        frozen_model_sha256=pred['model_sha256'],prediction_sha256=pred['decoded_sha256'],target_sha256=case['target_sha256'],
        formal_E16_and_directed_side_pass=False,raw_PCD_uncached_contract_time_measured=False,
        full_contract_pass=False,original_full_pilot_G=500,utc=utc())
    attached=bm_object_diagnostic(pred['model_run']) if key=='BM79A_native_local' else None
    write(dest/'evaluation.json',dict(report=report,score=score,BM_object_gate_diagnostic=attached))
    text=(f"# 本轮模型实测：{key}\n\n"
          f"焊缝提取正确率：{report['extraction_correct_rate_percent']:.6f}%（{tp}/{g}）。\n\n"
          f"误检：{fp} 条；漏检：{fn} 条；输出：{value['predicted']} 条。\n\n"
          f"输出精确率：{report['output_precision_percent']}%；匹配对象最大几何误差：{report['matched_maximum_geometry_error_mm']} mm。"
          "None 表示无输出或无正确匹配，不能记为0毫米。\n\n"
          f"范围：{scope}；完整父匹配：{full_parents}/{len(target['parents'])}；"
          f"开发继续门：{'通过' if gate else '不通过'}。完整合同尚未通过。\n")
    (dest/'实测指标.md').write_text(text,encoding='utf-8')
    import json;print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
