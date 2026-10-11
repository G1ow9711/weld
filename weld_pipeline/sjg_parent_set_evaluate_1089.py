"""Independent native parent scoring after prediction artifact hash is frozen."""
import argparse
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prediction-run',required=True);args=ap.parse_args()
    folder=OUT/args.prediction_run;r=read(folder/'receipt.json')
    assert r['status']=='FROZEN_PARENT_PREDICTION_READY'
    assert digest(folder/'prediction.json')==r['prediction_sha256']
    decoded=read(folder/'prediction.json');key=r['source']
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']==key)
    assert digest(OUT/case['target_file'])==case['target_sha256']
    target=read(OUT/case['target_file'])
    score=score_native_parents(decoded,target,[False]*len(target['parents']))
    category=score['categories']['overall'];g,tp,fp=category['G'],category['TP'],category['FP']
    errors=[]
    for match in score['matches']:
        errors.append(match['bidirectional_sampled_maximum_mm'])
        errors.extend(match['ordered_endpoint_errors_after_allowed_whole_correspondence_mm'] or [])
    maximum=max(errors) if errors else None
    gate=(tp==g and fp==0) if key in ('21','BM79A_native_local') else (
        tp/g>=.8 and fp/g<=.2) if key=='10' else tp/g>=.5 and fp/g<=.5
    summary=dict(source=key,scope='TRAIN developmental parent geometry' if key!='64' else 'DEV developmental parent geometry',
        G=g,TP=tp,FP=fp,FN=g-tp,extraction_correct_rate=tp/g,
        prediction_precision=None if tp+fp==0 else tp/(tp+fp),maximum_matched_error_mm=maximum,
        quick_rejection_gate_pass=gate,full_contract_pass=False,
        formal_direction_side_E16_not_certified=True,full_pilot_G=500,
        cached_downstream_inference_seconds=r['cached_downstream_inference_seconds'],
        prediction_sha256=r['prediction_sha256'],checkpoint_sha256=r['checkpoint_sha256'],utc=utc())
    if r.get('incomplete_training'):
        summary.update(incomplete_training=True,registered_final_evaluation=False,
            completed_training_updates=r['completed_training_updates'],required_training_updates=r['required_training_updates'])
        summary['scope']+='; incomplete training checkpoint diagnostic, not registered final trial'
    write(folder/'evaluation.json',dict(summary=summary,score=score))
    mm='无法计算（无正确匹配）' if maximum is None else f'{maximum:.6f} mm（仅正确匹配）'
    report=(f'# 本轮实际模型测试：{key}\n\n'
        f'- 焊缝提取正确率：**{100*tp/g:.2f}%（{tp}/{g}）**。\n'
        f'- 输出精确率：**{100*tp/(tp+fp):.2f}%（{tp}/{tp+fp}）**。\n' if tp+fp else
        f'# 本轮实际模型测试：{key}\n\n- 焊缝提取正确率：**{100*tp/g:.2f}%（{tp}/{g}）**。\n- 输出精确率：**无法计算（零输出）**。\n')
    report+=(
        f'- 定位精度：**{mm}**。\n- 误检：**{fp}条**；漏检：**{g-tp}条**。\n'
        f'- 测试范围：{summary["scope"]}，单件；完整9件G500分母保留。\n'
        f'- 快速否决门：{"通过" if gate else "未通过"}；完整合同：**未达标**。\n\n'
        f'上游缓存后的下游推理耗时{r["cached_downstream_inference_seconds"]:.3f}秒，不能作为合同全流程时间。'
        '正式方向、侧别及E16属性尚未验收。每个阳性父对象均输出或计入误检，没有按真实数量截取。\n')
    if r.get('incomplete_training'):
        report=(f'**未完成训练的诊断：{r["completed_training_updates"]}/{r["required_training_updates"]}次更新。'
                '原训练预算失败；本结果不代表登记的完整终测通过。**\n\n')+report
    (folder/'模型实测指标.md').write_text(report,encoding='utf-8')
    print('ACTUAL_EXTRACTION',key,f'TP/G={tp}/{g}',f'CORRECT={100*tp/g:.2f}%',f'FP={fp}',f'FN={g-tp}',
        'MAX_ERROR_MM',maximum,'QUICK_GATE',gate,'FULL_CONTRACT',False,flush=True)


if __name__=='__main__':main()
