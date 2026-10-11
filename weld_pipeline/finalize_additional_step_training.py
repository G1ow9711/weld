"""Join136/137 actual receipts to the verified STEP round without acceptance."""
import json
from pathlib import Path

from training.labels import sha256


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def main():
    parent=Path('weld_pipeline/reports/step_training_round_20260930/completion_receipt.json')
    receipt=read(parent)
    assert not receipt['full_contract_pass']
    model=Path(receipt['checkpoint'])
    assert sha256(model)==receipt['checkpoint_sha256']
    trials={}
    for seed,folder in [(136,'horizontal_patch_tree_seed136'),(137,'endpoint_process_attributes_seed137')]:
        root=Path('weld_pipeline/runs')/folder
        complete=read(root/'complete.json');config=read(root/'config.json')
        assert sha256(root/'heads.joblib')==complete['head_sha256']
        for source,wanted in config.get('source_hashes',config.get('sources',{})).items():
            assert sha256(source)==wanted
            assert sha256(root/'source_snapshot'/Path(source).name)==wanted
        assert not complete['full_contract_pass']
        trials[str(seed)]=complete
    tree=trials['136'];process=trials['137']
    assert not tree['eligible_on_fit']
    assert process['not_integrated_into_v4'] and process['batch_scalar_equivalent_comparisons']==160
    assert sha256(process['evaluator_source'])==process['evaluator_sha256']
    for field,value in process['fit_joint'].items():
        assert value['truth']==1605 and 0<=value['valid']<=value['known_truth']<=1605
        assert process['untrained_joint'][field]['truth']==128
    path=Path('合同/STEP核查与模型训练进展_20260930.md')
    body=path.read_text('utf8')
    marker='追加136/137：'
    if marker not in body:
        lines=body.splitlines()
        i=next(i for i,s in enumerate(lines) if s.startswith('| 135 |'))
        f,c=tree['fit_summary'],tree['untrained_check_summary']
        lines.insert(i+1,f"| 136 | 局部点谱128树，236314采样点；H FIT{f['valid_children']}/{f['truth_count']}，FP{f['false_predictions']} | {c['valid_children']}/{c['truth_count']}，FP{c['false_predictions']} | FIT退化，未替换v4 |")
        body='\n'.join(lines)+'\n'
        body=body.replace('135已完成120轮，当前无本轮训练进程。','135已完成120轮，136/137追加训练与评分亦已结束，当前无本轮训练进程。')
        addition=[marker+'六类共享端点/单子段标志头已实际训练，合计11846条字段监督（不是独立工件数）；挡板距、板高、孔高、底板边距及檐子/侧板切分，各类128树。只用25FIT已有明确标签；原始Excel不改，内部切点和多子段父Any不广播。挡板−1、负数及板高0含义不明样本排除；孔0为无孔、边距51为>50，状态分类与确切数值回归分开。前后挡板对应起终点为未冻结开发语义。','',
            '条件FIT表中全部已知端点标签拟合通过，但实际几何下已下降；未训练板高、孔高、边距和标志失败。下表“联合”把未知/几何漏检计为未通过，使用参考匹配的PCD接触候选作诊断，尚未验证独立运行时接触选择；只比端点，不能替代任意沿线位置的完整E3验收。137独立保留为失败诊断头，未并入已核查v4。','',
            '| 字段 | FIT已知端点条件对照 | 未训练已知端点条件对照 | FIT联合/1605 | 未训练联合/128 |',
            '|---|---:|---:|---:|---:|']
        names={'front_wall':'前挡板距','back_wall':'后挡板距','start_height':'起点板高','end_height':'终点板高',
            'start_hole_height':'起点孔高/无孔','end_hole_height':'终点孔高/无孔','start_base_edge':'起点底板边距',
            'end_base_edge':'终点底板边距','has_eave':'檐子','side_plate_split':'侧板切分'}
        for field,label in names.items():
            fit=process['fit_conditional'][field];check=process['untrained_conditional'][field]
            addition.append(f"| {label} | {fit['valid']}/{fit['known']} | {check['valid']}/{check['known']} | {process['fit_joint'][field]['valid']} | {process['untrained_joint'][field]['valid']} |")
        addition+=['',
            '137评分恢复记录：六类权重已保存后，逐条调用线程池开销过高；受控中断该评分进程，按已保存同一组权重恢复批次评分。160项实际输入的批次/逐条状态、数值及置信度对照一致，权重SHA256不变。此为评分执行方式调整，未改变模型、阈值、3mm误差门槛或独立检查口径。状态置信度不是已校准数值误差概率。','',
            '追加证据：weld_pipeline/runs/horizontal_patch_tree_seed136/complete.json；weld_pipeline/runs/endpoint_process_attributes_seed137/complete.json及fit/check_conditional.json、fit/check_joint.json；weld_pipeline/reports/step_training_round_20260930/additional_training_receipt.json。当前主要差距仍包括几何与端点属性泛化、完整子段/沿线真值及正式测试，完整合同状态保持NOT_ACHIEVED。']
        body+='\n'+'\n'.join(addition)+'\n'
        path.write_text(body,encoding='utf8')
    supplemental={'status':'ADDITIONAL_HEADS_TRAINED_AND_SCORED_FULL_CONTRACT_NOT_ACHIEVED',
        'parent_receipt_sha256':sha256(parent),'retained_checkpoint':str(model),
        'retained_checkpoint_sha256':sha256(model),'trials':trials,'report':str(path),'report_sha256':sha256(path),
        'full_contract_pass':False,'retained_model_changed':False,'untrained_checks_used_for_tuning':False,
        'speed_optimization':'DEFERRED_BY_USER','active_training_processes_for_these_trials':False}
    out=parent.parent/'additional_training_receipt.json'
    out.write_text(json.dumps(supplemental,ensure_ascii=False,indent=2),encoding='utf8')
    print('ADDITIONAL_ROUND_RECORDED',supplemental['status'],flush=True)


if __name__=='__main__':
    main()
