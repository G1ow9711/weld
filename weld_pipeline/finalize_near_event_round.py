"""Bind completed trials, intervention evidence, raw inference, and source preservation."""
import json
from pathlib import Path

from training.labels import sha256

OUT = Path('weld_pipeline/reports/near_event_followup_156_157_20261001')
REPORT = Path('合同/近事件定位与边界错因验证_20261001.md')
RUN = Path('weld_pipeline/runs/near_event_capacity_156')


def main():
    if OUT.exists() or REPORT.exists():
        raise ValueError('Preserve completed near-event round')
    trial = json.loads((RUN/'complete.json').read_text('utf8'))
    config = json.loads((RUN/'config.json').read_text('utf8'))
    prior_path = Path('weld_pipeline/reports/material_event_followup_20261001/completion_receipt.json')
    prior = json.loads(prior_path.read_text('utf8'))
    proof_path = Path('weld_pipeline/reports/near_event_target_proof_156_20261001/complete.json')
    proof = json.loads(proof_path.read_text('utf8'))
    diagnostic_path = Path('weld_pipeline/reports/near_event_head_interventions_157_20261001/complete.json')
    diagnostic = json.loads(diagnostic_path.read_text('utf8'))
    if (trial['models_completed'] != 2 or trial['selected'] is not None
            or sha256(RUN/'config.json') != trial['config_sha256']
            or not proof['training_allowed'] or not diagnostic['actual_baselines_exact']
            or not diagnostic['all_teacher_targets_1221_FP0']):
        raise ValueError('Full failed-trial, representation and diagnostic evidence required')
    for path, digest in config['sources'].items():
        if sha256(path) != digest:
            raise ValueError('Frozen near-training source changed: '+path)
    runtimes = {}; model_hashes = {}
    for variant, result in trial['variants'].items():
        model = RUN/variant/'final.pt'; digest = sha256(model)
        if digest != result['checkpoint_sha256'] or result['fit']['truth_count'] != 1221:
            raise ValueError('Near-model checkpoint or denominator changed')
        if digest != diagnostic['frozen_checkpoint_sha256'][variant]:
            raise ValueError('Diagnostic model differs')
        raw_path = Path('weld_pipeline/reports/near_event_runtime_156_20261001')/variant/'complete.json'
        raw = json.loads(raw_path.read_text('utf8'))
        if not raw['all_checks_passed'] or raw['checkpoint_sha256'] != digest:
            raise ValueError('Actual PCD-only replay not verified')
        if any(p['probability_replay_max_error'] != 0. or not p['predictions_exact']
               for p in raw['pieces'].values()):
            raise ValueError('Exact runtime replay required for the declared result')
        runtimes[variant] = dict(receipt=str(raw_path), sha256=sha256(raw_path))
        model_hashes[variant] = digest
    for path, digest in {**prior['protected_checkpoint_sha256'], **prior['original_data_hashes']}.items():
        if sha256(path) != digest:
            raise ValueError('Protected original/checkpoint changed: '+path)
    lines = [
        '# 近事件定位与边界错因验证',
        '', '合同目标继续执行。两组新模型未达到采用门槛，完整合同未通过。',
        '', '## 实际完成',
        '', '固定同16个训练来源、全部1221个水平直线子段，将事件支持范围从64毫米改为16毫米；原候选、起点/终点/内部切分三类角色、原始标签、97/255维输入及解码参数保持。',
        '', '先验证7个物理案例及正确标签的完整表示：1221/1221、误报0。包含亚采样位置、6.268毫米短段、相邻短尾、多个事件和反向遍历。当前批次最短长度不作为合同最小长度规则。',
        '', '再实际训练两组120轮×64步，采用相同seed151初始化、相同点/来源采样和优化器；第一组保持相同近点的原物理损失尺度，第二组使用毫米损失尺度。梯度、负偏移显式掩码及两个损失单位等价检查通过。没有重新拟合或改写旧模型。',
        '', '| 模型 | 完整训练子段检出 | 提取率 | 误报 | FP/G | 结论 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    base = trial['baseline']
    lines.append(f"| 154旧整件模型 | {base['valid_children']}/1221 | {base['valid_extraction_rate']:.2%} | {base['false_predictions']} | {base['false_rate_per_truth']:.2%} | 失败 |")
    for name, result in trial['variants'].items():
        fit = result['fit']
        lines.append(f"| {name} | {fit['valid_children']}/1221 | {fit['valid_extraction_rate']:.2%} | {fit['false_predictions']} | {fit['false_rate_per_truth']:.2%} | 失败，未采用 |")
    lines += ['', '第一组误报以机器完整收据619个为准；先前口头加总719个有误，已更正。训练容量改善不能算独立来源泛化或合同验收。两组都未通过，因此没有读取新的9件几何留出或两件观察数据来调参。',
              '', '## 错因干预诊断', '',
              '固定已完成模型的全网格预测，分别替换前景、起终点、内部切分为正确监督，只用于离线诊断。运行入口不读取这些正确值；没有发布任何带真值的推理版本。', '',
              '| 第一组诊断 | TP/G | FP |', '|---|---:|---:|']
    for key, title in [('learned', '全部学习值'), ('teacher_FG', '仅正确前景'),
                       ('teacher_outer', '仅正确起终点'), ('teacher_cut', '仅正确内部切分'),
                       ('teacher_outer_cut', '正确起终点与切分，保留学习前景')]:
        score = diagnostic['results']['physical_matched'][key]
        lines.append(f"| {title} | {score['valid_children']}/1221 | {score['false_predictions']} |")
    lines += ['', '这批训练数据中的主要缺口在边界事件学习，单独纠正前景不足以解决。正确边界下现有前景可还原全部子段；不能据此把学习前景单独判定为全合同通过，也不能用真实边界替代模型运行。下一轮应进一步分离事件存在判别和位置回归的误差，并核查边界所需的有限接触/上下文信息；不盲目追加半径或轮次。',
              '', '## 原始运行与保护', '',
              '两组各对207-FR79M、203-TT1A执行实际原始PCD读取与推理；四次热图最大差均为0，子段预测逐值相同。读取隔离及三个负控制通过，未使用STEP、Excel、真值或预处理缓存。这只证明训练/运行一致，实际准确率仍失败。',
              '', f"重新复核{len(prior['original_data_hashes'])}个原始输入/参考文件及115、v4、v5r1三个保护检查点的哈希，均不变。所有新训练、失败、源码快照和机器收据保留。", '',
              '## 全合同剩余项', '',
              '完整几何提取率严格大于95%、误报率不超过5%，端点及整线误差不超过3毫米，整体及各适用类别仍未联合通过。立焊、曲线、焊接侧与全部必交属性保持原合同范围；本轮没有将未训练类别算作不适用。', '',
              '板高采用“实际接触板及局部底板基准”。板厚、带方向法向等现有STEP派生候选仍未审核；方向正负、板侧对应、完整逐字段及沿线真值和正式封存200件验收仍未完成。运行时间按用户要求暂缓优化，10秒指标尚未通过。', '',
              '机器证据：`weld_pipeline/reports/near_event_followup_156_157_20261001/completion_receipt.json`。']
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf8')
    OUT.mkdir(parents=True)
    receipt = dict(status='TWO_NEAR_EVENT_TRIALS_AND_DIAGNOSTICS_COMPLETED_FULL_CONTRACT_NOT_ACHIEVED',
        models_completed=2, config_sha256=sha256(RUN/'config.json'),
        report=str(REPORT), report_sha256=sha256(REPORT),
        model_checkpoint_sha256=model_hashes, target_proof_sha256=sha256(proof_path),
        diagnostic_sha256=sha256(diagnostic_path), runtime_proofs=runtimes,
        trial_complete_sha256=sha256(RUN/'complete.json'), prior_protection_receipt_sha256=sha256(prior_path),
        protected_checkpoint_sha256=prior['protected_checkpoint_sha256'],
        original_data_hashes=prior['original_data_hashes'], all_originals_unchanged=True,
        adopted_model=None, held_or_observed_used_for_tuning=False,
        formal_test_200_available=False, reviewed_attribute_truth_complete=False,
        speed_gate='DEFERRED_BY_USER', full_contract_pass=False, goal_should_remain_active=True,
        finalizer_sha256=sha256(__file__))
    (OUT/'completion_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')
    print('NEAR_EVENT_ROUND_FINALIZED', 'MODELS', 2, 'ORIGINAL_FILES', len(prior['original_data_hashes']),
          'PROTECTED_WEIGHTS', len(prior['protected_checkpoint_sha256']), 'FULL_CONTRACT_NOT_ACHIEVED', flush=True)


if __name__ == '__main__':
    main()
