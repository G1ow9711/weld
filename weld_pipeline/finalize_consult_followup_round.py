"""Freeze continued-training results, input protection, and remaining work."""
import json
from pathlib import Path

from training.labels import sha256


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def main():
    out=Path('weld_pipeline/reports/consult_followup_20261001')
    if out.exists():
        raise ValueError('Frozen continued-work report already exists')
    integration=read('weld_pipeline/reports/structured_height_v5r1_integration_v3_20261001/complete.json')
    if not integration['integration_passed']:
        raise ValueError('Candidate integration has not passed')
    audit=read('weld_pipeline/reports/contract_child_20260930/audit.json');originals={}
    for piece in audit['independent_workpieces']:
        if sha256(piece['pcd'])!=piece['pcd_sha256']:
            raise ValueError('Original PCD changed: '+piece['pcd'])
        originals[piece['pcd']]=piece['pcd_sha256']
        for path,digest in piece['reference_hashes'].items():
            if sha256(path)!=digest:
                raise ValueError('Original reference changed: '+path)
            originals[path]=digest
    for row in audit['excel_headers']:
        if sha256(row['path'])!=row['sha256']:
            raise ValueError('Original workbook changed')
        originals[row['path']]=row['sha256']
    cad=read('weld_pipeline/reports/step_geometry_20260930/summary.json')
    for row in cad:
        detail=read(Path('weld_pipeline/reports/step_geometry_20260930')/(row['name']+'.json'))
        if sha256(detail['step'])!=row['step_sha256']:
            raise ValueError('Original STEP changed')
        originals[detail['step']]=row['step_sha256']
    baseline=Path('weld_pipeline/runs/typed_branch_surface_v3_seed115/best.pt')
    retained=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt')
    candidate=Path('weld_pipeline/runs/contract_structured_height_candidate_v5r1/candidate.pt')
    if sha256(baseline)!=audit['source_checkpoint_sha256'] or sha256(retained)!='5ecfff09246fcfb2c1174b5348dd5b7de917e48cc54d5d35061af56d19dfb692':
        raise ValueError('Protected checkpoint changed')
    trials={}
    folders=dict(trial141='canonical_horizontal_heatmaps_seed141',trial142='contextual_local_height_seed142',
        trial143='joint_relation_crossfit_seed143',trial146='continuous_local_height_seed146',trial147='joint_event_capacity_seed147')
    for key,folder in folders.items():
        root=Path('weld_pipeline/runs')/folder;config=read(root/'config.json');complete=read(root/'complete.json')
        for source,digest in config['sources'].items():
            if sha256(source)!=digest:
                raise ValueError('Frozen trial source changed: '+source)
        trials[key]=dict(path=str(root),config_sha256=sha256(root/'config.json'),
                        complete_sha256=sha256(root/'complete.json'),result=complete)
        if 'heads_sha256' in complete and sha256(root/'heads.joblib')!=complete['heads_sha256']:
            raise ValueError('Fixed trial heads changed')
    rows=[]
    for name in ('224-GR24A1','207-FR79M','232-TT1A','224-FR71A'):
        root=Path('weld_pipeline/reports/continuous_raw_v2_20261001')/('raw_'+name)
        verification=read(root/'raw_verification.json');score=read(root/'joint_height_diagnostic.json')
        if verification['negative_controls']!=3 or not verification['sources_unchanged'] or not verification['input_unchanged']:
            raise ValueError('Required raw verification failed')
        old=read(Path('weld_pipeline/reports/finite_base_uv_20261001')/('raw_'+name)/'joint_height_diagnostic.json')
        rows.append(dict(name=name,geometry=score['geometry'],height=score['height_counts'],
                         frozen140_UV_height=old['height_counts'],raw_receipt_sha256=sha256(root/'raw_verification.json')))
    physical=read('weld_pipeline/reports/continuous_sampling_20261001/synthetic_receipt.json')
    context=read('weld_pipeline/reports/fresh_point_context_20261001.json')
    if not physical['passed'] or not context['passed']:
        raise ValueError('Physical or training/runtime consistency check failed')
    out.mkdir(parents=True)
    report=Path('合同/按咨询建议续训与验证_20261001.md')
    if report.exists():
        raise ValueError('Preserve existing continued-work report')
    lines=['按咨询建议续训与验证 — 2026年10月1日','',
        '合同目标仍未达到。已实际完成五项续训试验、两类有限表面关联改动、原始PCD隔离评分和新开发候选的集成检查。计算耗时优化继续后置。以下所有参考、拟合、开发留出和原始联合结果分开报告。','',
        '运行条件：AllFaces纯XYZ输入，米转毫米；STEP仅作离线核查和监督。原输入/标签保留。几何按全部有序真实子段和整线采样计数，端点与整线≤3mm；整体及适用类别TP/G严格>95%、FP/G≤5%。两侧板厚≤1mm、带方向单位法向≤3°，高度/距离≤3mm；每个字段含漏检和未知分别>95%。未知、未审核和未测类别均不能算通过。','',
        '数据边界：25个原始FIT来源，其中原4个DEV已是拟合数据；16增强未当独立来源。232-TT1A与224-FR71A合计128子段，是已观察的开发工件；不称封存盲测。三折按原始来源分组，增强父来源核查完成，但未证明这些组等同业务拓扑家族。24个弧线子段集中一个来源，其中一折训练没有正曲线；完整立焊/曲线独立验证及正式200件未完成。','',
        '| 试验 | 实際训练及结果 | 决定 |','|---|---|---|',
        '| 141 | 1,169,178行、四个共享事件树头；横焊FIT1466/1467、FP2；两观察89/128、FP7 | 泛化低于保留几何，不替换 |',
        '| 142 | 101维接触上下文、端点±3/±1/0扰动、128树；参考FIT2121/2392 | 关联覆盖不足，未新评观察 |',
        '| 143 | 三折×三种子×两结构，共18个神经网络，固定24轮 | 来源留出采用门槛失败 |',
        '| 146 | 连续采样关联、同101维/128树；FIT2376/2392，两观察真实端点条件248/256 | 接入另一个开发候选，完整字段仍失败 |',
        '| 147 | 原fold0两模型由24继续至120轮；无消息588/1221、FP656；消息492/1221、FP785 | 训练容量门槛失败，未新读留出/观察 |','',
        '143的完整H直线来源留出分母为1443，V只作无标签邻接，曲线未借用全FIT模型评分。末轮和解码在训练前固定：','',
        '| 种子 | 无消息TP/1443、FP | 有消息TP/1443、FP | FN+FP降幅 | 改善来源/25 |','|---|---:|---:|---:|---:|']
    for row in trials['trial143']['result']['comparisons']:
        a,b=row['no_messages'],row['messages']
        lines.append(f"| {row['seed']} | {a['valid_children']}/1443、{a['false_predictions']} | {b['valid_children']}/1443、{b['false_predictions']} | {row['error_reduction']*100:.2f}% | {row['improved_sources']}/25 |")
    lines+=['','采用要求为每个种子错误下降至少20%、多数来源改善且TP/FP均不恶化；三个均未通过。训练来源容量诊断又发现只有约22%子段正确、稀疏起终/cut漏检；147调整监督及延长训练后仍未通过，不能把这次小型图试验称接头图理论不可行或能力上限。后续应重建有限板面/材料实例和业务事件关联，先通过训练容量，再做来源留出。','',
        '139逐行回放完成：236311行特征/标签/来源权重准确重建，训练行阈值错误0；完整推理网格中的未采样事件有大量假阳性。未发现“后3个工艺标签未知导致几何监督被排除”的问题。108项等价姿态规范化/起终恢复检查通过。正确热图送入固定解码器1605/1605、FP0，只是本批表示上限；不算模型识别成绩。','',
        '板高按用户确认“实际接触板及局部底板基准”。78个失败端点中，76有局部规律边界采样夹住。新算子区分采样点距离和连续有限面几何误差；相邻间隔不超过局部中位数1.8倍才插值，拒绝长空洞和外推；底板需要二维支持三角片及材料侧探针。21个物理反例通过，含倾斜、平移、窄板、孔/分离岛、单线/两轮廓行和2.99/3.01mm内外区分。合同3mm门槛未放宽，隐藏孔洞与实体身份仍未被证明。','',
        '修正了一个原始核验实现错误：146首版错误复用V专用上下文，H的50维结构与6维尺寸被置零，造成FR退化。另建全直线PCD上下文，三件H输入对训练缓存逐值一致、V对旧函数逐值一致；四件原始隔离按同一固定模型重做。旧错误收据完整保留，没有将首版失败改写成通过。','',
        '| 原始PCD | 几何TP/G、FP | 140+UV起/终板高 | 146修正后起/终板高 |','|---|---:|---:|---:|']
    for row in rows:
        g=row['geometry'];old=row['frozen140_UV_height'];new=row['height'];n=g['truth_count']
        lines.append(f"| {row['name']} | {g['valid_children']}/{n}、{g['false_predictions']} | {old['start_height']['new_candidate_valid']}/{n}、{old['end_height']['new_candidate_valid']}/{n} | {new['start_height']['new_candidate_valid']}/{n}、{new['end_height']['new_candidate_valid']}/{n} |")
    lines+=['','两观察板高联合从88/128、91/128提高为94/128、94/128；漏检/未知计入分母，参考仍NOT_REVIEWED。几何保留97/128=75.78%、FP29=22.66%，这是必须继续解决的模型失败，不能由候选端点条件96.88%掩盖。GR中126个V子段的高度角色未冻结，全部仍算高度失败。','',
        '已生成可运行开发候选：`E:/点云数据提取焊缝/weld_pipeline/runs/contract_structured_height_candidate_v5r1/candidate.pt`。一份检查点包含原v4几何/板厚法向及146板高头。新入口为`weld_pipeline.predict_structured_height_candidate_v2`；CSV及子段Excel同步，板厚法向五点剖面和板高端点剖面按子段键关联，未实现字段显式unknown。首版漏传plate_context的失败已保留，修订版已补齐。','',
        '两件集成核查通过：原始PCD重建；builtins.open/io.open/Path.open三个反例拒绝STEP/Excel/参考/训练缓存；几何完全一致；高度数值等价≤1e-6mm且状态/结构一致；CSV与子段Excel、剖面/曲线键、单位法向、未知空值和PCD/权重哈希一致。旧严格浮点相等发现最大约1e-14mm差异，仅修改核验为1e-6mm数值等价，没有改变模型或合同门槛。集成通过不能称合同验收。','',
        '调用示例：在项目目录运行 `.venv/Scripts/python.exe -m weld_pipeline.predict_structured_height_candidate_v2 --pcd <AllFaces.pcd> --checkpoint weld_pipeline/runs/contract_structured_height_candidate_v5r1/candidate.pt --outdir <新的输出目录>`。默认旧入口和115/v4权重保留；新入口只作为开发候选供检查。','',
        '仍待完成：几何跨来源泛化；有限材料身份/同体边与真实接头/业务应焊判别；真实事件和分段理由；法向正负、板侧与焊接侧规则冻结；审核全子段/沿线板厚、法向、高度等真值；挡板−1等状态语义、过孔、整体底板外边距、檐子等字段完整估计；V/曲线角色与独立数据、BM完整曲线、正式200件验收。现有数据不能把这些缺项计作成功，算法仍有明确失败；运行时间优化继续暂缓。','',
        '下一阶段：先建立PCD有限板面/材料实例与接头角色多候选，再在FIT中诊断稀疏事件和负接触；保证训练/原始运行输入一致，训练容量通过后才进行来源分组、多种子验证。缺审核真值不作为停止独立几何工作的理由。持续合同目标保持进行中，不标complete。','',
        f"数据保护复核：{len(audit['independent_workpieces'])}份原PCD及参考、{len(audit['excel_headers'])}份Excel、{len(cad)}份STEP与115/v4权重哈希一致。候选SHA256：{sha256(candidate)}。本轮所有训练已结束；没有进行10秒性能验收。",'',
        '完整机器收据：`weld_pipeline/reports/consult_followup_20261001/completion_receipt.json`。各试验源码、采样数组、冻结配置、完整分母评分及首版失败均保留。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    receipt=dict(status='CONTINUED_TRAINING_AND_CANDIDATE_INTEGRATED_FULL_CONTRACT_NOT_ACHIEVED',
        report=str(report),report_sha256=sha256(report),trials=trials,raw=rows,integration=integration,
        original_data_hashes=originals,original_pcd_sets=len(audit['independent_workpieces']),
        original_excel=len(audit['excel_headers']),original_step=len(cad),protected_115_sha256=sha256(baseline),
        protected_v4_sha256=sha256(retained),candidate=str(candidate),candidate_sha256=sha256(candidate),
        all_training_trials_completed=True,speed_gate='DEFERRED_BY_USER',formal_test_200_available=False,
        customer_attribute_truth_reviewed=False,full_contract_pass=False,goal_should_remain_active=True)
    (out/'completion_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf8')
    (candidate.parent/'validated_development_receipt.json').write_text(json.dumps(dict(
        integration_passed=True,checkpoint_sha256=sha256(candidate),report=str(report),
        integration_receipt='weld_pipeline/reports/structured_height_v5r1_integration_v3_20261001/complete.json',
        full_contract_pass=False,customer_attribute_truth_reviewed=False),indent=2),encoding='utf8')
    print('FOLLOWUP_REPORT_READY',str(report),receipt['status'],flush=True)


if __name__=='__main__':
    main()
