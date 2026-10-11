"""Freeze real material/event trials, batch verification, and unchanged-input evidence."""
import collections
import json
from pathlib import Path

from training.labels import sha256

OUT=Path('weld_pipeline/reports/material_event_followup_20261001')
REPORT=Path('合同/材料身份与事件训练验证_20261001.md')
USAGE=Path('合同/开发候选批量调用说明_20261001.md')


def read(path):return json.loads(Path(path).read_text('utf8'))


def bind_sources(config):
    sources=config.get('source_sha256',config.get('sources',{}))
    for path,digest in sources.items():
        if sha256(path)!=digest:raise ValueError('Frozen source changed: '+path)


def main():
    if OUT.exists() or REPORT.exists() or USAGE.exists():raise ValueError('Preserve finalized evidence')
    previous_path=Path('weld_pipeline/reports/consult_followup_20261001/completion_receipt.json')
    previous=read(previous_path)
    if sha256(previous['report'])!=previous['report_sha256']:raise ValueError('Previous report changed')
    for path,digest in previous['original_data_hashes'].items():
        if sha256(path)!=digest:raise ValueError('Original PCD/label/Excel/STEP changed: '+path)
    protected={
        'weld_pipeline/runs/typed_branch_surface_v3_seed115/best.pt':previous['protected_115_sha256'],
        'weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt':previous['protected_v4_sha256'],
        previous['candidate']:previous['candidate_sha256']}
    for path,digest in protected.items():
        if sha256(path)!=digest:raise ValueError('Protected baseline/candidate changed')
    material_root=Path('weld_pipeline/runs/material_support_crossfit_seed148')
    material=read(material_root/'complete.json');bind_sources(read(material_root/'config.json'))
    if material['eligible_for_later_geometry_trial']:raise ValueError('Report requires actual failed material adoption gate')
    for name,digest in material['checkpoint_sha256'].items():
        if sha256(material_root/name)!=digest:raise ValueError('Material classifier changed')
    data_root=Path('weld_pipeline/data/material_support_v1');data=read(data_root/'complete.json')
    bind_sources(read(data_root/'config.json'))
    for name,digest in data['manifests'].items():
        if sha256(data_root/name/'manifest.json')!=digest:raise ValueError('Material source feature manifest changed')
        for file,expected in read(data_root/name/'manifest.json')['files'].items():
            if sha256(data_root/name/file)!=expected:raise ValueError('Material feature/label evidence changed')
    pilot_root=Path('weld_pipeline/runs/joint_event_vote_capacity_seed151')
    pilot=read(pilot_root/'complete.json');bind_sources(read(pilot_root/'config.json'))
    if sha256(pilot_root/'final.pt')!=pilot['checkpoint_sha256']:raise ValueError('Frozen voting baseline changed')
    ablation_root=Path('weld_pipeline/runs/joint_event_vote_ablation_152')
    ablation=read(ablation_root/'complete.json');bind_sources(read(ablation_root/'config.json'))
    for name,result in ablation['variants'].items():
        if sha256(ablation_root/name/'final.pt')!=result['checkpoint_sha256']:raise ValueError('Paired voting model changed')
    if pilot['fit_gate_passed'] or ablation['selected'] is not None:raise ValueError('Actual event trial results differ from report scope')
    batch_path=Path('weld_pipeline/reports/structured_candidate_batch_20261001/complete.json')
    batch=read(batch_path);bind_sources(batch)
    if not batch['all_checks_passed']:raise ValueError('Actual mixed batch not verified')
    raw_path=Path('weld_pipeline/reports/material_support_raw_148_20261001/complete.json')
    raw=read(raw_path)
    if not raw['all_checks_passed']:raise ValueError('Actual material feature runtime differs')
    physical_path=Path('weld_pipeline/reports/signed_material_support_physical_20261001.json')
    physical=read(physical_path)
    if physical['extractor_sha256']!=sha256('weld_pipeline/signed_material_support.py') or not physical['all_passed']:
        raise ValueError('Physical support verification changed')
    audit_root=Path('weld_pipeline/reports/candidate_material_identity_full_148v3_20261001')
    audit=read(audit_root/'complete.json');bind_sources(audit['config'])
    fg=collections.Counter();bg=collections.Counter()
    for result in audit['pieces'].values():
        fg.update(result['foreground_sample_class_counts']);bg.update(result['background_sample_class_counts'])
    collisions_path=Path('weld_pipeline/reports/material_feature_collisions_148_20261001.json')
    collisions=read(collisions_path)
    OUT.mkdir(parents=True)
    lines=['材料身份与事件训练验证 — 2026年10月1日','',
        '合同目标仍未达到。本阶段完成19件STEP离线材料审计、48个材料/应焊分类模型、4个事件定位模型，共52个新模型的实际训练，以及开发候选的批量接口核验。新模型未通过预先固定的采用门槛；115/v4/v5r1均保留。运行时间优化继续暂缓。','',
        '合同口径保留：全部真实有序子段为分母，端点及整线≤3mm、整线双向采样间距≤1mm；整体和适用H/V/直线/曲线均TP/G严格>95%、FP/G≤5%。板厚≤1mm，带方向单位法向≤3°；高度/距离≤3mm，每个字段计漏检/未知分别>95%。用户板高基准仍为实际接触板及局部底板；方向/板侧异步确认未得到答复，不把沉默当冻结。','',
        '数据边界：本材料试验为19个有STEP的原始FIT来源、每个非弧候选固定五个采样比例。STEP只用于离线有限面候选监督，所有参考NOT_REVIEWED。两件观察未用于本阶段选型；缺CAD的原始6件未被当成材料负例。几何小试仅fold0的16训练来源、1221个H直线子段，V/曲线与其余字段不在小试成绩内，完整合同要求仍开放。','',
        f"有限接触核查覆盖16345个固定样点，原标焊1117个全部存在同体与跨体配对；未标焊中{bg['SAME_OR_CROSS_BODY_AMBIGUOUS']}点也存在跨体配对。不能因存在同体厚度边而删真焊，也不能把跨体接触直接等同应焊。两侧多候选保留，未声称唯一材料身份或5点等同整线证明。首版全量审计因222参考缺STEP路径字段停止，15件结果保留；修订仅从同哈希CAD描述解析路径，原始配准/阈值和参考不改，完成19件。",'',
        '166维纯PCD特征保留接触面有符号两侧支持与对应背面。半径160mm、面层0.15mm，沿线8/24/80mm窗及固定横向分箱；法向独立翻转、板侧交换的八种等价帧训练展开/推理平均。没有名称、实体/面ID、绝对XYZ或CAD输入。26项物理/刚体反例通过；FR79M与203从原始PCD重新生成候选和特征，3215行与训练逐值相同，预测重放误差0；每件三个参考/缓存读取负控制、16个实际法向等价帧、输入和模型哈希通过。','',
        '材料候选与原应焊标记分别训练；三种子、三折来源留出、同166维独立排序侧信息作控制。12个全FIT头及36个来源留出头均实际完成。训练样点中材料全对，应焊1117/1117、FP2；不是几何或合同验收。','',
        '| 种子 | 有符号应焊TP/1117 | FP | 控制TP/1117 | 控制FP | FN+FP下降 | 改善来源/19 |','|---|---:|---:|---:|---:|---:|---:|']
    for seed,rows in material['source_held_aggregate'].items():
        new=rows['signed']['business'];control=rows['side_sorted']['business'];decision=material['source_decisions'][seed]
        lines.append(f"| {seed} | {new['true_foreground']} | {new['false_foreground']} | {control['true_foreground']} | {control['false_foreground']} | {decision['error_reduction']*100:.2f}% | {decision['improved_sources']} |")
    lines+=['','应焊来源留出召回88.09%/89.08%/88.90%，三种子均未满足“错误下降20%、多数来源改善且TP/FP不恶化”的固定采用要求，也没有达到合同>95%。跨体材料候选召回约92.6%—93.8%，较控制约84.1%—85.2%改善，但材料候选改善不等于业务焊缝达标。','',
        '精确碰撞审计发现521-CM56A两组局部描述完全相同、原应焊标签不同；八等价帧的不变分类器至少会错2点，与训练结果一致。这是当前局部描述的限制，不是完整点云、任意模型或项目能力上限，也没有据此改标签。需要核对整个工件中的接头位置/材料角色和业务选侧信息。','',
        '事件试验将窄Gaussian热图改为64mm邻域的事件存在/有符号偏移投票。五项亚采样/边界/多个事件/反向物理检查通过，正确投票完整还原1221/1221、FP0；只是表示上限。实际训练120轮后299/1221、FP1087。已冻结权重核查：203端点平均偏移24.20/23.77mm、LB18 5.73/6.38mm、FR80 1.81/2.49mm；低总损失不能当3mm精度通过。','',
        '追加配对2×2定位对照，原局部卷积半径42mm和偏移损失为基线；仅改毫米尺度定位损失、仅扩大到106mm、两者同时，各同初始化前缀/采样/120轮/16来源，目标和显式偏移掩码哈希不变。结果为：','',
        '| 变体 | 完整H直线TP/1221 | FP | 决定 |','|---|---:|---:|---|',
        f"| 原投票151 | {pilot['fit']['valid_children']} | {pilot['fit']['false_predictions']} | 不采用 |"]
    for name,result in ablation['variants'].items():
        lines.append(f"| {name} | {result['fit']['valid_children']} | {result['fit']['false_predictions']} | 不采用 |")
    lines+=['','扩大范围改善了部分训练件，但三个均未通过完整FIT几何门槛，未读新的几何留出或两观察件，也未继续加轮次。只改尺度/范围不足；下一步必须核对模型所见的整个工件相对位置、材料接头角色和事件输出关系，再作有证据的新表示/学习对照。不能把这四个失败神经网络称正确监督不能被学习。','',
        '工程接口新增：`weld_pipeline.structured_candidate_batch`提供单件/批量CLI及run_batch库函数，一个批次共享一份v5r1检查点。按调用者工件ID同步子段CSV/Excel、逐件运行记录、输入/模型/代码哈希和错误码。未知属性留空；正常零输出为空表及SUCCESS_NO_WELDS。损坏/缺失/非有限输入、重复ID、非法条目单独失败并继续后续工件。','',
        '实际混合批次9项：3成功（FR79M36行、BX1 2行、合成无接触0行），6预期失败；错误码、失败后继续、空表表头、CSV/Excel逐格一致、调用者ID、未知空值、三个参考/缓存读取负控制、输入/模型不变均通过。合成无接触例不是正式无焊工件独立验收。该接口仍输出开发候选，不代表精度和全部字段通过。','',
        '保留候选最新已有原始开发检查是97/128子段、FP29；板高按未审核新基准起终各94/128。本阶段未重新评分两观察件，权重/原数据哈希相同，也未把辅助分类召回换成这些子段指标。','',
        '全合同剩余项：现有几何泛化和事件定位；有限材料/接头角色与应焊选择；焊接侧/分段原因、法向正负和板侧语义冻结；审核全子段/沿线板厚、法向、局部高度、孔高、挡板、边距及檐子等字段真值与估计；V/曲线独立验证、BM完整曲线；正式200件及交付完整性、干净环境复现。计算耗时按用户要求后置，尚无10秒性能通过证据。','',
        f"保护复核：{previous['original_pcd_sets']}份原始PCD及参考、{previous['original_excel']}份Excel、{previous['original_step']}份STEP、115/v4/v5r1检查点哈希一致。全部52个训练模型已结束，完整合同未达到，持续目标保留active。",'',
        '机器收据：`weld_pipeline/reports/material_event_followup_20261001/completion_receipt.json`。失败首版、配置、源码快照、采样/目标/权重、逐来源分类和完整子段评分均保留。']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf8')
    split=read('weld_pipeline/data/full_available_fit_v1/split.json');example=[]
    for name in ('207-FR79M','224-BX1A'):
        original=read(Path(split['piece_roots'][name])/'manifest.json')
        example.append(dict(workpiece_id=name,pcd=str(Path(original['record']['input']).resolve())))
    example_path=OUT/'batch_example.json'
    example_path.write_text(json.dumps(example,indent=2,ensure_ascii=False),encoding='utf8')
    schema=dict(version='structured_height_candidate_batch_v1',input=dict(type='list',required_item_fields=['workpiece_id','pcd'],
        unique_nonempty_workpiece_id=True,pcd='existing ASCII XYZ PCD file; input numeric units m'),
        result_statuses=dict(SUCCESS_WITH_CANDIDATES='Inference/export succeeded; geometry and attributes remain development predictions.',
            SUCCESS_NO_WELDS='Successful inference produced zero child rows and a header-only table.',FAILED='Item failed; inspect reason_code and message.'),
        error_codes=read('weld_pipeline/reports/structured_candidate_batch_20261001/runtime_output/batch_summary.json')['error_codes'],
        exit_codes={'0':'all items succeeded','1':'one or more item failures'},
        unknown_attributes='empty value plus explicit unknown state; never zero placeholders',
        batch_summary='one row per requested item, including failures',coordinate_units='mm',
        single_shared_checkpoint=True,full_contract_pass=False)
    (OUT/'batch_api_schema.json').write_text(json.dumps(schema,indent=2,ensure_ascii=False),encoding='utf8')
    usage=['开发候选批量调用 — 2026年10月1日','',
        '当前检查点为v5r1开发候选。批量接口已通过实际混合输入核查；几何和全部字段未达到合同验收，不能作为最终交付通过凭证。','',
        '输入：AllFaces、ASCII XYZ PCD，数值单位米，程序统一转毫米。每项提供唯一非空workpiece_id和pcd路径；同一批次使用同一模型和固定参数。项目目录下可运行：','',
        '```powershell',
        '& .\.venv\Scripts\python.exe -m weld_pipeline.structured_candidate_batch --manifest "weld_pipeline/reports/material_event_followup_20261001/batch_example.json" --checkpoint "weld_pipeline/runs/contract_structured_height_candidate_v5r1/candidate.pt" --outdir "weld_pipeline/reports/我的新批次输出"',
        '```','',
        '库调用：`run_batch(items, checkpoint, outdir)`，来自`weld_pipeline.structured_candidate_batch`。单件CLI改用`--pcd <路径> --workpiece-id <编号>`；输出目录必须不存在，保留旧结果。','',
        '逐件目录含run_record.json及成功件output目录。output内predicted_children.csv、predicted_children.xlsx、child_curves.json及属性剖面按子段键关联。workpiece_id使用调用者编号。未知属性空值保留，焊接侧和未审核字段尚未冻结。','',
        '批次batch_summary.json包含全部请求，包括失败，记录输入/模型/代码哈希、运行时间、状态与错误。失败继续处理其他项。SUCCESS_WITH_CANDIDATES表示程序完成并发布开发预测；SUCCESS_NO_WELDS表示空结果且有表头；FAILED查看reason_code/message。CLI全部运行成功返回0，有失败返回1，精度验收单独评分。','',
        '机读输入/状态/错误码字典：`weld_pipeline/reports/material_event_followup_20261001/batch_api_schema.json`。已验证混合批次收据：`weld_pipeline/reports/structured_candidate_batch_20261001/complete.json`。']
    USAGE.write_text('\n'.join(usage)+'\n',encoding='utf8')
    receipts={name:sha256(path) for name,path in dict(material=material_root/'complete.json',pilot=pilot_root/'complete.json',
        paired_ablation=ablation_root/'complete.json',batch=batch_path,raw_material=raw_path,physical=physical_path,
        offline_audit=audit_root/'complete.json',collision=collisions_path).items()}
    receipt=dict(status='MATERIAL_AND_EVENT_TRIALS_FINISHED_BATCH_VERIFIED_FULL_CONTRACT_NOT_ACHIEVED',
        report=str(REPORT),report_sha256=sha256(REPORT),batch_usage=str(USAGE),batch_usage_sha256=sha256(USAGE),
        evidence_sha256=receipts,models_completed=52,material=material['source_decisions'],
        event_pilot=pilot['fit'],event_ablation={n:r['fit'] for n,r in ablation['variants'].items()},
        material_adoption_eligible=False,event_adoption_eligible=False,batch_checks_passed=True,
        protected_checkpoint_sha256=protected,original_data_hashes=previous['original_data_hashes'],
        all_originals_unchanged=True,all_training_trials_completed=True,
        observed_check_read_for_architecture_tuning=False,formal_test_200_available=False,
        reviewed_attribute_truth_complete=False,speed_gate='DEFERRED_BY_USER',full_contract_pass=False,goal_should_remain_active=True,
        finalizer_sha256=sha256(__file__))
    (OUT/'completion_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf8')
    print('MATERIAL_EVENT_ROUND_FINALIZED','MODELS',52,'BATCH_VERIFIED','FULL_CONTRACT_NOT_ACHIEVED',str(REPORT),flush=True)


if __name__=='__main__':main()
