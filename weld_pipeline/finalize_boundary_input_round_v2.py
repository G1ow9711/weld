"""Bind the three trained trials, ten raw replays, coverage diagnostics and originals."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from training.labels import sha256

OUT = Path('weld_pipeline/reports/boundary_input_round_160_163_20261002_v2')
REPORT = Path('合同/边界定位与工艺输入核查_20261002_v2.md')
RUN = Path('weld_pipeline/runs/near_event_consistency_160')


def read(path):
    return json.loads(path.read_text('utf8'))


def main():
    if OUT.exists() or REPORT.exists():
        raise ValueError('Preserve completed boundary-input round')
    trial = read(RUN/'complete.json'); cfg = read(RUN/'config.json')
    if trial['models_completed'] != 3 or trial['selected'] is not None or trial['observed_check_read']:
        raise ValueError('Expected three frozen unsuccessful controls')
    if sha256(RUN/'config.json') != trial['config_sha256']:
        raise ValueError('Original training configuration changed')
    for path, digest in cfg['sources'].items():
        if sha256(path) != digest:
            raise ValueError('Completed training source changed: '+path)
    model_hashes = {}; model_runtimes = {}
    for name, result in trial['variants'].items():
        path = RUN/name/'final.pt'; digest = sha256(path)
        if digest != result['checkpoint_sha256'] or result['fit']['truth_count'] != 1221:
            raise ValueError('Fixed model or full denominator changed')
        raw_path = Path('weld_pipeline/reports/near_consistency_runtime_160_20261001')/name/'complete.json'
        raw = read(raw_path)
        if (not raw['all_checks_passed'] or raw['checkpoint_sha256'] != digest
                or len(raw['pieces']) != 2 or raw['held_or_observed_read']):
            raise ValueError('Two actual pure-PCD model replays required')
        for piece in raw['pieces'].values():
            if (not piece['all_checks_passed'] or not piece['predictions_exact']
                    or piece['probability_replay_max_error'] != 0. or piece['negative_controls'] != 3):
                raise ValueError('Exact model replay and all negative controls required')
        for path, frozen in raw['sources'].items():
            if sha256(path) != frozen:
                raise ValueError('Frozen raw verifier source changed')
        model_hashes[name] = digest; model_runtimes[name] = dict(path=str(raw_path), sha256=sha256(raw_path))
    base_path = Path('weld_pipeline/reports/physical_event_candidates_161_20261002/complete.json')
    flags_path = Path('weld_pipeline/reports/physical_event_process_flags_162_20261002/complete.json')
    comp_path = Path('weld_pipeline/reports/contact_component_candidates_163_20261002/complete.json')
    base, flags, comp = read(base_path), read(flags_path), read(comp_path)
    if (base['coverage']['finite']['children'] != 1221 or comp['totals']['truth_children'] != 1221
            or comp['totals']['truth_endpoints'] != 2442 or flags['original_child_endpoint_count'] != 2442
            or comp['pool_limited_runtime_adopted'] or base['pure_pool_limited_runtime_adopted']):
        raise ValueError('Full scoped candidate diagnostics without adoption required')
    proposal_proofs = {}
    for label, audit_path, raw_path, error_key in (
            ('finite_planes', base_path, Path('weld_pipeline/reports/physical_event_runtime_161_20261002/complete.json'), 'pool_position_max_error_mm'),
            ('contact_components', comp_path, Path('weld_pipeline/reports/contact_component_runtime_163_20261002/complete.json'), 'pool_max_error_mm')):
        raw = read(raw_path)
        audit_binding = raw.get('audit_complete_sha256', raw.get('audit_sha256'))
        if not raw['all_checks_passed'] or len(raw['pieces']) != 2 or audit_binding != sha256(audit_path):
            raise ValueError('Two raw-PCD candidate-pool rebuilds required')
        if any(not p['all_checks_passed'] or p[error_key] != 0. or p['negative_controls'] != 3
               for p in raw['pieces'].values()):
            raise ValueError('Candidate-pool raw reconstruction is not exact')
        for path, digest in raw['sources'].items():
            if sha256(path) != digest:
                raise ValueError('Actual candidate runtime source changed')
        proposal_proofs[label] = dict(path=str(raw_path), sha256=sha256(raw_path))
    for folder in (base_path.parent, comp_path.parent):
        complete = read(folder/'complete.json'); protocol = read(folder/'config.json')
        if sha256(folder/'config.json') != complete['config_sha256']:
            raise ValueError('Candidate protocol changed')
        for path, digest in protocol['sources'].items():
            if sha256(path) != digest:
                raise ValueError('Candidate diagnostic source changed')
        for name, digest in complete['piece_receipts'].items():
            piece_path = folder/(name+'.json'); piece = read(piece_path)
            if sha256(piece_path) != digest or sha256(folder/(name+'_pools.npz')) != piece['pool_sha256']:
                raise ValueError('Full diagnostic artifacts changed')
    if (sha256(flags_path.parent/'endpoint_rows.json') != flags['rows_sha256']
            or sha256('weld_pipeline/audit_physical_event_process_flags.py') != flags['source_sha256']):
        raise ValueError('Original process-flag audit changed')
    prior_path = Path('weld_pipeline/reports/near_event_followup_156_157_20261001/completion_receipt.json')
    prior = read(prior_path)
    for path, digest in {**prior['protected_checkpoint_sha256'], **prior['original_data_hashes']}.items():
        if sha256(path) != digest:
            raise ValueError('Protected original/checkpoint changed: '+path)
    correction = flags_path.parent/'semantic_correction.md'
    if not correction.exists() or not comp['original_IsSplitedForDualRobot_means_side_standing_plate_split']:
        raise ValueError('Confirmed side-standing-plate semantics required')
    time = datetime.now(timezone(timedelta(hours=8))).isoformat()
    lines = ['# 边界定位与工艺输入核查', '', f'核查时间：{time}。完整合同仍未达到，持续目标保持执行。',
        '', '## 已完成的实际工作', '',
        '补齐上轮三个120轮×64步模型的六次原始PCD回放；热图最大差均为0，子段预测逐值相同，三种读取负控制均通过。运行只读取该件原始PCD及冻结模型，未使用STEP、Excel、真值或预处理缓存。三个模型仍失败，没有新采用模型。', '',
        '| 上轮固定训练对照 | 有效子段 | 提取率 | 误报 | FP/G |', '|---|---:|---:|---:|---:|']
    for name, result in trial['variants'].items():
        fit = result['fit']
        lines.append(f"| {name} | {fit['valid_children']}/1221 | {fit['valid_extraction_rate']:.2%} | {fit['false_predictions']} | {fit['false_rate_per_truth']:.2%} |")
    lines += ['', '本轮新增训练模型为0；完成三个既有训练模型的运行核验、两种纯PCD几何候选的完整覆盖诊断和各两次原始重建，共10次实际原始PCD回放。没有把诊断、复制文件或回放次数计作模型训练。', '',
        '## 几何候选覆盖', '',
        '只用点云派生面、有限范围和接触分量边界构造位置池；不读取标签生成候选，不加入均匀网格，不删除短段、合并相邻真实边界或加入标签推导的退让常数。正确标签仅用于离线评价。', '',
        '| 候选来源 | 子段两端均在3mm内 | 所有子段端点覆盖 |', '|---|---:|---:|',
        f"| 有限面交点及原候选两端 | {base['coverage']['finite']['covered_children']}/1221（{base['coverage']['finite']['covered_child_fraction']:.2%}） | {base['coverage']['finite']['covered_endpoints']}/2442 |",
        f"| 再加入实际接触分量两端 | {comp['totals']['covered_children']}/1221（{comp['covered_child_fraction']:.2%}） | {comp['totals']['covered_endpoints']}/2442 |", '',
        '这些是候选位置的覆盖上限，不是模型提取率或验收成绩。两种候选各在207-FR79M、203-TT1A上从原始PCD重新构建，候选及有限面/接触分量与缓存逐值一致、最大差0；读取隔离通过。物理短段、相邻边界、平移、绕重力轴旋转、反向和符号检查通过。', '',
        '两种候选均不足以限定模型输出；没有启用强制吸附或只从池中选点的解码。接触分量方法仍缺989个真实端点。预检曾因刚体变换后数值重复位置数量变化而失败；保留失败源码，改用几何集合双向最大误差检查，未修改候选生成或去重规则。', '',
        '## 工艺标记与遗漏', '',
        '原始343个显式工作范围切分端点，有限面池缺325个，加入接触分量后仍缺315个。带工作范围Any标记的1096个端点中，接触分量池缺764个；不带该标记的1346个端点也缺225个。因此工作范围规则与遗漏明显相关，但不能把全部问题归结为配置缺失。内部切分端点112个仍有30个缺口。', '',
        '训练资料中存在跨六件重复的工作区坐标模式，旧过程平面模型也曾从训练标签学习这些共享位置。它们只是开发候选证据，坐标基准、变换和端部退让规则尚未冻结；没有把这些坐标写成查表输出或自动当作客户已确认规则。已异步请求生成标注的工艺配置/源码路径，或工作范围坐标基准、边界和端部退让规则；尚未收到答复。项目源及DataSet没有检索到对应配置文件。', '',
        'IsSplitedForDualRobot已确认表示“被侧边立板切分”。此前配置问题中把它写成“双机器人分区”有误，已向用户更正；保留原字段名及原标注，按侧立板语义处理，不再请求机器人分工规则。', '',
        '## 下一项实际改进', '',
        '工艺参数依赖项等待现有配置或规则证据。独立几何继续核查接触面连续边界、非竖向端面和孔边：207-FR79M未带工作范围标记的若干遗漏仍约34.815mm，现缓存没有圆弧候选，不能仅凭距离推断圆孔原因。先核查原始PCD与离线STEP的实际局部面、边界及采样覆盖，再冻结改动和训练；不盲目继续追加相同损失或轮次。', '',
        '## 合同状态与保护', '',
        '保留开发候选两件既存观察结果仍97/128（75.78%）、误报29（22.66%）；该结果非盲测。整体及各适用类别要求TP/G严格大于95%、FP/G不超过5%、端点及整线最大偏差不超过3mm，仍未联合通过。立焊、曲线、焊接侧、全部必交属性、客户审核真值、正式200件、离线交付复验均继续在目标范围内。', '',
        '板高基准保持已确认的实际接触板及局部底板。板厚和带方向法向等STEP派生候选未自动升级为审核真值。法向正负/板侧确认问题仍未解决。运算时间按用户要求后置，10秒指标尚未通过。DataSet仍为9月20日的22件目录，没有发现新正式200件清单。', '',
        f"重新核对{len(prior['original_data_hashes'])}个原始输入/参考文件和{len(prior['protected_checkpoint_sha256'])}个115/v4/v5r1保护权重哈希，全部保持；三个本轮核验的最终模型哈希也一致。没有改写原标签、原输入或旧权重。", '',
        '机器证据：`weld_pipeline/reports/boundary_input_round_160_163_20261002/completion_receipt.json`。']
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf8')
    OUT.mkdir(parents=True)
    receipt = dict(status='PROGRESS_FULL_CONTRACT_NOT_ACHIEVED', finished_local=time,
        newly_trained_models_this_turn=0, previously_trained_models_verified=3,
        model_raw_replay_count=6, geometric_proposal_raw_replay_count=4, total_raw_replays=10,
        model_hashes=model_hashes, model_runtime_proofs=model_runtimes, proposal_runtime_proofs=proposal_proofs,
        trial_complete_sha256=sha256(RUN/'complete.json'), geometric_base_sha256=sha256(base_path),
        process_flags_sha256=sha256(flags_path), contact_component_sha256=sha256(comp_path),
        semantic_correction_sha256=sha256(correction), source_preservation_receipt_sha256=sha256(prior_path),
        protected_checkpoint_sha256=prior['protected_checkpoint_sha256'], original_data_hashes=prior['original_data_hashes'],
        all_originals_unchanged=True, adopted_model=None, pool_limited_runtime_adopted=False,
        report=str(REPORT), report_sha256=sha256(REPORT), pending_workrange_config_answer=True,
        IsSplitedForDualRobot_means_side_standing_plate_split=True,
        held_or_observed_read_for_tuning=False, formal_test_200_available=False,
        reviewed_all_attribute_truth_complete=False, full_contract_pass=False, goal_should_remain_active=True,
        speed_gate='DEFERRED_BY_USER', finalizer_sha256=sha256(__file__))
    (OUT/'completion_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
    print('BOUNDARY_INPUT_ROUND_FINALIZED', 'RAW_REPLAYS', 10, 'ORIGINAL_FILES', len(prior['original_data_hashes']),
          'PROTECTED_WEIGHTS', len(prior['protected_checkpoint_sha256']), 'FULL_CONTRACT_NOT_ACHIEVED', flush=True)


if __name__ == '__main__':
    main()
