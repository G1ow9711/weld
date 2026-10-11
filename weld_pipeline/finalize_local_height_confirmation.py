"""Record the actual local-height work, original-data protection and failures."""
import json
import shutil
from pathlib import Path

from training.labels import sha256


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def main():
    out = Path('weld_pipeline/reports/local_height_confirmation_20260930'); out.mkdir(parents=True, exist_ok=True)
    if (out / 'completion_receipt.json').exists():
        raise ValueError('Frozen confirmation receipt already exists')
    audit = read('weld_pipeline/reports/contract_child_20260930/audit.json')
    for piece in audit['independent_workpieces']:
        assert sha256(piece['pcd']) == piece['pcd_sha256']
        for path, wanted in piece['reference_hashes'].items():
            assert sha256(path) == wanted
    for row in audit['excel_headers']:
        assert sha256(row['path']) == row['sha256']
    cad = read('weld_pipeline/reports/step_geometry_20260930/summary.json')
    for row in cad:
        detail = read(Path('weld_pipeline/reports/step_geometry_20260930') / (row['name'] + '.json'))
        assert sha256(detail['step']) == row['step_sha256']
    baseline = Path('weld_pipeline/runs/typed_branch_surface_v3_seed115/best.pt')
    retained = Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt')
    assert sha256(baseline) == audit['source_checkpoint_sha256']
    assert sha256(retained) == '5ecfff09246fcfb2c1174b5348dd5b7de917e48cc54d5d35061af56d19dfb692'
    prior_path = Path('weld_pipeline/reports/step_training_round_20260930/additional_training_receipt.json')
    prior = read(prior_path)
    assert sha256(prior['report']) == prior['report_sha256']
    folders = {'138': 'contact_plate_extent_seed138', '139': 'horizontal_independent_heatmaps_seed139',
               '140': 'local_height_datum_seed140'}
    trials = {}
    for seed, folder in folders.items():
        root = Path('weld_pipeline/runs') / folder
        config = read(root / 'config.json'); receipt = read(root / 'complete.json')
        for path, wanted in config['sources'].items():
            assert sha256(path) == wanted
        checkpoint = root / 'heads.joblib'
        expected = receipt.get('heads_sha256', receipt.get('head_sha256'))
        assert sha256(checkpoint) == expected
        trials[seed] = {'folder': str(root), 'config_sha256': sha256(root / 'config.json'),
                        'receipt_sha256': sha256(root / 'complete.json'), 'receipt': receipt}
    local = Path('weld_pipeline/runs/local_height_datum_seed140'); raw = []
    for version in ('raw_joint_', 'raw_joint_v2_', 'raw_joint_v3_'):
        for name in ('224-GR24A1', '207-FR79M', '232-TT1A', '224-FR71A'):
            folder = local / (version + name)
            receipt = read(folder / 'raw_verification.json'); diagnostic = read(folder / 'joint_height_diagnostic.json')
            assert receipt['guarded_pcd_reads'] > 0 and receipt['negative_controls_passed'] == 3
            assert not receipt['runtime_step_used'] and not receipt['runtime_truth_used']
            raw.append({'version': version, 'name': name, 'geometry_valid': diagnostic['geometry']['valid_children'],
                        'truth_children': diagnostic['geometry']['truth_count'], 'height_counts': diagnostic['height_counts'],
                        'raw_receipt_sha256': sha256(folder / 'raw_verification.json')})
    runtime = out / 'runtime_source_snapshot'; runtime.mkdir(exist_ok=True)
    paths = ['weld_pipeline/local_height_datum_model.py', 'weld_pipeline/local_height_datum_model_v2.py',
             'weld_pipeline/local_height_datum_model_v3.py', 'weld_pipeline/local_base_sampling.py',
             'weld_pipeline/verify_local_height_joint_raw.py', 'weld_pipeline/verify_local_height_joint_raw_v2.py',
             'weld_pipeline/verify_local_height_joint_raw_v3.py', 'weld_pipeline/verify_contact_extent_raw.py', __file__]
    for p in paths:
        shutil.copy2(p, runtime / Path(p).name)
    definition = {'user_confirmed_height_object_and_datum': '实际接触板及局部底板基准',
        'candidate_implementation': 'Same joint standing CAD face and horizontal local base face; opposite board boundary above/below own base.',
        'fixed_face_projection_limit_mm': 3., 'one_sided_child_interior_query_mm': .01,
        'candidate_implementation_reviewed_as_customer_truth': False,
        'original_labels_overwritten': False, 'vertical_and_curve_height_role_frozen': False}
    (out / 'user_definition.json').write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding='utf8')
    table_paths = {}
    for role, title in [('FIT', '板高原标签与局部基准_训练核对_20260930.csv'),
                        ('OBSERVED_CHECK', '板高原标签与局部基准_观察核对_20260930.csv')]:
        target = Path('合同') / title
        shutil.copy2(local / (role + '_板高原标签与局部底板候选.csv'), target)
        table_paths[role] = {'path': str(target), 'sha256': sha256(target)}
    report = Path('合同/板高基准确认与续训核查_20260930.md')
    lines = ['板高基准确认与续训核查 — 2026年9月30日', '',
        '当前完整合同状态：NOT_ACHIEVED。用户已确认板高按“实际接触板及局部底板基准”；已落实到STEP有限面核查、共享模型训练和原始点云联合评分。原数据、旧模型与全部失败记录保留。', '',
        '执行条件已确认：运行输入为AllFaces纯XYZ点云，GW1是顶部面；STEP只作离线核查/监督。真实原始件标全，未标接触线作为负例，部分增强未知区保持掩码。按有序真实子段评分，GR三段/FR一段沿原真值训练，没有“同长度必须同段数”的硬规则。整线/端点≤3mm，整体和适用类别TP/G严格>95%、FP/G≤5%；两侧板厚≤1mm、带方向单位法向≤3°，其余距离/高度≤3mm，各字段含漏检/未知分别>95%。耗时优化仍后置，10秒未计通过。', '',
        '数据角色：原21TRAIN/4DEV与115基线保留；新25真实FIT含原4DEV，2件观察工件232-TT1A、224-FR71A共128子段，已多次观察，不作为封存独立验收。当前19个FIT件有本轮CAD参考；另6个旧补充来源未具有这轮板高CAD参考。正式200件及足够审核全属性真值仍未提供。', '',
        '保留v4几何部分：25FIT为1570/1605、50误报；两观察为97/128=75.78%、29误报。几何泛化与完整字段验收仍失败，旧报告与回执未覆盖。板厚/材料外向法向已有共享候选头和五点对照，尚受审核真值、板侧约定、曲线及全类别覆盖限制。', '',
        '| 新试验 | 完成工作 | 实际结果 |', '|---|---|---|',
        '| 139 | 四个独立64树学习应焊、起点、终点、断点；236311点，无名称/绝对坐标查询 | H FIT1011/1467，FP600；观察108/128，FP9；FIT失败，不采用 |',
        '| 138 | 39维纯PCD板面/局部基准特征，128树学习有限面局部跨度 | FIT候选2823/2940；观察234/256；跨度未等同最终板高 |',
        '| 140 | 固定实际接触板面与本接头局部底板面，生成带来源的板高候选，训练128树共享残差头 | 19FIT横焊2384/2392端点；观察252/256，起终各126/128；仅参考端点条件对照 |', '',
        '140核查保留实际板volume/face和底板volume/face，点偏离有限面时只在该身份内投影，最大3mm；向子段内部0.01mm取单侧剖面。位于底板上方/下方分别记录。板高方向不明确、换板或远面无法证明时保留未知，孔的材料间隔不填成焊缝。', '',
        '旧标签与新基准候选的差异：训练件2264个可比较旧标签中978项超过3mm；观察件232项中68项超过3mm。这是需要核验的差异，不据此自动宣布原标签错误或把候选升为审核真值。示例：', '',
        '| 工件 | 旧标签 | 实际接触板/本接头底板候选 |', '|---|---:|---:|',
        '| 222-TT1A部分端点 | 240mm | 160mm |', '| 527-LB24A1部分端点 | 240mm | 180mm |',
        '| 224-FR71A一端点 | 393mm | 291.8603mm |', '',
        '按E2计入真实几何漏检后，条件98.44%不能代表板高达标。下表是运行时v3的未审核新基准候选联合诊断；两端独立、分母包括全部子段（含未冻结V角色）。原旧标签对照另存JSON。', '',
        '| 工件 | 几何有效/子段 | 起点板高候选有效/全部子段 | 终点板高候选有效/全部子段 |',
        '|---|---:|---:|---:|']
    for row in raw:
        if row['version'] == 'raw_joint_v3_':
            c = row['height_counts']; g = row['truth_children']
            lines.append(f"| {row['name']} | {row['geometry_valid']}/{g} | {c['start_height']['new_candidate_valid']}/{g} | {c['end_height']['new_candidate_valid']}/{g} |")
    lines += ['', '原始运行核查：上述4件均重新读取完整PCD，builtins.open/io.open/Path.open三种反例拒绝STEP、Excel、参考与训练缓存；板高只按实际预测线与新PCD候选关联，未用真值选接触面。首版/内向v2/密度v3共12份记录保留。v3仍未解决底板可观测判定和实际高度泛化，未并入保留的v4正式候选。', '',
        '物理诊断：部分原标注端点在同一有限STEP面边界外约0.185mm；平面距离接近0不等于处于实际面内。底板端点的面积探针不能要求越过板边，亦不能把稀疏点缺失自动认作无材料。FIT GR探针在同一真实CAD底板面内，但PCD近邻有7—10mm间距；密度适配按PCD统计记录参数，板高/焊缝3mm误差门槛未改变。有限接触面、孔/台阶、平移/法向符号、局部底板、内向支撑和稀疏面/线/孔等物理断言检查通过；它们不能替代完整验收。', '',
        '还需完成：几何整体及各类别泛化、有限连通底板身份/边界/孔洞识别，板高V/曲线对象角色，挡板−1状态拆分、板侧与焊接侧/分段原因、子段沿线审核真值、完整批量/异常/空结果接口、离线交付与正式独立测试。合同E5要求补足审核真值后验收；本轮派生表继续标NOT_REVIEWED，原标签和全部子段分母保留。', '',
        '训练核对表：合同/板高原标签与局部基准_训练核对_20260930.csv；观察核对表：合同/板高原标签与局部基准_观察核对_20260930.csv。共享新板高候选头：weld_pipeline/runs/local_height_datum_seed140/heads.joblib；旧v4仍保留。', '',
        '新回执：weld_pipeline/reports/local_height_confirmation_20260930/completion_receipt.json；源码、配置、实际训练数组、每端有限面与局部底板证据、三版原始联合失败记录可复查。25原始PCD及参考、22Excel、22STEP、115/v4权重与旧报告哈希全部保持。本轮138/139/140训练与全部原始联合核查已结束；耗时记录仅诊断。']
    report.write_text('\n'.join(lines) + '\n', encoding='utf8')
    receipt = {'status': 'HEIGHT_DATUM_TRAINED_AND_RAW_SCORED_FULL_CONTRACT_NOT_ACHIEVED',
        'user_definition': definition, 'trials': trials, 'raw_joint_runs': raw, 'tables': table_paths,
        'report': str(report), 'report_sha256': sha256(report), 'prior_receipt_sha256': sha256(prior_path),
        'runtime_source_hashes': {p: sha256(p) for p in paths}, 'retained_v4_sha256': sha256(retained),
        'original_pcd_and_reference_sets_unchanged': len(audit['independent_workpieces']),
        'excel_unchanged': len(audit['excel_headers']), 'step_unchanged': len(cad), 'baseline_115_unchanged': True,
        'previous_report_unchanged': True, 'active_training_for_these_trials': False,
        'formal_test_200_available': False, 'customer_reviewed_new_truth': False,
        'speed_optimization': 'DEFERRED_BY_USER', 'full_contract_pass': False}
    (out / 'completion_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
    print('LOCAL_HEIGHT_CONFIRMATION_RECORDED_FULL_CONTRACT_NOT_ACHIEVED', flush=True)


if __name__ == '__main__':
    main()
