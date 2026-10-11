"""Generate a progress comparison strictly from completed run receipts."""
import json
from pathlib import Path
from training.labels import sha256

RUNS = [
    ('typed_pilot_v5_seed93', '基础序列模型'),
    ('typed_pilot_focal_v1_seed94', '稀少端点加权'),
    ('process_planes_v1_seed95', '共享工艺切分面'),
    ('process_planes_aug_v2_seed96', '增加等价换向增强'),
    ('surface_process_v1_seed97', '增加单侧共面特征'),
    ('surface_process_v2_seed97', '增加两侧共面特征'),
    ('local_surface_v1_seed98', '去除全局位置与尺寸输入'),
    ('local_hardneg_v1_seed99', '增加训练误报区域采样'),
    ('expanded_local_v1_seed100', '扩至20件并合并粗细面候选'),
    ('invariant_surface_v1_seed101', '去除候选范围距离输入'),
    ('structured_surface_v1_seed102', '加强漏端点及误切断点监督'),
    ('context_surface_v1_seed103', '增加多尺度面范围特征'),
    ('mixed_surface_v1_seed104', '增加圆弧及圆孔直边联合训练'),
    ('distilled_mixed_surface_v1_seed106', '增加训练标签一致性蒸馏'),
]


def main():
    entries = []
    for name, label in RUNS:
        folder = Path('weld_pipeline/runs')/name
        if not (folder/'complete.json').exists():continue
        record = json.loads((folder/'best_validation.json').read_text(encoding='utf-8'))
        config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
        scores = record['validation']
        entry = {'run':name, 'description':label, 'epoch':record['epoch'],
                 'checkpoint_sha256':sha256(folder/'best.pt'),
                 'valid':sum(s['valid_welds'] for s in scores.values()),
                 'truth':sum(s['truth_count'] for s in scores.values()),
                 'predictions':sum(s['prediction_count'] for s in scores.values()),
                 'macro_f1':record['strict_weld_macro_f1'],
                 'all_pieces_pass':all(s['all_gates_pass'] for s in scores.values()),
                 'per_group':scores, 'configuration':config['arguments']}
        entry['false_or_duplicate'] = entry['predictions']-entry['valid']
        entries.append(entry)
    selected = max(entries, key=lambda e:(e['all_pieces_pass'], e['macro_f1']))
    payload = {'scope':'Completed development experiments, not new blind customer acceptance',
               'model_selection':'unchanged macro weld F1 after per-piece gate status',
               'selected_run':selected['run'], 'experiments':entries}
    dest = Path('weld_pipeline/reports/process_plane_progress.json')
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 工艺切分与共面特征训练记录', '',
        '分段标签已用于实际训练。目前仍未整体验收达标。下表仅列正常完成的训练，以原定焊缝级宏F1选择每轮实验的权重；未把各工件最好的不同轮次拼成一套结果。', '',
        '| 试验 | 所选轮次 | 有效/参考 | 无效或重复预测 |',
        '|---|---:|---:|---:|']
    for e in entries:
        lines.append(f"| {e['description']} | {e['epoch']} | {e['valid']}/{e['truth']} | {e['false_or_duplicate']} |")
    lines += ['', '当前选定模型：`'+selected['run']+'/best.pt`。', '',
        '| 开发验证工件 | 有效/参考 | 预测条数 | 精度及误检门限 |',
        '|---|---:|---:|---|']
    for name, s in selected['per_group'].items():
        lines.append(f"| {name} | {s['valid_welds']}/{s['truth_count']} | {s['prediction_count']} | {'通过' if s['all_gates_pass'] else '未通过'} |")
    lines += ['', '“有效”须满足一对一匹配、类型和段数/段序列正确，且全局及每段端点误差均≤3毫米。通过门限要求每件有效率严格大于95%，无效或重复预测不超过参考条数的5%。某件全部找回但多报焊缝，也算未通过。', '',
        '## 标签如何进入模型', '',
        'SegmentCount与JSON有序曲线提供段数及内部断点监督；Front/Back提供工作范围端点类别；两项Any作为整条焊缝的属性。Any不代替段序列，不为未知内部断点编造切分原因。', '',
        '客户图示及Excel中文表头已确认：IsSplitedForDualRobot表示被侧边立板切分，不能仅据英文名解释为机器人分工。通过间隙、相连檐板、挡板、板高、过孔和最外侧底板边缘的定义见reports/process_semantics/客户图示与标签对应.md；新增距离标签解析已完成，辅助训练头尚未加入。', '',
        '只从8件训练标签统计跨工件重复的X切分位置：10、3585.2、7152毫米，将其作为共享可学习参数。预测时只读点云和同一套权重，不读取该件的Excel、JSON、CAD或参考PLY。这些位置在客户现有坐标系内学习，任意坐标变换后的工艺切分尚未验证。', '',
        '等价换向增强只改变序列排列和截面坐标基，不移动实际点云或切分位置。共面特征查询候选接触面内的点，补充原有全空间近邻距离。两侧版本同时检查底板两侧，但本轮对照未提高整体成绩。局部模型另去掉全局位置与外形尺寸的直接输入，世界坐标仍用于共享切分面。', '',
        '## 验证边界与待办', '',
        '前期使用8件训练，扩样试验使用20件真实工件，混合曲线试验增加TD2-LB8A为21件；均保留原4件开发验证。没有将验证件改为训练件，也没有把裁块算成新工件；早期测试已经参与开发的数据不重新称为盲测。立焊训练只有GR24A1的三段组合，FR146A的单段组合尚缺训练覆盖。圆弧候选和曲线解码已实现，104为首次直线/圆弧联合训练；当前按原4件宏F1选定的102仍不支持圆弧，不能作为完整达标交付。', '',
        '现有35件数据没有90—110条参考焊缝的工件。其他条数或单次计时不能替代规定的三次速度验收。原始输入复跑、导出文件与时间以各运行目录内的inference_receipt.json、independent_score.json和导出核查记录为准。', '',
        '额外13件的预处理已执行。BX2A的稠密采样与LB18E相邻面分离问题已修复；仅用细尺度会使LB24A1漏28条，该版本未训练。现采用粗尺度保留、细尺度补充，data/expanded_surface_v3共1350条参考的表示检查通过。历史FR174Q的2条参考与立板位置存在疑点，暂缓加入训练；其余12件与原8件合成20件。四件验证及两件保留不动。内存映射和误报采样均已实测；表示检查不能算模型准确率。', '',
        '继续处理误检、FR146A漏检及内部断点，再开展保留数据验证。源代码快照、数据哈希、权重、逐轮日志和逐条评分均在各运行目录中；客户原始文件未改写。', '']
    raw_rows=[]
    for folder in sorted((Path('weld_pipeline/runs')/selected['run']).glob('raw_validation_*')):
        if not (folder/'independent_score.json').exists():continue
        score=json.loads((folder/'independent_score.json').read_text(encoding='utf-8'))
        receipt=json.loads((folder/'inference_receipt.json').read_text(encoding='utf-8'))
        raw_rows.append(f"| {folder.name} | {score['valid_welds']}/{score['truth_count']} | {score['prediction_count']} | {score['input_guard']['status']} | {receipt['raw_pcd_to_csv_seconds']:.2f} |")
    if raw_rows:
        lines += ['## 所选模型的原始点云复跑', '',
                  '| 输出目录 | 有效/参考 | 预测条数 | 输入隔离检查 | PCD至CSV/秒 |',
                  '|---|---:|---:|---|---:|',*raw_rows,'',
                  '以上均为开发复核，输出后才读取客户标签计分。计时不满足90—110条、重复三次的样件条件，不作为速度验收。Excel、CSV、PLY的读回结果见输出目录export_qa.json。','']
    Path('weld_pipeline/阶段记录_工艺切分与共面特征.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'selected_run':selected['run'], 'valid':selected['valid'], 'truth':selected['truth'],
                      'false_or_duplicate':selected['false_or_duplicate'], 'all_pieces_pass':selected['all_pieces_pass']}, ensure_ascii=False))


if __name__ == '__main__':main()
