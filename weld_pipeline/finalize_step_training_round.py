"""Reproducible end-of-round evidence, never a claim of full acceptance."""
import csv
import json
import shutil
from pathlib import Path

from training.labels import sha256


def read(path):
    return json.loads(Path(path).read_text('utf8'))


def main():
    run=Path('weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4')
    names=['224-GR24A1','527-FR146A','TD2-LB8A','232-TT1A','224-FR71A'];raw=[]
    for name in names:
        folder=run/('raw_'+name);r=read(folder/'raw_verification.json')
        assert r['guarded_pcd_read_count']>0 and r['reference_cache_open_guard_active']
        assert r['maximum_coordinate_difference_mm']<=1e-6
        assert read(folder/'export_qa.json')['csv_excel_ply_all_coordinates_and_ids']=='PASS'
        assert read(folder/'child_attribute_export_qa.json')['status'].startswith('PASS_')
        raw.append({'name':name,'truth':r['score']['truth_count'],'valid':r['score']['valid_children'],
            'false':r['score']['false_predictions'],'pcd_reads':r['guarded_pcd_read_count'],
            'maximum_coordinate_difference_mm':r['maximum_coordinate_difference_mm']})
    old=read('weld_pipeline/reports/contract_child_20260930/audit.json');hashes=[]
    for piece in old['independent_workpieces']:
        assert sha256(piece['pcd'])==piece['pcd_sha256']
        for path,wanted in piece['reference_hashes'].items():
            assert sha256(path)==wanted
        hashes.append(piece['name'])
    for row in old['excel_headers']:
        assert sha256(row['path'])==row['sha256']
    cad=read('weld_pipeline/reports/step_geometry_20260930/summary.json')
    for row in cad:
        detail=read(Path('weld_pipeline/reports/step_geometry_20260930')/(row['name']+'.json'))
        assert sha256(detail['step'])==row['step_sha256']
    baseline=Path('weld_pipeline/runs/typed_branch_surface_v3_seed115/best.pt')
    assert sha256(baseline)==old['source_checkpoint_sha256']
    rounds={}
    for seed,folder in [(128,'contract_full_fit_horizontal_seed128'),(131,'invariant_horizontal_seed131'),
                        (133,'invariant_horizontal_direct_seed133'),(135,'oriented_context_horizontal_seed135')]:
        receipt=read(Path('weld_pipeline/runs')/folder/'complete.json');assert receipt['epochs']==120
        rows=[json.loads(s) for s in (Path('weld_pipeline/runs')/folder/'epochs.jsonl').read_text('utf8').splitlines()]
        assert len(rows)==120 and [r['epoch'] for r in rows]==list(range(1,121))
        rounds[str(seed)]={k:receipt[k] for k in ['epochs','selected_epoch','elapsed_seconds','fit_summary','unseen_check_summary']}
    supplemental_dependencies=read('weld_pipeline/runs/oriented_context_horizontal_seed135/additional_runtime_dependencies.json')
    for path,wanted in supplemental_dependencies.items():
        assert sha256(path)==wanted
        shutil.copy2(path,Path('weld_pipeline/runs/oriented_context_horizontal_seed135/source_snapshot')/Path(path).name)
    attrs132=read('weld_pipeline/runs/expanded_plate_attributes_seed132/complete.json')
    attrs134=read('weld_pipeline/runs/structured_plate_attributes_seed134/complete.json')
    joint=read('weld_pipeline/reports/joint_structured_attributes_20260930/complete.json')['roles']
    main_fit=[r for r in joint['FIT']['pieces'] if not r['name'].startswith(('EDH2P','N8C','TD2'))]
    g=sum(r['truth_count'] for r in main_fit)
    data_fit={'truth':g,'geometry':sum(r['geometry_valid'] for r in main_fit),
              **{k:sum(r['joint_candidate_counts'][k] for r in main_fit) for k in ['thickness_1','thickness_2','normal_1','normal_2']}}
    geometry=read('weld_pipeline/runs/contract_full_fit_hybrid_v1/training_receipt.json')['roles']
    summary={'status':'TRAINED_AND_REFERENCE_AUDITED_FULL_CONTRACT_NOT_ACHIEVED',
        'checkpoint':str(run/'best.pt'),'checkpoint_sha256':sha256(run/'best.pt'),'single_checkpoint':True,
        'geometry_roles':geometry,'dataSet_fit_joint_candidate_counts':data_fit,'joint_candidate_roles':joint,
        'horizontal_trials':rounds,'attribute_trial_132':attrs132,'attribute_trial_134':attrs134,
        'raw_exports':raw,'raw_guard_method':'builtins.open + io.open + explicit pathlib.Path.open; negative reference-read control.',
        'original_pcd_and_reference_sets_unchanged':len(hashes),'original_excel_unchanged':len(old['excel_headers']),
        'original_step_unchanged':len(cad),'baseline_115_unchanged':True,'formal_test_200_available':False,
        'reviewed_customer_attribute_truth':False,'speed_acceptance':'DEFERRED_BY_USER','full_contract_pass':False}
    out=Path('weld_pipeline/reports/step_training_round_20260930');out.mkdir(parents=True,exist_ok=True)
    (out/'completion_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    (run/'raw_batch_verification.json').write_text(json.dumps({'status':'PASS_FIVE_RAW_REBUILDS_AND_ALL_EXPORT_CHECKS',
        'pieces':raw,'full_contract_pass':False},indent=2),encoding='utf8')
    receipt=read(run/'training_receipt.json');receipt.update(raw_verified=True,raw_verification_file=str(run/'raw_batch_verification.json'))
    (run/'training_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['STEP核查与模型训练进展 — 2026年9月30日','',
        '完整合同状态：NOT_ACHIEVED。已继续训练并核查指定DataSet内全部22份STEP；精度门槛保持，耗时优化后置。以下成绩区分拟合、未训练检查和未审核参考，不作为正式验收。','',
        '已确认的执行条件：CAD采样AllFaces纯XYZ为运行输入；GW1是顶部面。原始件标全，未标接触线作为负例，部分增强未知区继续屏蔽。按有序真实子段计数，端点及整线3mm、整体和适用类别严格>95%、FP/G≤5%；板厚两侧≤1mm、带方向单位法向两侧≤3°，其余距离/高度/边距≤3mm，各字段含几何漏检分别>95%。方向翻转和板侧交换按对应关系判错。10秒验收暂缓，当前不记为通过。','',
        '数据使用：指定DataSet共22件，19件进入本版FIT、2件（232-TT1A、224-FR71A）保持未训练检查、BM79A缺完整曲线几何而未编造真值。几何FIT另含此前6件补充数据，共25真实件1605子段；V头另用16增强条目，不能把增强当独立件。原4件DEV已纳入本版监督，后续只算FIT；原21/4划分及115基线保留。两件检查数据没有参与梯度或本轮检查点选择，但已在开发中查看，不等于甲方封存200件正式盲测。','',
        'STEP核查结果：22份、459实体，458实体的平行大面间距与体积惯性厚度相互支持。Excel/几何JSON无现成板厚和带方向法向审核标签，但STEP可推导训练参考。已完成真实表面网格与PCD配准，222-TT1A、232-TT1A初始PCA配准失败已用板面方向及8种有手性姿态修复；失败记录保留。法向用实体内外测试校正材料外向，不能直接沿用CAD参数法向。','',
        '已为1462个直线子段建立五点参考，其中1460个具有沿线共同、唯一的两块板与外向面身份。候选不是客户审核真值；板1/板2暂定为cross(n1,n2)与有向焊缝切线同向的右手顺序。反向几何匹配时反转沿线点序、交换板侧，材料外向正负不翻转。曲线和不明确接触保留未知。','',
        '“GR同长立焊三段、FR一段”的具体含义：GR有42条立焊（16条120mm、8条70mm、18条85mm），全部保留三子段；120mm示例为40.2+40.2+39.6mm。FR有12条120mm立焊，均单段。GR三段中点仍接触同一对板件，切分不由换板造成。原父线侧板分割Any分别true/false；长度不能决定切分，也不能据此制定“所有120mm都三段”的规则。该差异已进入共享监督和工艺头，不要求用户代替模型诊断。','',
        '追加训练与检查：','',
        '| 试验 | FIT与实际工作 | 未训练/原DEV结果 | 采用情况 |','|---|---|---|---|',
        '| 124/125 | 候选树、等价姿态结构监督 | 原DEV立焊均0/12 | 未采用 |',
        '| 126 | 完整120轮，聚焦立焊增强 | 原DEV378/420，FP22，V0/12 | 未提高 |',
        '| 127 | 纯PCD板面上下文三头 | 原DEV378/420，FP31，V0/12 | 未采用 |',
        '| 128/129 | 横焊120轮及共享立焊/工艺头；FIT1570/1605，FP50 | 97/128，FP29 | 保留为开发几何内核，未独立达标 |']
    for seed in ['131','133','135']:
        r=rounds[seed];f,c=r['fit_summary'],r['unseen_check_summary']
        lines.append(f"| {seed} | 横焊120轮，FIT {f['valid_children']}/{f['truth_count']}，FP{f['false_predictions']} | {c['valid_children']}/{c['truth_count']}，FP{c['false_predictions']} | 退化，未替换128 |")
    lines+=['','130初版属性头只用GR/FR186参考子段；132扩展为19件1332子段、34920等价姿态样本。132在真值五点上板厚拟合1328/1332，但真实预测几何下板厚仅1147/1605、1131/1605。134据FIT诊断去掉板厚回归的端点位置特征，加入纯PCD板面支撑/平行面结构，方向分类头冻结；同一256树配置，未训练参考板厚提高到122/128、121/128，法向各126/128。此处参考条件成绩尚未计几何漏检。','',
        '保留v4模型，实际预测几何与属性一起核算的保守诊断如下；全部要求字段仍未实现，因此表中局部超过95%不能称合同通过。','',
        '| 数据角色 | 真值子段 | 几何核心有效 | 板厚1 | 板厚2 | 法向1 | 法向2 |','|---|---:|---:|---:|---:|---:|---:|',
        f"| 指定DataSet的19件FIT | {g} | {data_fit['geometry']} | {data_fit['thickness_1']} | {data_fit['thickness_2']} | {data_fit['normal_1']} | {data_fit['normal_2']} |",
        f"| 全25件FIT，含旧补充 | 1605 | 1570 | 1295 | 1295 | 1299 | 1299 |",
        '| 两件未训练检查 | 128 | 97 | 93 | 92 | 96 | 96 |','',
        '两件检查的几何核心有效率75.78%，FP/G=29/128=22.66%，均失败；真实立焊与曲线在这两件没有正真值，不能宣布这两类已独立通过。六件旧补充尚无本轮板面参考映射，271子段不能计作属性参考成功。板厚、法向采用STEP派生候选比较，正式E2还缺审核与冻结侧序。','',
        '运行核查：五份原始PCD重新读文件，显式拦截builtins.open/io.open/Path.open，禁止STEP、Excel、几何真值及训练/参考缓存；反例控制证明Path读取被阻止。早期脚本只拦builtins/io，Python3.10的Path缓存accessor未被覆盖，该旧guard标识不作完整隔离证明。以下五件使用修正后的拦截重跑，几何与缓存最大坐标差均≤1e-6mm，CSV/Excel/PLY和子段坐标、长度、单位法向、未知状态、沿线五点引用全部检查通过。','',
        '| 原始PCD | 有效/真值 | FP |','|---|---:|---:|']
    lines += [f"| {r['name']} | {r['valid']}/{r['truth']} | {r['false']} |" for r in raw]
    lines += ['',
        '未完成项：未训练几何泛化；焊接侧/分段原因与适用性冻结；挡板、板高、过孔、整体底板外边距、檐子等其余字段的完整估计及逐字段验收；独立立焊/曲线样本、BM完整曲线真值及正式200件测试。旧挡板−1同时代表无对象和超50mm，不能当−1mm回归，也不能自动转成其中一种状态。当前unknown和未审核参考均不算正式通过。','',
        '保留开发候选：E:/点云数据提取焊缝/weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt。一个检查点包含几何和属性权重，运行只读PCD。原115未替换。输出板厚与材料外向法向的候选值、五点沿线引用和来源；其他未实现字段明确unknown。CSV置信度是方向分类五点最小置信度，不能解释为已校准板厚误差概率。','',
        '调用：在项目目录使用 .venv/Scripts/python.exe -m weld_pipeline.predict_structured_candidate_model --pcd <AllFaces.pcd路径> --checkpoint weld_pipeline/runs/contract_full_fit_hybrid_attributes_v4/best.pt --outdir <输出目录>。','',
        '证据：weld_pipeline/reports/step_training_round_20260930/completion_receipt.json；step_geometry_20260930；all_step_references_20260930；step_reference_pose_repair_v1；step_contact_continuity_candidates_v1；joint_structured_attributes_20260930；v4/raw_*。候选审核表：合同/STEP派生板厚与法向_候选审核样表_20260930.csv。分段图：weld_pipeline/reports/step_alignment_20260930/同长立焊_原标注分段对照.png。','',
        f"数据保护复核：25份原始PCD及各自参考、22份Excel、22份STEP与115权重哈希不变。候选模型SHA256：{sha256(run/'best.pt')}。135已完成120轮，当前无本轮训练进程。耗时仅记录诊断，本轮没有进行10秒性能验收。"]
    report=Path('合同/STEP核查与模型训练进展_20260930.md');report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    previous=Path('合同/项目确认与新指标训练进展_20260930.md')
    body=previous.read_text('utf8')
    marker='追加训练/STEP核查更新：'
    if marker not in body:
        body=marker+'完整最新记录见[STEP核查与模型训练进展_20260930.md](STEP核查与模型训练进展_20260930.md)。以下为121—123初轮记录；STEP派生参考已继续训练，正式指标仍未达到。\n\n'+body
        body=body.replace('| STEP | 仅供人工查看，运行时禁止读取 |','| STEP | 离线参考核查、推导训练候选；运行时禁止读取 |')
        previous.write_text(body,encoding='utf8')
    print('ROUND_EVIDENCE_READY',str(report),summary['status'],flush=True)


if __name__=='__main__':
    main()
