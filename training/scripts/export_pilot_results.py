"""Export real checkpoint predictions and a bounded, point-level pilot report."""
import sys, json, csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from training.labels import sha256

run=ROOT/'training/runs/pilot_v1_seed42'
base=ROOT/'training/data/pilot_v1'
out=ROOT/'training/reports/pilot_v1'
out.mkdir(parents=True,exist_ok=True)
training=json.loads((run/'training_result.json').read_text(encoding='utf-8'))
validation=json.loads((run/'validation_results.json').read_text(encoding='utf-8'))
test=json.loads((run/'internal_test_results.json').read_text(encoding='utf-8'))
split=json.loads((ROOT/'training/configs/pilot_split.frozen.json').read_text(encoding='utf-8'))
assert validation['checkpoint_sha256']==test['checkpoint_sha256']==sha256(run/'best.pt')==training['best_checkpoint_sha256']
overfit=json.loads((ROOT/'training/runs/pilot_v1_overfit_bn01/overfit_result.json').read_text(encoding='utf-8'))
epochs=[json.loads(x) for x in (run/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if x]
assert len({e['epoch'] for e in epochs})==len(epochs)

fig,axs=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
axs[0].plot([e['epoch'] for e in epochs],[e['loss'] for e in epochs],color='black')
axs[0].set(xlabel='Sampled epoch',ylabel='Weighted CE + Dice',title='Training loss')
ev=[e for e in epochs if 'validation' in e]
axs[1].plot([e['epoch'] for e in ev],[e['validation']['macro_f1'] for e in ev],marker='o',color='#24509b')
axs[1].axvline(training['best_epoch'],ls='--',color='gray',label='Selected checkpoint')
axs[1].set(xlabel='Sampled epoch',ylabel='Macro point F1',ylim=(0,1.02),title='Two held-out validation workpieces')
axs[1].legend(fontsize=8)
for ax in axs: ax.grid(alpha=.2)
fig.savefig(out/'training_curve.png',dpi=170);plt.close(fig)

clouds=run/'cloudcompare';clouds.mkdir(exist_ok=True)
for r in test['groups']:
    name=r['name']; p=base/name
    raw=np.load(p/'xyz_world_mm.npy',mmap_mode='r')
    first=np.load(p/'voxel_indices.npy',mmap_mode='r')
    xyz=np.array(raw[first])
    probability=np.load(run/'predictions_internal_test'/f'{name}_voxel_probability.npy')
    assert len(xyz)==len(probability)
    vertex=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1'),('weld_probability','f4')])
    for i,a in enumerate('xyz'): vertex[a]=xyz[:,i]/1000
    for c in ['red','green','blue']: vertex[c]=150
    positive=probability>=.5
    vertex['red'][positive]=230;vertex['green'][positive]=40;vertex['blue'][positive]=40
    vertex['weld_probability']=probability
    path=clouds/f'{name}_predicted_region.ply'
    PlyData([PlyElement.describe(vertex,'vertex')],text=False,
        comments=['Model-predicted weld neighborhood, not reconstructed seam centerlines.',
                  'XYZ uses original AllFaces.pcd numeric coordinates (world meters). Threshold 0.5.']).write(str(path))
    reread=PlyData.read(str(path))
    assert len(reread['vertex'])==len(xyz)
    curves=json.loads((p/'curves_world_mm.json').read_text(encoding='utf-8'))
    points=[]
    for w in curves:
        for s in w['segments']:
            a,b=np.array(s['start_world_mm']),np.array(s['end_world_mm'])
            points.append(a+np.linspace(0,1,int(np.ceil(np.linalg.norm(b-a)/2))+1)[:,None]*(b-a))
    reference=np.concatenate(points)/1000
    v=np.empty(len(reference),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
    for i,a in enumerate('xyz'):v[a]=reference[:,i]
    v['red']=30;v['green']=180;v['blue']=70
    PlyData([PlyElement.describe(v,'vertex')],text=False,
        comments=['Customer-supplied reference curves; NOT model extraction. Original PCD world coordinates.']).write(str(clouds/f'{name}_reference_curves.ply'))
    # Equal XY axes permit a direct outline comparison; reference lines are not predictions.
    fig,axs=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    rng=np.random.default_rng(42)
    ix=rng.choice(len(xyz),min(12000,len(xyz)),replace=False)
    for ax in axs:
        ax.scatter(xyz[ix,0]/1000,xyz[ix,1]/1000,s=.3,c='#bcbcbc')
        ax.set_aspect('equal',adjustable='box');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
    axs[0].scatter(reference[:,0],reference[:,1],s=.3,color='#20934c')
    predix=np.flatnonzero(positive)
    if len(predix)>60000:predix=rng.choice(predix,60000,replace=False)
    axs[1].scatter(xyz[predix,0]/1000,xyz[predix,1]/1000,s=.3,color='#d52d2d')
    axs[0].set_title('Customer reference centerlines')
    axs[1].set_title('Predicted point regions (p >= 0.5)')
    fig.suptitle(f"{name} | point F1 {r['f1']:.3f} | precision {r['precision']:.3f} | recall {r['recall']:.3f}")
    fig.savefig(out/f'{name}_comparison.png',dpi=150);plt.close(fig)

with (out/'point_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['集合','工件','点级Precision','点级Recall','点级F1','点级IoU','TP','FP','FN','TN','预处理后分块推理秒'])
    for subset,results in [('验证',validation),('内部测试',test)]:
        for r in results['groups']:
            writer.writerow([subset,r['name'],*[r[k] for k in ['precision','recall','f1','iou','tp','fp','fn','tn','seconds_model_and_tiling']]])

# Reference-conditioned recall diagnoses missed neighborhoods only. These are
# supplied weld IDs, not model-predicted weld instances or endpoint estimates.
reference_errors=[]
type_counts={}
for subset,results,key in [('验证',validation,'validation'),('内部测试',test,'internal_test')]:
    for r in results['groups']:
        name=r['name'];p=base/name
        instance=np.load(p/'instance.npy',mmap_mode='r')
        inverse=np.load(p/'inverse.npy',mmap_mode='r')
        probability=np.load(run/f'predictions_{key}'/f'{name}_voxel_probability.npy')
        valid=instance>=0
        curves=json.loads((p/'curves_world_mm.json').read_text(encoding='utf-8'))
        total=np.bincount(instance[valid],minlength=len(curves))
        hit=np.bincount(instance[valid],weights=(probability[inverse[valid]]>=.5),minlength=len(curves)).astype(int)
        for i,w in enumerate(curves):
            recall=float(hit[i]/total[i]) if total[i] else None
            reference_errors.append([subset,name,w['id'],w['number'],w['type'],int(total[i]),int(hit[i]),recall])
            if subset=='内部测试':
                c=type_counts.setdefault(w['type'],[0,0])
                c[0]+=int(hit[i]);c[1]+=int(total[i])
with (out/'reference_neighborhood_recall.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['集合','工件','参考焊缝ObjectId','参考焊缝编号','参考焊缝类型','前景点数','被检出前景点数','点召回率_非焊缝有效率'])
    writer.writerows(reference_errors)
type_text='；'.join(f"{k}：{a/b:.2%}（{b}个前景点）" if b else f"{k}：无可统计前景点" for k,(a,b) in type_counts.items())

rows=[]
for title,result in [('验证集（2组）',validation),('内部测试集（5组）',test)]:
    v=result['micro']
    rows.append(f"| {title} | {v['precision']:.2%} | {v['recall']:.2%} | {v['f1']:.2%} | {result['macro_f1']:.2%} |")
per_piece='\n'.join(f"| {r['name']} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.2%} |" for r in test['groups'])
resume=json.loads((run/'resume_receipt.json').read_text(encoding='utf-8'))
total_train_seconds=sum(r['train_seconds'] for r in epochs)
text=f'''# 小样本训练结果

已完成一次真实曲线标签监督的SpUNet训练。该模型输出焊缝附近的点区域，尚未输出完整焊缝编号、起终点和长度；下面的F1是点级分割指标，不能当作焊缝有效提取率或3 mm端点验收结果。

## 数据与设置

- 原始22组中，14组具备可核对曲线，共1292条参考焊缝。剩余8组未用于训练，也未当作负样本。
- 几何筛查后冻结7组训练、2组验证、5组内部测试。FR系列全部留作内部测试；没有随机拆点或把同件裁块放入不同集合。
- 14组曲线经坐标转换后到原始点云均在6 mm内。标签为距曲线≤4 mm的前景、≥6 mm的背景，中间带和已检出的实例冲突点忽略。
- 本机RTX 3060 Laptop 6 GB。SpUNet-v1m1，XYZ输入，2 mm体素，32768点裁块，batch 1、梯度累积4步；每个训练工件每轮抽4块，每轮28块，并非完整扫描训练库。
- 使用类别加权交叉熵与Dice、AdamW、5轮预热及余弦衰减；固定种子42，首次基线未加入增强或面分割输入。
- Windows兼容处理：AMP训练、FP32推理；BatchNorm统计更新系数由0.01调整为0.1；通过Python文件流保存中文路径下的权重。

## 已完成的运行

固定裁块含{overfit['points']}点、{overfit['positive_points']}个前景点。300步训练后损失下降{overfit['loss_drop_fraction']:.2%}，同一裁块推理F1为{overfit['metrics']['f1']:.2%}，通过拟合检查。这不是独立测试成绩。

跨工件训练完成{training['completed_epochs']}轮；验证集选择第{training['best_epoch']}轮权重。纯训练循环累计约{total_train_seconds/60:.1f}分钟，验证、数据准备和检查点读写另计。已在第{resume['resumed_from_epoch']}轮保存并退出，从第{resume['next_epoch']}轮恢复模型、优化器、混合精度缩放器和随机状态，完成后续训练。

训练进程记录的PyTorch已分配显存峰值为{training['peak_allocated_mib']:.0f} MiB；该值不代表显卡全部占用。内部测试在权重冻结后运行一次，没有用于选权重或调整阈值。仅完成种子42的一次基线，不声称多种子稳定性。

## 独立点级结果

| 集合 | Precision（合并计数） | Recall（合并计数） | F1（合并计数） | F1（逐件平均） |
|---|---|---|---|---|
{chr(10).join(rows)}

预测按几何划块覆盖整件输入，再回映全部原始点。测试点的真值不参与裁块、输入特征或预测。阈值固定0.5；ignore点不参与点级指标。

| 内部测试工件 | Precision | Recall | F1 |
|---|---|---|---|
{per_piece}

内部测试按参考焊接类型统计前景点召回率：{type_text}。逐条参考焊缝的邻域漏检统计保存在`reference_neighborhood_recall.csv`，其中编号来自客户真值，不是模型提取出的焊缝清单。

本次划分存在明确的类型覆盖缺口：7组训练工件的816条参考焊缝全部为平焊；42条立焊在验证集，12条立焊在内部测试集，训练集没有立焊。当前结果只能作为平焊占主导的首轮基线。现有立焊仅来自2件工件，无法同时覆盖独立的训练、验证、测试三个集合，后续需要补充含立焊的工件。

![训练曲线](training_curve.png)

## 文件与复跑

- 权重：`training/runs/pilot_v1_seed42/best.pt`；恢复检查点：同目录`last.pt`。
- 完整划分：`training/configs/pilot_split.frozen.json`；坐标、标签和文件哈希：`training/data/pilot_v1/manifest.json`。
- 逐件统计：`training/reports/pilot_v1/point_metrics.csv`；完整日志与验证历史：训练运行目录。
- CloudCompare文件：`training/runs/pilot_v1_seed42/cloudcompare`。`predicted_region.ply`红色为模型预测区域，灰色为其他点；`reference_curves.ply`绿色为客户参考曲线。
- PLY坐标与原始`AllFaces.pcd`一致，可直接叠加。它们与客户另行导出的局部毫米坐标PLY不一定同坐标。
- 比较图为XY投影，仅辅助观察；三维位置以PLY和定量检查为准。

激活环境后，可运行：

```powershell
. .\\training\\env.ps1
python .\\training\\scripts\\train_pilot.py --mode evaluate --run pilot_v1_seed42 --subset validation
```

本次耗时记录从预处理后的体素几何到点概率，不包含原始PCD读取、焊缝实例合并、曲线拟合或CSV导出，不能用于判定10秒验收。正式800/100/200组数据尚未齐备；当前结果仅用于判断基线可训练性和定位泛化问题。

下一步先补足立焊训练工件，按结构族和焊接类型共同设计划分；再依据验证集错误开发面/法向辅助、实例分离和曲线端点重建。内部测试若被用于具体调参，应转为开发数据，后续重新留出未使用样本。

最佳权重SHA256：`{test['checkpoint_sha256']}`。
'''
(out/'小样本训练结果.md').write_text(text,encoding='utf-8')
print('Saved actual prediction clouds, plots, metrics and report:',out)
