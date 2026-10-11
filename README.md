# 点云焊缝提取

从真实 XYZ 点云识别焊缝、分段、父子关系及输出属性的研发源码。项目仍在训练与验证，尚未达到完整验收要求。

数据集仓库：https://github.com/G1ow9711/Dataset （用户仍在补充上传）。本仓库不含数据、权重、生成缓存、浏览器登录信息或服务器交接历史。

## 当前状态

历史 run446 在冻结上游输入上复现 259/269（96.28%），另有 5 个误报。这不是独立测试集成绩，也不是完整原始 XYZ 输入的端到端成绩。原最佳权重保留，未被实验模型覆盖。

后续实验包括端点事件选择、联合区间评分、父结构连接，以及 STEP 提供离线有限面和材料关系监督。实验模块不得仅凭训练拟合率作为交付模型。模型质量和属性优先，真实点云读取至全部输出不超过 10 秒仍是最终要求，目前未验证达成。

## 源码导航

- `weld_pipeline/`：历史管线、几何、评分与实验模型。
- `training/`：训练组件和第三方骨干源码，保留上游许可证与来源记录。
- `tools/`：服务器数据准备、训练、评估和诊断脚本。
- `SOURCE_MANIFEST.json`：本次源码快照文件与 SHA256。

当前联合区间实验入口：`tools/train_stable_joint_interval_residual.py`、`tools/infer_stable_joint_interval_full_output.py`。候选同分排序固定，零残差在已检查开发工件上恢复原输出。实验分数不代表验收达成。

当前离线表面监督入口：`tools/prepare_surface_boundary_samples.py`、`tools/label_surface_boundary_samples.py`、`tools/prepare_exact_step_surface_teacher_v3.py`、`tools/train_boundary_surface_material_relation.py`。STEP 只用于离线标签，运行特征来自 XYZ。未知材料归属保持未知；不同材料不等于无焊缝。

## 环境与复现边界

服务器实验使用 Python、PyTorch 2.8/CUDA 12.8、NumPy 1.26.4、scikit-learn 1.5.2，并依赖 SciPy、几何处理库及部分任务的 Gmsh。`training/requirements-lock.txt` 是历史 Windows 环境锁，不可直接当作 Linux 服务器安装清单。

多数历史脚本依赖特定 `server_rebuild/` 缓存、冻结上游模型及数据布局；当前源码快照不能仅靠克隆仓库复现全部训练。数据仓库也未完整上传。运行前需核对脚本入口、所需权重和数据清单，缺失文件不能用虚构数据替代。

第三方许可证位于 `training/third_party/`。本仓库没有额外授予项目自有代码的开源许可证。

## 业务与评价约束

先识别水平底板系统，再找立板及有限立面；立面相交形成立焊缝，立面与底板相交形成平焊缝。多底板可以不同厚度、不同高度，不能强制共享一个全局平面。底板拼缝不自动作为本任务焊缝。

挑檐间距恰好等于 35 mm 也需要分段；该规则不是短焊缝长度阈值或通用几何合并阈值。孔、外边界和无限平面交线不能自动当作焊缝。同工件旋转、裁剪版本必须属于同一数据分组。

咨询或复核时请同时提供提交 SHA、相关代码路径、具体输入依赖和评估口径，避免混淆条件拟合、冻结输入复现与完整端到端提取。

## 实验性网格端点分支

`server_grid_feature_stream.py` 从有限载体的输入片段生成分块网格特征，`server_grid_endpoint_selector.py` 显式区分合成网格来源，并维护有界端点候选。网格不是观测接触，也不自动成为焊缝。`server_grid_candidate_runtime.py` 是共享候选评分入口。

训练入口为 `tools/prepare_grid_endpoint_training.py`、`tools/train_grid_endpoint_head.py`；联合损失保持同一参考子段两端的绑定，不将多个参考解释的端点任意交叉。准备脚本只采用已登记 TRAIN 范围，缺少可表达正例保持未知。运行仍依赖本地冻结输入和权重，仓库未附这些产物。

`tools/evaluate_grid_endpoint_comparison.py` 固定旧检测器的查询与类型，比较网格端点头训练前后，并在冻结预测后评分。这仅隔离端点变化：仍使用单子段输出，尚未实现新的可空整父选择、完整属性及坐标系导出。此分支未替换原模型，亦未证明原始 XYZ 全流程十秒达标。

## 实验性整父集合选择

`server_whole_parent_features.py` 将原生多子段父焊缝和端点分支的实际输出映射到相同的真实载体特征，保留曲线、子段顺序及间隙。数值上塌缩到一点的极短子段仅使用其真实包含片段的积分点极限，不修改输出几何；该数值处理不是业务分段阈值。

`server_offline_parent_set_reference.py` 仅用于训练及评价，保留原 D3 的完整匹配代价，不能将逐候选正确数之和当作整组正确数。`server_reference_parent_selection.py` 在输入候选上执行允许空集的选择，通过软竞争代价和完全相同子段的使用约束减少重复输出。近邻图不是物理同焊缝判据，贪心选择不保证全局最优。两父代价头的登记范围为两倍较小子段数，超出此容量的训练对会显式失败；这不是 D3 的普遍数学上界。

主要入口为 `tools/prepare_reference_parent_selection.py`、`tools/train_reference_parent_selection.py`、`tools/train_structured_parent_residual.py` 和 `tools/evaluate_reference_parent_selection.py`。结构残差零初始化时恢复原候选评分和选择，再使用完整参考集合的 TP−FP 改进目标训练。参考中未匹配的代价只表示当前标注评价约定，不能解释为物理上没有焊缝。

这些实验改善了部分训练拟合，但开发样本仍有严重漏检和误报，尚未替换原最佳模型。当前完整父集合处理的是已冻结上游候选；新增多段与曲线候选覆盖、属性、坐标系及最终 CSV 仍需继续完成。它不是完整 XYZ 管线的独立验收或时延结果。

局部峰值网格对照入口为 `tools/evaluate_grid_peak_comparison.py`；固定候选预算下的覆盖变化不能替代完整输出质量评价。
