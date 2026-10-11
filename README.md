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

## 全量分组与批量输入准备

新增入口依次为 `tools/freeze_full_dataset_split.py`、`tools/qualify_full_dataset_labels.py`、`tools/prepare_full_training_registry.py`、`tools/prepare_full_training_inputs.py`。同工件版本保留在同一集合；历史已见或曝光不明的数据归训练，保护开发件单独诊断，测试只用于最终评分。未知裁剪衍生关系仍需核查，不能仅凭清单宣称已消除所有泄漏。

标签以PLY和Excel交叉核查几何及有序分段；导出编号变化只有在几何唯一一一对应时恢复。无法解决的版本冲突显式隔离。父端点属性不复制给内部分段，数值哨兵、缺失掩码、导出坐标系和原曲线保留。资格核查可以检查各集合的标签质量，不允许把验证/测试标签用于拟合。

训练登记保留所有合格TRAIN版本及家族均衡权重，只对完全相同的点云共享输入计算。准备流程使用原run162/run26冻结骨干和有限几何模块，保存未经模型筛选的完整片段特征与直线/圆弧载体；压缩缓存保持原值，已核查的历史输入可复用。每件缓存完成验证后，仅清理本入口自有临时中间产物。

这些入口完成的是训练数据接入和特征准备，不代表新模型已经训练、提升或验收。数据、标签、冻结权重、分组清单及缓存仍不随代码公开；运行需要项目数据布局和已核验的权重。

## Experiment history and rollback

`tools/record_training_experiment.py` records an immutable local experiment entry with the source revision, public source revision, configuration/result text snapshots and checkpoint SHA256 fingerprints. Large weights and data remain outside Git. Register source and fixed configuration before a run, then commit the completed or failed result record separately. Rollback must restore the corresponding source, configuration, split and checkpoint whose fingerprint matches the record. Historical runs imported before Git was initialized cannot acquire a fabricated original training commit.

This public repository contains the recording tool. Private dataset manifests, experimental results, consultation transcripts and weight files are retained in the server development repository and are not automatically published by the tool. Milestone tags must distinguish input preparation, completed training, completed evaluation and an accepted model.

## Full dataset supervision safeguards

`tools/audit_full_training_version_compatibility.py` checks every registered training label view against other views of byte-identical workpiece inputs. Parent export IDs and row order do not define geometry identity; ordered child geometry, weld types, gaps and closure remain part of the contract. Exact geometry matches allow known endpoint field and signed frame comparisons. Different geometry or ambiguous duplicate-parent alignment requires an explicit reference policy before combining set losses. All source views are retained; unannotated space does not become physical negative supervision.

`tools/prepare_full_model_use_policy.py` adds separate selection and protected diagnostic lists without changing the frozen train/validation/test assignment. The loader must enforce fit, selection, final evaluation and diagnostic permissions. Historical role names and byte identity checks alone do not certify independent generalization; later exposure and nonidentical input derivative relationships still require evidence. These tools are admission audits and do not train or promote a model.

## Full-shard complete-reference warm-start

`server_full_shard_dataset.py` reads every admitted TRAIN version with complete ordered geometry, four-state partial process labels and signed raw export frames. Input features remain separate from supervision, and held or diagnostic views cannot enter the fitter. `prepare_full_task_supervision.py` streams line and arc owner witnesses from complete frozen finite carriers as inputs arrive; every positive witness is checked by original D3. Missing witnesses stay in the full reference denominator.

`server_full_weld_model.py` and `train_full_weld_warmstart.py` implement a downstream warm-start for free line/arc geometry, weld type, endpoint process states, frame axes and teacher ordered-parent structure. Point features use the original frozen upstream weights. The query head inherits a compatible earlier query experiment; this is not the complete original run446 selector. Unknown numeric fields and frames have explicit masks. Missing exact carriers receive declared weak input-grid assignments and complete free-geometry targets, without being renamed D3-positive or background. Same-input versions are separate expected-loss reference views; their parent structures and fields are not merged.

This warm-start is one training phase. Teacher parent order is TRAIN supervision, not a deployment input. Whole predicted-parent proposal/selection, independent complete-output evaluation, baseline regression and CSV integration remain required before any promotion. The reference classifier has no warm-start loss; shared encoder updates can still change its scores. Epochs and data checks do not count as completed optimization-and-evaluation rounds for consultation. `verify_full_weld_warmstart.py` checks admission, initialization parity, masked unknowns and a real in-memory optimizer update; it does not report extraction accuracy. All execution still requires external frozen weights, qualified labels and the registered input shards.

## Input-only complete parent runtime and anchored geometry

`server_full_prediction_runtime.py` scans finite input profiles and produces carrier and free line/arc hypotheses. `server_predicted_parent_pool.py` assembles ordered parents using genuine frozen run446 link and duplicate priors, with explicit shared-boundary reconstruction, positive gaps and closure. Teacher parent order and reference geometry are excluded from this runtime. Resource limits fail explicitly; rejected hypotheses do not become mandatory predictions. Full dataset resource scaling remains required for larger inputs.

`server_ordered_parent_set_model.py` learns nullable parent utility from ordered predicted geometry. `train_full_predicted_parent_selector.py` uses the exact complete reference matching loss and checks selected output against original D3. It preserves separate same-input reference views, does not interpret unmatched reference costs as physical background, and reports actual optimizer updates separately from reference views processed. A pool with no correctly expressible parent may produce no useful update; it must be repaired rather than presented as successful selector training. Use declared pool and experiment names to preserve every frozen recipe.

`server_legacy_anchored_full_weld_model.py`, `server_runtime_aligned_training_samples.py` and `train_full_weld_anchored.py` provide a separate complete-task warm-start. A frozen legacy query branch preserves its existing scores. Learned free endpoints and circle parameters start with zero residuals around actual finite input geometry. All complete reference targets remain supervised, including weak anchors; incompatible carrier-bracket losses are masked rather than removing children. Assignments use the same declared fixed input positions as the anchored runtime. The compatible query experiment is not the complete final run446 selector.

`verify_legacy_anchored_full_weld.py` checks actual-input initialization, unknown-safe complete supervision and a real parameter update while retaining frozen legacy weights. `audit_full_query_geometry.py` diagnoses assigned-query coverage only after actual input predictions are frozen; assigned queries are offline supervision, not deployment output. These controls are not extraction accuracy or independent evaluation. Training, full output selection, signed frame business semantics, held-out evaluation, baseline regression and final CSV remain separate milestones. Checkpoints, private references, protocols and results are external to this public source snapshot.


## Complete set gradients and exact lazy competition

`server_streaming_parent_set_fit.py` accumulates the same complete selected-set hinge gradient in deterministic chunks, retaining every affected parent and relation. The selector loads one workpiece at a time and can schedule every requested job in every pass. `verify_streaming_parent_set_fit.py` checks full loss, gradients and a real optimizer update against the dense calculation on frozen input tensors; this is a numerical control, not model quality evaluation.

`server_lazy_parent_competition.py` queries the exact neighbour graph defined over all predicted parents, including distance ties. Nonpositive unary members cannot enter runtime greedy selection because pair penalties are nonnegative. Training additionally requests every internal comparator edge, even when its unary is nonpositive. Genuine run446 relation features and priors are retained. `verify_lazy_parent_competition.py` checks full-graph, feature and complete-set score parity against a frozen eager graph without opening reference geometry.

Use a new explicit experiment name for `--competition-mode lazy` or `--resource-mode input_scaled_v1`. Input-scaled proposal bounds include all profile spans and aliases; parent/link budgets still fail explicitly on overflow. These options do not certify that every large workpiece fits available memory or that full-dataset training/evaluation has completed. Frozen weights and qualified input shards remain external dependencies.


## All registered training parent jobs

`tools/train_full_parent_dataset.py` streams every registered TRAIN workpiece and preserves every same-input label version. Each complete predicted pool is frozen before reading its references. Full-reference selection uses family-equal version weights, genuine original relation priors and complete set gradients; it does not convert unmatched reference costs into physical background labels. All registered jobs are required in every pass, with explicit waiting while input shards are prepared.

`server_cached_parent_relation_prior.py` caches sampled input primitives while calling the original relation feature method. `compile_compact_reference_table` samples bounded parent chunks and retains every original D3 cost row followed by one global assignment; local chunk TP counts are never summed to approximate global matching. Dedicated controls compare real frozen features, probabilities, complete cost matrices and scores with the eager implementation.

The controller records immutable per-job model/optimizer checkpoints, input receipts, frozen predictions and local experiment entries, then removes only its own rebuildable workpiece pool. Resume is bound to the source, registry and query checkpoint. Per-job training results belong to their corresponding checkpoint and cannot be reported as the final model's accuracy. Query-head training, frozen-model evaluation, held-out selection, final testing, signed frame semantics and CSV integration remain separate requirements. Input-scaled resource limits still fail explicitly on overflow; this source does not establish that every large workpiece has completed training.


The JSON writer preserves NumPy integer, boolean, scalar and array values while continuing to reject nonfinite values. `tools/recover_full_parent_output_record.py` accepts a verified output encoding failure only: it checks the saved calculation sources and checkpoint, regenerates the same selected geometry, completes original D3 evaluation and commits a durable resume pointer without repeating optimizer steps. Other failures require a separate diagnosis; a recovered output is not a new model-quality round.


## Complete held inputs and evaluation roles

`prepare_full_evaluation_registry.py` registers every qualified held label version while deduplicating only input computation. Selection, protected diagnostics and final testing retain separate roles; they cannot enter the training registry. `server_full_evaluation_dataset.py` supplies the same complete input tensor interface, without a training epoch or supervision API. Reference access requires an authenticated input-only prediction freeze. Final-test reference access also requires the declared validation-selected query and selector checkpoints. This workflow guard does not certify historical independence.

`prepare_full_evaluation_inputs.py` prepares raw held XYZ in a separate owned workspace. Its frozen point-feature stages wait for the complete TRAIN input producer to finish, then reuse the original input computations without fitting. Some preserved helper receipts have literal TRAIN routing tags; the held registration and shard receipts remain authoritative and prohibit fitting. Full fragment features and every label version are retained. `verify_full_evaluation_registry.py` and `verify_full_evaluation_xyz.py` check actual role admission and raw-input bindings; these checks are not model accuracy or complete-runtime validation.

`audit_full_workpiece_xyz_identity.py` compares exact point multisets across every qualified version, preserving duplicate-point multiplicity and ignoring PLY metadata and vertex order. It does not normalize pose, prove the absence of rotated/cropped derivatives, or certify the absence of historical development exposure. Input preparation, identity audits, training, held prediction scoring and CSV acceptance remain separate evidence.


## Complete prediction and genuine native input replay

`server_parent_prediction_runtime.py` authenticates the complete input pool and bound selector checkpoint, then computes every ordered-parent score and nullable whole-parent selection without a reference API. `verify_full_parent_prediction_runtime.py` compares actual input tokens, fitted scores and complete selections with the preserved consumer and saved control. Numerical parity does not certify extraction quality.

`evaluate_full_weld_predictions.py` runs one complete held role at a time. Predictions are frozen before each original label view is read by the D3 scorer. The default requires actual complete TRAIN visit sets and fitting receipts for both the query model and its bound selector, including the final durable selector checkpoint. Partial controls cannot authorize final testing. Final testing requires prior complete validation evidence and an immutable validation-selected model pair. The D3 geometry report does not certify process attributes, signed frames, CSV acceptance or historical independence.

`preserve_full_input_surface_context.py` observes completed input stages and losslessly retains genuine surface normals and input requests in a separate cache before temporary cleanup. It does not modify live preparation or frozen shards. Missing context is reported explicitly. `server_native_shard_input.py` uses these authenticated artifacts and complete original feature arrays to load the original native object, interval and connection models. `verify_native_shard_input.py` compares actual finite profiles, interval predictions, source frames and connection features with a retained original input. This does not imply that the final native parent selector or the incremental validity branch has been replayed.


## Complete native selection and learnable reference scoring

`server_native_parent_runtime.py` reuses the original object/interval/frame models, all legal connection features, sparse parent assembly and the frozen native precision selector. Every connection is streamed into an owned cache, and capacity failures are explicit. No reference geometry is read before complete predictions are frozen. `verify_complete_native_parent_runtime.py` and `verify_all_retained_native_parents.py` compare actual complete candidates, unary features, metadata, selection indices, probabilities and outputs with retained native artifacts. This native branch does not execute the separate incremental direction-free validity classifier; parity is not independent accuracy. Original weights, preserved artifacts and authenticated input context remain external dependencies.

`audit_full_reference_query_gate.py` compares the frozen reference prior at actual complete TRAIN assignments with a diagnostic logit-zero boundary using the declared first-epoch seeds. The expanded complete runtime has no reference-score admission threshold for fixed query positions: it retains all declared positions in every nonempty finite profile. Scores rank grid positions and enter sequence features and the length-weighted relation prior. Assignment scores below zero do not prove that a candidate was filtered or that complete reference geometry is unreachable. Other queries can also match a reference. Held and protected labels are excluded.

`server_reference_recall_weld_model.py` adds a zero-initialized trainable reference-score correction to the anchored model. The frozen original prior and correction input are explicitly detached from the geometry branch for this loss. Only known annotated-child assignments provide positive responsibility targets; a squared correction prior regularizes the change. Unassigned queries are not labeled physical absence. `verify_reference_recall_weld_model.py` compares actual initialization outputs and checks head-only gradients and parameter updates. Those controls do not establish geometry equivalence during combined fitting or improved final selections.

`train_full_weld_reference_recall.py` consumes every registered complete TRAIN view with the same family weighting, missing-field masks, line and arc geometry, original frame targets, closure and adjacency supervision. It preserves the controlled anchored initialization and records inherited and new optimizer steps separately. This source enables streaming training, not a validated new deployment model. Complete-query coverage, architecture-aware inference, fitting a selector bound to the new query checkpoint, validation selection, protected diagnostics, baseline regression, final testing, signed frame semantics, CSV and raw-XYZ runtime remain separate requirements. Private checkpoints, training protocols, references and results are not included.

## Bound query architecture and complete parent continuation

`server_query_checkpoint_model.py` strictly loads the anchored and reference-score query architectures. Completion verification uses authenticated fitting protocols, every registered TRAIN visit in every epoch, version-specific label evidence and actual optimizer history. A completion flag alone is insufficient. `verify_query_checkpoint_models.py` checks real query outputs against direct architecture loading and rejects malformed architectures and forged partial-fit completion claims; it does not evaluate complete extraction accuracy.

`prepare_full_recall_parent_pool.py` binds one immutable query snapshot and retains the complete finite input profile and candidate pool. `train_full_recall_parent_dataset.py` fits every registered input and separate same-input label view using complete expected D3 reference utility. It requires genuine complete query fitting by default; explicitly partial controls remain separate. The original best relation priors and an optional selector initialization are external dependencies, not substitutes for a fitted, query-bound final selector.

`continue_full_recall_training.py` waits for authentic complete query fitting, freezes its checkpoint, then runs matching all-parent fitting and separate model-selection and protected-diagnostic roles. It does not automatically authorize final-test model selection. `control_full_recall_parent_fit.py` consumes one entire actual TRAIN control pool, keeps every candidate and label view, and exercises the same full-set loss. Its process binding prevents restarting a still-running producer or mistaking a reused PID for that producer. Control results are neither complete dataset fitting nor independent model quality.

## Scoped input read-cache recovery

`reclaim_completed_training_read_cache.py` advises the kernel to discard clean read-cache pages for completed registered TRAIN point clouds and immutable input shards. It uses read-only file descriptors, checks unchanged file bindings and allows active inputs to be excluded. It does not delete data, reset models, lower memory guards or change global cache settings. `watch_completed_training_read_cache.py` applies the same operation when conservative cgroup headroom falls below its declared threshold, records each operation and stops when all configured completion receipts exist. File advice is advisory; observed resource recovery is separate from fitting progress and model quality.
