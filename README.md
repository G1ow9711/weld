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


## Frozen-output process diagnostics

`server_complete_output_assessment.py` reports complete-parent geometry separately from the original child-level D3 score, preserving the same global parent assignment. It measures required known field states and exact numerical errors without treating censored states or unknown labels as measured distances. Raw ordered correspondence and signed endpoint-axis errors are separate diagnostics; whole-parent reversal does not certify business direction, welding side or frame semantics. Field errors remain conditional on geometry matches, with unmatched known labels reported explicitly. Attribute tolerances, split reasons, CSV and raw-input timing need separate acceptance evidence.

`assess_full_weld_output_fields.py` authenticates frozen prediction, model, registry and prior evaluation bindings before loading references. Existing TRAIN copies append one semantic audit field; reconstructing their original canonical bytes is allowed only when the original frozen SHA matches exactly. Held receipts require byte equality. Held assessment requires the complete registered role and a fully fitted model pair; final testing also requires prior immutable validation selection. Different post-update TRAIN checkpoints are never aggregated as one final model score. The tool does not fit or edit predictions. `verify_complete_output_assessment.py` checks partial-parent, sentinel, unknown-label, missing-output, signed-axis and receipt-binding cases using explicitly synthetic fixtures. These checks do not establish extraction accuracy or complete contract acceptance.


The full-training continuation dispatches the same selection and protected-diagnostic role names accepted by the held reader and evaluation CLI. After each complete registered role has frozen geometry scores, it runs the authenticated process-field and signed raw-axis assessment and verifies every registered version and saved result binding. These diagnostic measurements do not certify business frame semantics or field tolerances. Final testing remains excluded from automatic model selection. Updating a running source requires preserving and replacing its idle bound controller under a new protocol; fitting and input producers retain their own source bindings.


`supervise_full_training_inputs.py` keeps the original all-registered input producer and resource thresholds unchanged. It retries only the exact observed low-headroom guard exit of an incomplete finite-support or learned-surface stage, while refusing completed stages, unrelated latest errors or duplicate live preparers. The failed stage is copied, byte verified and renamed within its owned temporary filesystem before a retry. A declared conservative headroom policy delays restarting the producer. Other failures stop with evidence instead of skipping an input. These checks and retries do not establish model quality.


## Exact offline reference compilation

`server_exact_endpoint_reference.py` retains every prediction row, reference column and original D3 assignment cost. Impossible kind/type or structural pairs retain unavailable costs and their full prediction counts without materializing sampled geometry. The line endpoint broad phase permits any child pairing and either orientation; the original scorer still enforces whole-parent order, gaps and direction. Matching arcs and closed contours use the original geometry rules, including arbitrary circle anchors. One full global assignment remains unchanged.

The exact training backend requires current synthetic semantic controls plus authenticated actual whole-pool and frozen-output comparisons of every cost, true-positive count and global assignment. Separate loader controls compare all tokens, relation inputs and complete label views. Independent entry points reuse the original complete TRAIN job loader, objective, optimizer and prediction logic through private function namespaces. They bind comparison artifacts and source files, preserve registered TRAIN coverage and keep selection, protected diagnostics and final testing in their existing roles. This optimizes label-side training computation; it does not establish extraction accuracy, inference timing or model promotion.

## Finite mesh rank support and failure diagnosis

`sjg_union_sampling_mesh_995.local_support` applies the existing measured two dimensional support requirement before triangulation: fewer than three supported vertices cannot form an eligible triangle. Unsupported points remain unresolved in the original extractor. This does not discard a whole workpiece, add synthetic coordinates, change distance rules or alter a resource guard.

`tools/diagnose_finite_mesh_vertex_index.py` authenticates cached input features and captures the original failing mesh in a private local artifact without reading labels or STEP geometry. `tools/verify_finite_mesh_rank_support.py` reproduces the frozen original failure, checks the unsupported mesh and compares every preceding actual mesh result with its original fingerprint. These checks certify the numerical fix and existing boundary controls; they do not certify model quality or completion of full dataset training. Private geometry, receipts and experiment results are not included in this repository.

## Compact endpoint reference compilation

`server_compact_endpoint_reference.py` checks the original float64 line endpoints before materializing label-side samples. It retains every candidate row, original false-positive count, reference column and original global D3 assignment. Arcs, closed parents and possibly matching line pairs use the unchanged scorer. Whole-parent order, legal reversal, partial-child matches and the original endpoint boundary remain part of the scoring contract. The original runtime sampler determines endpoints even when stored points are present.

`verify_compact_endpoint_reference.py` reuses all original semantic controls and adds compact sampler checks. `verify_actual_compact_endpoint_reference.py` authenticates complete TRAIN pools or saved TRAIN outputs and compares every original cost and matching result. `verify_compact_reference_training_backend.py` compares the entire original job loader, tokens, relations and label views, including rejection of a forged numerical proof.

The independent compact training backend requires current synthetic, complete-pool and saved-output proofs before changing only the offline reference compiler. Its control, full-dataset trainer and continuation preserve original models, complete-set objectives, optimizers, inference candidates, role separation and field evaluation. This reduces computation for unavailable pairs; it does not prove faster full training, model quality or formal business-frame acceptance. Private comparison results, geometry, datasets and weights are excluded from this source repository.


## Validation-fixed final TEST continuation

`continue_validation_fixed_test.py` predeclares family-weighted complete D3 utility, recall and false-positive tie breakers before TEST assessment. It authenticates every complete validation view, training receipts, model hashes and field assessment before fixing the model pair. Protected diagnostics and TEST scores cannot select the model. Known process-field errors are reported without inventing unfrozen business-frame tolerances.

`verify_validation_fixed_selection.py` checks held-role rejection, missing and duplicated views, complete count identities, family weighting and deterministic ordering. It also passes an actual saved TRAIN assessment to the existing final-selection verifier and requires rejection. The final continuation requires this current admission proof and binds the registry and source hashes while waiting. After complete training and validation, it runs all registered TEST geometry and field assessments; it does not automatically promote the model or certify independent generalization. Private evidence, weights and datasets remain external dependencies.


## Bounded original relation reuse

`server_bounded_cached_relation_prior.py` limits reusable decoded parent geometry. `server_bounded_lazy_parent_competition.py` limits reusable relation features while still returning every requested edge. Cache eviction recomputes original geometry and probabilities; it does not remove candidates, change full-input neighbour radii or ties, or alter complete-set objectives.

The bounded training backend requires authenticated comparisons of all frozen original pair features and penalties, complete graph edges, replay after eviction, and actual complete-job optimizer parity. Its numerical controls compare input tokens, every reference view and matching cost, eight real optimizer histories, final model and optimizer tensors, scores and selections. The original model, objective and optimizer remain unchanged. Separate entry points bind proofs and sources before applying bounded reuse to actual controls, all-TRAIN parent fitting and complete held-role continuation. These controls establish numerical compatibility, not extraction accuracy or complete process memory limits.

## Resuming point-feature resource failures

`server_completed_point_features.py` authenticates completed point-feature components against original weights, model and inference sources, input blocks, every feature artifact and exact once-only core coverage before reuse. Incomplete components are not treated as complete. The input supervisor recognizes only the actual terminal low-headroom guard error in supported geometry or point-feature stages. It archives and byte-verifies the single incomplete component before a same-filesystem rename and retry with the original inference algorithm. Completed components, inputs, model updates and original resource thresholds are retained. Other errors stop instead of skipping a case.

The completed TRAIN read-cache watcher derives active parent exclusions from its explicit completion-receipt configuration, so a new final training experiment cannot leave an obsolete parent-directory exclusion or completion gate. Read-only file advice preserves data bytes and does not change global cache settings.


## Reusing reciprocal primitive distances

`server_reciprocal_cached_relation_prior.py` executes both original directed relation methods and preserves their original float64 arithmetic. Identical analytic primitive distances are reused only inside one immutable pair context, and released when that context ends, including exceptional exits. Parent geometry reuse remains bounded; complete candidates, graphs, samples, features and learned prior probabilities are retained.

`verify_reciprocal_cached_relation_prior.py` authenticates a complete input-only pool and checks every saved relation feature and penalty. It also checks reversed pairs, original directed arithmetic, changed contexts, workpiece resets, invalid pairs and the same subset in opposite benchmark orders. This independent helper does not replace a running training backend. Numerical and substep timing controls do not establish whole-training speed or extraction quality; optimizer compatibility and authenticated adoption are separate requirements. Private receipts, geometry, results, weights and account state remain outside this source snapshot.


## Complete model output export

`server_full_weld_csv_export.py` writes one row for every selected ordered child together with exact parent and child JSON. It preserves raw endpoint states and confidence, analytic curve parameters, circular traversal signs, and original signed endpoint axes and origins. Unknown values stay blank with explicit states. Compatible wall, outer-edge and no-hole codes are exported separately from exact millimetre values. Incompatible predictions remain visible without rewriting their state. Parent Any flags are never broadcast to children; no short-child deletion or overlap merge occurs.

`export_full_weld_predictions.py` authenticates frozen model outputs and complete held roles using the existing role verifier. TEST requires the prior validation-fixed model. It reads no reference geometry, fits nothing and checks a complete CSV/JSON round trip. TRAIN snapshots from different updates remain diagnostics, not one model's score. The semantic verifier checks missing fields, scalar codes, raw signs, a short child, arcs and a closed circle, and rejects modified frames, invalid outputs or changed denominators.

`continue_full_output_export.py` waits for the existing final TEST controller's real completion, verifies the fixed model pair, then exports selection, protected diagnostics and final TEST without changing their roles or model selection. Admission requires current semantic controls and a nonempty authenticated actual TRAIN export. `archive_completed_model_csv_export.py` can losslessly bundle a completed generated TRAIN export; all members are read back and matched to original hashes before redundant raw artifacts are removed. Its persistent archive restores the original relative CSV and curve paths. Raw inputs, predictions and weights are outside that archival operation.

This is a canonical development interchange format. Business segmentation reasons, missing plate normals, untrained child flags, customer spreadsheet layout and physical signed-frame semantics are not certified by exporting data. Export completion does not establish extraction quality, independent generalization or final input-to-output timing. Private output bundles, numerical results, weights, datasets and account state are excluded from the source repository.

## Reciprocal distance reuse in complete TRAIN fitting

The reciprocal TRAIN backend combines the existing admitted bounded caches and exact reference compiler with pair-scoped reuse of original primitive-distance arrays. Both directed original relation calculations still run; candidate geometry, samples, input graph, features, priors, complete-set loss, AdamW settings and registered TRAIN visits remain unchanged. `reciprocal_relation_training_backend.py` requires the full numerical relation control and an actual complete-job eight-step optimizer comparison before production admission. It rejects stale implementation fingerprints and mismatched backend evidence.

`verify_reciprocal_relation_training_backend.py` compares all job tokens, every reference view and cost, eight actual updates, optimizer states, final scores and selections against the already admitted bounded recipe. `train_reciprocal_full_recall_parent_dataset.py` uses the original complete TRAIN orchestration. `continue_reciprocal_full_recall_training.py` waits for complete query fitting, then trains all registered input jobs and retains the original separate selection and protected diagnostic roles. Final TEST still requires a model fixed through validation. Numerical compatibility and a local runtime measurement do not establish new model quality or complete training speed.

## Full input covariance audit across model-use roles

`audit_full_cross_role_rigid_invariants.py` reads every admitted workpiece XYZ point and its multiplicity, using actual TRAIN, selection, protected diagnostic and TEST registries. It checks every pair crossing those roles. Counts and covariance eigenvalues are necessary conditions for rigid near-bijections; the screening bound includes coordinate perturbation and a conservative float64 allowance. Input hashes and complete registration coverage are authenticated. Matching moments remain unresolved registration candidates. They are never interpreted as physical identity or inserted into the model.

The audit reads no weld reference geometry, fits or scores no model, and changes no input, role, split or signed business frame. The tolerance concerns only input identity screening. Full registration, cropped or resampled derivatives, historical exposure and independent generalization are not certified by this audit.

## Input registration with different point counts

`audit_full_cross_role_input_alignment.py` follows the full moment audit by registering every actual cross-role input pair. It applies no equal-point-count or covariance retrieval exclusion. Full-input principal axes seed proper rigid rotations; two trimmed ICP recipes propose full and partial overlap transforms. Every retained sample contributes to the final distances in both directions. A synthetic case verifies recovery of a known rotation and translation despite different point counts.

This is an input-only diagnostic. It changes no model, role, split, raw point cloud, inference output or signed business frame. Sampling, a finite pose search and proximity statistics cannot prove physical identity or exclude all cropped and resampled derivatives. Suspected matches require full-point follow-up; independent generalization and historical exposure remain separate assessments.

## Full-point follow-up of cross-role input hypotheses

`audit_full_point_alignment_followup.py` authenticates a completed cross-role sample audit and chooses its closest full-shape and partial-shape hypothesis for each pair of model-use roles. It keeps each proposed proper rotation and translation fixed, then scores every original XYZ point and its multiplicity in both directions. Chunked queries bound temporary memory without omitting points. Unequal counts, transformed duplicates, directional crop overlap, chunk boundaries and reflection rejection have numerical controls.

The output reports distance distributions and overlap fractions in original input units. These are diagnostic measurements at selected fixed hypotheses, not business acceptance thresholds, registration optimality, physical identity or a whole-dataset independence certificate. No model is fitted, no weld references are read, and no input, role, split or signed business frame is changed. Private point clouds, evidence and results remain outside this public source snapshot.

## Exact point-support containment across model-use roles

`audit_cross_role_exact_point_support.py` checks every directed cross-role pair against a nearest-point tree containing every target point. A bounded source sample only rejects impossible exact-support hypotheses. Every original source point must then match a target coordinate exactly before complete support containment is declared. Source and target counts need not match; duplicated coordinates and additional target geometry are retained. A negative control changes a point omitted by the screen and verifies that the full check rejects it.

This is a read-only input audit in the confirmed original millimetre frame. It fits no transform or model, reads no weld references, and changes no role or geometry. Coordinate support containment is not multiset injection or proof of physical identity. The audit does not exclude arbitrary rotated or resampled derivatives. Confirmed cross-role containment requires grouped-split review before independent evaluation; source samples alone cannot certify independence.

## Corrected partitions and complete training continuation

`repair_full_dataset_family_split.py` consumes a completed cross-role input-support audit and conservatively joins confirmed related inputs while retaining every label version. Existing TRAIN exposure remains TRAIN; protected diagnostic permissions remain separate. `qualify_repaired_full_dataset_labels.py` runs the original qualifier against all original sources and checks geometry, attributes, typed array bytes and signed frames against the previous qualified views. `register_repaired_full_dataset.py` registers every admitted TRAIN, selection, protected diagnostic and TEST view in a separate dataset directory. Exact coordinate containment is one derivative test, not certification of arbitrary rotated or resampled physical identity.

`server_partitioned_full_dataset.py` inherits the original complete input, supervision and held readers. `reuse_repaired_full_input_shards.py` authenticates identical raw-cloud SHA256 and every original feature file before rebinding owner metadata. It never uses another cloud’s features merely because that cloud contains similar points. `complete_partitioned_full_training_inputs.py` incrementally consumes the existing full input producer and executes all original point/geometry stages for additional TRAIN inputs through `prepare_partitioned_full_training_case.py`. Resource and storage gates retain their original thresholds; errors preserve evidence rather than silently skipping an input. Temporary feature links require their recorded caches to remain available.

`train_partitioned_full_weld_reference_recall.py` loads a genuine earlier full-state query checkpoint with unchanged tensors and TRAIN normalization. Historical updates are pretraining, not completed passes under corrected family weights. The corrected curriculum executes fresh complete passes with the original optimizer and losses; its original optimizer state remains archived in the earlier checkpoint. `prepare_partitioned_full_task_supervision.py` re-executes the original full owner-witness algorithm in the declared partition.

`run_partitioned_full_dataset_stage.py --dataset-name <partition> --module <preserved-entry> -- <original-arguments>` binds registration readers, storage and child process routing in an isolated process. Original training and evaluation functions remain unchanged. Its executed CLI AST redirects only the original fixed output-directory literal, and the execution binding records the original source plus the context. `server_partitioned_query_fit.py` reuses every original complete-view and per-step check against the declared TRAIN registry; partial query checkpoints cannot authorize full parent training or final TEST. Complete parent training, validation-only selection, protected diagnostics, final TEST and canonical output export use the same frozen model pair and original numerical controls.

Relevant preserved entries include `tools.continue_reciprocal_full_recall_training`, `tools.prepare_full_evaluation_inputs`, `tools.continue_validation_fixed_test` and `tools.continue_full_output_export`. They still require locally qualified datasets, numerical-control receipts and authentic frozen weights outside this public code repository. Executable orchestration, metadata controls and completed input preparation are not extraction accuracy, independent generalization or full acceptance results.

## Finite contact region and local plate recovery

`server_supported_joint_regions.py` visits every finite original line candidate and keeps actual observed face-region pairs separate. It merges only overlapping intervals of the exact same geometric line. Cached face centers cannot reject an original endpoint line before local raw surface fitting. `server_plate_pair_evidence.py` preserves the original local plane, opposite-face, coverage and directed-normal evidence checks. Stable thickness regions are retained separately; ambiguous transitions keep unknown attributes and require the same witnessed own faces on both sides. Unsupported observation gaps and short supported regions are preserved. Geometric sorting directions are not business weld coordinate frames.

`verify_supported_joint_regions.py` checks candidate order, missed later intervals, local thickness changes, stale plane centers, finite face identity, dense-neighbor support, short regions and single-plate rejection on synthetic geometry. `recover_all_finite_joint_regions.py` replays all declared raw inputs into separate unlabelled contact hypotheses. These controls and hypotheses are not training labels, confirmed welds, a learned model quality result or a completed customer contract evaluation. Private inputs, results and checkpoints remain external.

## Final model regression against original baseline inputs

`continue_validation_fixed_baseline_regression.py` waits for the existing complete validation-fixed final TEST execution, verifies complete query and parent fitting, and applies the same frozen model pair to every input in an external original-baseline collection. The collection must preserve all original cases, references and baseline prediction bindings and match each raw input SHA256 to an authorized TRAIN input. Every new prediction is frozen before original reference geometry is read. Baseline results do not select or promote a model; a historical score is not substituted for the actual current baseline. Rebuildable complete candidate pools are evicted only after durable outputs and byte checks.

`verify_validation_fixed_baseline_regression.py` checks every original nonempty baseline score against its saved evidence and rejects missing or duplicate cases, held inputs, substituted input/reference/prediction files and a genuine unfinished query checkpoint. Actual completed final-model regression remains a separate execution. This development geometry comparison does not certify held independence, attributes, signed business frames or contract timing. Private collections, scores and weights remain external.

## Original frozen-input inference and reference views

`verify_original_best_frozen_replay.py` distinguishes archived prediction scoring, actual inference using the original frozen intermediate inputs, and prior reconstruction from raw point clouds. It requires every original TRAIN case, identical full ordered raw XYZ, authenticated model and prediction bytes, audited inference file reads, and separate original and current parsed reference scores. Inference writes only to a new result directory; target and STEP reads are rejected. Full decoded-output parity includes every saved prediction field. Original-input parity is not complete raw-input pipeline reproduction or independent model quality.

`continue_original_best_reference_comparison.py` waits for the existing complete final-model regression and scores exactly its already frozen predictions against the authenticated original reference view. The same final model pair and all original cases are required. Current parsed-reference results remain separate; historical references do not change training labels or select a model. Private inputs, intermediate artifacts, references, scores and weights remain external.

## Checkpoint-declared backbone training source roles

`audit_backbone_training_role_bindings.py` reads the actual checkpoint-embedded training-source declarations and their authenticated original cohort arrays. It compares every registered raw PLY with the same point count against each complete original ordered XYZ array, with no coordinate transform. Exact matches retain current TRAIN or held roles and every registered label version; ambiguous or absent matches remain unresolved. Same-count screening does not exclude cropped, reordered or transformed derivatives. Checkpoint declarations and exact input identity do not reconstruct all historical fitting, missing older data manifests or external exposure. No model fitting, extraction scoring, role changes or independence certificate is produced. Local original artifacts and registered raw inputs are required and remain outside this public repository.


## Held input preparation in a separate process

`tools/prepare_lightweight_partitioned_held_inputs.py` resumes an authenticated held input preparation context while loading only the original input controller and workers. This preserves the CPU fragment producer's requirement that training model modules have not been imported. It retains completed stage receipts, original feature functions and frozen point weights, and records the new launcher separately from the prior failed context.

Pass `--dataset-name`, a new `--preparation-name`, and the existing `--previous-preparation-name`. The prior registration, source hashes and stage receipts must remain valid. The launcher uses the original controller lock and owned input storage; it prepares input features without fitting models or scoring test references.


## Audit complete TRAIN task assignments

`tools/audit_complete_training_task_coverage.py` checks all registered TRAIN views against a frozen complete query checkpoint, authenticates the owner witnesses and labels, and reports carrier gaps and endpoint label masks. A missing carrier witness does not remove that child from free geometry supervision and is not an inference recall ceiling. This audit fits no model, reads no held reference and reports no prediction accuracy. Its reports remain local experiment artifacts.


## Diagnose TRAIN arc witness gaps

`tools/diagnose_training_arc_witnesses.py` reproduces missing TRAIN arc witnesses with the original input profiles, conservative circle filter, projected intervals and D3 rules. It separates proximity to a finite input arc from proximity to its underlying circle and records ordered endpoint, tangent and whole-curve errors. Reference projections are offline diagnostics; the tool does not extend deployed curves, insert reference endpoints, fit models or score held references.


## 原始XYZ局部信息与几何精度训练

新增 `server_local_query_weld_model.py` 和 `server_query_point_neighborhood.py`：保留完整冻结原查询及其输入归一化，在实际原始XYZ邻域上添加零残差初始化的局部信息分支。训练责任分配使用已部署的输入网格，参考只分配完整监督，不生成输入查询；缺匹配载体的子段继续学习自由几何，已知属性、坐标系、父序和连接均保留。

`tools/train_full_local_query_geometry_mixed.py` 在不同完整训练视图间交替更新，保留全部原始种子变化。`server_cached_local_query_training.py` 保存真正的输入数组及逐查询邻域绑定；`server_memoized_training_queries.py` 仅复用原计算的确定性准备结果，不缓存预测，也不进入部署或held推理。完整准备等价入口为 `tools/check_full_train_query_memoization.py`。

`tools/train_full_local_query_geometry_metric.py` 从实际已完成的局部模型继续学习，开启独立优化器。`server_metric_geometry_losses.py` 将直线端点欧氏误差、65个有序解析圆弧采样点误差及已知载体区间误差按3毫米量纲计算；原类型、方向、闭合、已知字段掩码、坐标系与父连接损失保留。训练中的65点采样不替代验收时的完整曲线采样要求，几何和时间验收门槛未修改。

`tools/check_local_metric_warmstart.py` 要求显式 `--version`，在一份包含直线和圆弧的完整TRAIN工件上检查实际更新、其他子损失逐字节保持及冻结原参数保持。公开入口不含私有样本默认标识；准备等价检查的分母由训练登记动态计算。这些调用参数和登记数量的通用处理不改变几何学习算法。运行仍需外部数据、登记、权重和检查回执，不能仅靠克隆本仓库复现训练。

`tools/compare_full_query_geometry_models.py` 与 `tools/compare_cached_full_query_geometry_models.py` 检查同一输入查询下的完整训练单子几何；旧预测只能在逐记录哈希、权重和输入责任绑定全部相同后复用。允许独立子反向的局部诊断不是完整父焊缝提取准确率，也不是独立测试成绩。`tools/run_local_full_dataset_stage.py` 和 `tools/run_local_baseline_stage.py` 支持该完整参数架构，实际完成全部训练覆盖后仍需完整父集合、验证固定、最终测试、字段及CSV和原最佳同条件对照。此分支尚未被验收或替换原最佳模型。


## 实验性整父训练关系复用

`server_disk_cached_parent_competition.py` 在一份冻结TRAIN输入内，用有界临时SQLite存储复用原32列关系特征及原冻结单线程ExtraTrees先验。输入几何变更会拒绝复用；全输入邻域图、原评分、全部请求的边和完整参考分母保持。空间不足只停止存储并重算，不减少候选或输出。缓存随所属工件关闭清理，不保存正在训练的selector分数，不跨运行复用预测。

`tools/verify_disk_cached_parent_competition.py` 比较全部冻结原关系并检查反序回放、输入变更及存储上限；`tools/verify_disk_relation_training_backend.py` 在完整实际输入上执行两种方法的八次更新，比较全部历史、参数、优化器和最终选择。`tools/run_disk_cached_local_full_dataset_stage.py` 要求这两项当前源码绑定的证据，才在独立进程中替换TRAIN关系读取方法；完整查询拟合、父集合、验证固定、测试、字段、CSV及原最佳回归仍走原入口。

该复用方案服务于训练迭代。它不证明新模型质量或完整推理十秒达标；后者必须从新鲜原始XYZ开始计时，不能用已有关系回放代替。数据、冻结关系数组、权重及私有检查结果不随源码发布。

## 实验性米制端点修正

`server_interval_endpoint_residual_model.py` 在已完成的局部 XYZ 模型上增加零初始化的沿载体端点位移，保留原自由几何、类型、属性、坐标系及父连接输出。位移以米表示，推理端点投影到包含实际输入查询的位置范围；训练投影采用恒等回传，允许越界修正继续学习。

`tools/compare_interval_endpoint_residual.py` 用同一完整 TRAIN 工件、原始 XYZ、输入查询、监督和目标比较两种方法。它检查初始输出逐字节一致、原冻结参数不变以及真实更新。该新结构已有完整数据训练入口和严格模型核验，以及整父训练、验证固定、最终测试和 CSV 接续入口；原最佳回归已接入，真实输入至 CSV 耗时仍需核验，未替换原最佳模型。运行需自行提供已完成的模型及外部数据。

全量入口：`tools/train_full_interval_endpoint_geometry.py`，它以有计数核验的源码转换复用原完整训练器，保留所有数据、字段掩码、更新、恢复和完成检查。`server_interval_endpoint_checkpoint.py` 单独核验新增参数及完整训练记录，拒绝部分检查模型。`tools/run_interval_endpoint_full_dataset_stage.py` 与 `tools/compare_interval_endpoint_full_query_geometry_models.py` 接续完整输出与同输入位置对照。

`tools/run_interval_endpoint_baseline_stage.py` 以同一验证固定模型评估原最佳的全部原样本及精确历史标签。`tools/verify_interval_endpoint_baseline_regression.py` 保留原输入、标签、预测与评分的所有权检查，使用真实诊断模型及其实际记录拒绝不完整训练。回归结果不用于选模型。

`tools/compare_saved_complete_query_diagnostics.py` 在全部注册 TRAIN 预测记录、完整模型和输入责任均核验一致后，逐条比较两份已完成的局部诊断，列出改善、退步及分类结果。它可等待明确绑定的现有评估进程，不重新训练或推理；独立子反向和分支择优检查仍不代表完整焊缝提取或测试准确率。

`tools/watch_train_file_cache_headroom.py` 独立核验存活 TRAIN 进程，在内存达到阈值时对未被打开或映射的登记原始文件给只读缓存释放提示。它核对原文件元数据和进程身份，不改文件内容、权重、优化器或训练配置；此操作用于训练资源管理，不证明提取质量或原始输入至 CSV 的推理时间。

## 完整几何学习对照

`tools/compare_nonlinear_geometry_residual.py` 对照零初始化的非线性端点与自由圆弧修正；`tools/compare_geometry_direction_surrogate.py` 对照保持前向几何输出、让几何误差训练方向判断的实验梯度。两者只使用已登记 TRAIN 的完整工件和原始 XYZ，保留全部已知字段、父段和连接监督。独立小范围对照不能代表全数据、完整提取或验收成绩。相应模型位于 `weld_pipeline/server_nonlinear_geometry_residual_model.py` 和 `weld_pipeline/server_geometry_direction_surrogate_model.py`。

`tools/compare_line_direction_surrogate.py` 进一步把实验梯度限定为预测直线的方向头，并核验该修正对输入上下文的梯度为零。它保持原有圆弧与已知字段监督，仍需要完整数据和最终输出评估。模型入口为 `weld_pipeline/server_line_direction_surrogate_model.py`。


## 保留有限曲线的有向组装与完整重复检查

`weld_pipeline/server_supported_directed_parent_pool.py` 保留每条输入曲线的两个方向，将可训练连接评分用于实际父候选搜索。共享切点要求共同输入事件和精确坐标；组装保持有限线弧及端点坐标系，不进行端点平均或重新拟合圆弧。`tools/check_supported_directed_parent_pool.py` 检查几何、端点字段、真实训练梯度进入运行、空输入和资源上限。

`tools/check_original_best_complete_repeatability.py` 在独立进程核对原始冻结输入的完整候选决策、父序列、全曲线与 CSV。`tools/check_interval_complete_output_repeatability.py` 对已登记 TRAIN 原始点云重新生成全部查询和候选，使用固定完整查询权重与明确的部分父权重快照检查完整输出重复性。它们是开发控制，不代替完整训练、准确率、独立测试或原始输入全流程时延验收。


`tools/check_interval_fixed_seed_training_repeatability.py` 在两个独立 CUDA 进程中重复一个完整 TRAIN 视图的短训练，逐项比较输入、监督、损失、权重、优化器及输出。`tools/audit_complete_query_reference_collisions.py` 核对全部 TRAIN 的实际查询是否对应多个几何或父结构，保留所有参考版本，不自动把差异认定为错误标签。

`weld_pipeline/server_supported_interval_events.py` 为有限直线和圆弧生成统一输入事件，保留原始边界、观测范围、原始点投影和网格来源，整对选择端点后直接构造原曲线上的区间。网格不代表接触证据，观测空隙不会被标成有支持。`tools/prepare_supported_interval_training_events.py` 通过同一生成函数准备全部已登记 TRAIN 视图所需的输入事件缓存；训练责任分配只选择待缓存载体，不生成事件坐标。运行时仍需访问完整输入，训练缓存完成不代表模型完成训练或验收。


## 多视角点云图像辅助实验

`server_multiview_visual_features.py` 将完整 XYZ 投影为三个方向的点密度和深度图，提取多个邻域尺度的视觉信息，再映射回每个原始三维点。它不读取标签，不依赖已有焊缝候选，也不把图像像素直接当三维焊缝。

`tools/probe_multiview_visual_assistance.py` 在已登记 TRAIN 内按完整家族分组交叉训练，对照单视图与三视图。拟合使用有权重的点采样，推理覆盖所选工件全部原始点；报告点级误报、召回和参考曲线附近点的覆盖。此辅助实验不访问正式验证或 TEST，不替换原最佳权重，点区域成绩不能称完整焊缝提取准确率。运行需要外部数据和登记；可用 `--focus-job` 指定 TRAIN 工件，分组自动包含同家族的其他输入。


`tools/probe_visual_supported_arcs.py` 对已完成交叉拟合的视觉分数进行有限圆弧可行性检查：仅用高分原始点及观测高度层拟合圆，按实际支持的空隙分割范围。先冻结全部候选，再读取 TRAIN 标签衡量候选覆盖能力。当前范围仅为与 XY 平行的弧，不具备任意姿态保证；参考投影只用于离线测量，不能当作已预测的端点，候选覆盖率不是完整提取准确率。


## 全量视觉辅助训练与候选补充

`tools/train_full_visual_assistance.py` 使用全部已登记 TRAIN 工件及版本监督拟合视觉辅助模块。训练按原家族/版本权重分配，点采样按概率校正；全部输入点都参加推理。已知字段和整父关系仍由后续完整焊缝模型训练，此入口不宣称已经训练完整模型。正式验证/TEST 不用于该拟合。

`server_visual_candidate_augmentation.py` 是训练/运行共享的候选补充入口，保留全部原候选及编号，只追加由视觉高分原始点支持的有限弧。新增候选必须走自己的真实 XYZ/视觉特征，不伪造旧特征列。生成器仍限 XY 平行弧，误报、端点、任意姿态与完整输出需要后续处理。`server_tiled_multiview_visual_features.py` 对大尺寸输入使用重叠分块，保留完整滤波邻域和固定像素大小，不裁掉原始点。


`tools/prepare_visual_augmented_full_supervision.py` 在全量视觉候选完成后，为所有新增曲线生成真实输入事件，并追加通过原几何检查的训练区间。完整参考父结构、属性及坐标系仍绑定原标签，原有区间全部保留，所有缺候选的标签继续进入分母。`server_visual_augmented_input.py` 保留原输入编号与特征读取，对新增曲线拒绝伪造旧特征；完整模型仍需实现其独立特征分支。标签投影仅作为训练监督和容量测量，不进入运行候选或预测端点。


## 完整区间模型接入组件

`server_supported_interval_weld_model.py` 将完整的输入端点对评分与原模型的已知属性、坐标系及父关系预测连接；输出几何来自所选有限曲线，不再使用自由端点/圆弧回归。原候选继续使用真实旧特征，新增视觉曲线有独立的原始 XYZ、图像及几何输入分支。`server_visual_curve_context.py` 生成这种真实输入，`server_visual_augmented_training_samples.py` 保留全部参考和属性所属关系，同时分开两种输入行。

`tools/check_supported_interval_warm_fields.py` 核验完整已训练检查点，在一件完整 TRAIN 视图上检查零残差初始化的属性保持，并执行实际字段、父结构和连接更新；控制使用完整输入范围作为区间，不宣称已学会端点或完成全量训练。完整区间训练器、实际父候选生成与全量输出评价仍需接入。


The supported interval training stage now optimizes every endpoint event, complete paired interval alternatives, known weld fields, ordered parent closure, and the actual directed runtime link head. Native inputs retain genuine legacy features; added visual curves use separate raw XYZ/image features. Training preserves multiple owner alternatives and masks unavailable shared boundaries without relabeling them as gaps. Three complete TRAIN passes are registered; complete extraction, nullable parent selection, CSV and held evaluation remain separate required gates. No model acceptance is claimed by the optimization stage.
