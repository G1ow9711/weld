"""Prepare the next authorized review from actual completed results only."""
from pathlib import Path
import json
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/unified_parent_scoring_v1'
def main():
 sources={'training_set_actual':BASE/'set_competition_support_v5/evaluation_summary.json','conditional_both_heads_heldout':BASE/'set_competition_dual_heldout_v1/evaluation_summary.json','training':BASE/'competition_head_v4/training_receipt.json','fallback_faults':BASE/'solver_fault_fallback_v2.json','fallback_normal':BASE/'set_solver_fallback_normal_controls.json'}
 data={k:read(p) for k,p in sources.items()};dest=ROOT/'server_rebuild/pro_review/competition_v4_20261011';dest.mkdir(exist_ok=False)
 compact={k:dict(categories=data[k]['categories'],cases=data[k]['cases'],five_TRAIN_geometry_gate=data[k]['five_TRAIN_geometry_gate']) for k in ['training_set_actual','conditional_both_heads_heldout']}
 text='''继续前次焊缝项目技术复核。请基于实际证据批判下一步，不建议扫阈值、按工件硬编码或使用DEV64调参。用户允许自主咨询，当前仍未验收。
原服务器B0恢复254/269 TP FP4；上轮实验269/269 TP FP7（Hline FP7，分类门≤6），未推广。总体门TP≥256/269 FP≤13，Hline≥128/134 FP≤6，Vline≥126/132 FP≤6，HArc3/3 FP0。
本轮修改：竞争表示逐子段按真实直线/圆弧弧长33点并保留所有端点，不跨间隙，解析弧长中心；整体反向、点密度、刚体、循环起点、候选重排控制。发现闭合对开放比较起点依赖，修为任意一方闭合允许相对循环。全圆环仍显式不支持，不能宣称已覆盖。原8004父、原计数头固定；竞争头原固定96树等超参，仅对双方唯一GT对应监督，歧义未知。正常与故障回退选择器分开版本测试，不中途更改该实验。
五件实际选择评分及双头条件留出如下（上游表示/候选共享冻结，绝非全链OOF/独立验收）：
'''+json.dumps(compact,ensure_ascii=False,indent=2)+'''
已完成其他审查：7旧FP为6个完整错误父，单父TP0，整父删除不丢269TP，但未做ID过滤。已知唯一GT竞争对检索全覆盖，未知竞争覆盖不证明；集合交集/非传递反例已测试。
STEP只离线：从原XYZ无焊缝GT拟合配准，修复gmsh返回占位UV的错误；10/25/36共264条直线各5点均对应唯一一致的不同实体非平行面配对。原生有限面布尔线交连续支持严格10=120/208、25=10/24、36=26/32；10其余88只端部缺口≤0.01598mm，25其余14只端部缺口≤1e-6mm，36其余6某侧交集为空，不当作真实不存在焊缝。完整原实体未改变。
计数support_v2监督仍有未解除的假设：全部预测直线位于同角色显式有限标注线3mm凸管域就开放D3计数监督；这不是有限接触/局部标注闭合证明。没有因CAD抽样而新增负标签。最终推理只能XYZ+unit，不能STEP/GT/工件ID查表。
请回答：1. 本轮实际/联合留出结果说明主要误差来源是什么？给出最小可证伪的下一项模型实验，不用测试集调参。2. 对已有明确导出焊缝父列表，局部监督闭合究竟需要什么足够且可实现的证据？避免无限追求CAD零误差，同时不把未标注真实接触当负。3. 36近9mm端点错位与10重复截短如何用XYZ有限面/端点事件可学特征解决？哪些输入/标签对照必须做？4. 正式全链/属性/CSV、≤35分段、多底板台阶与≤10秒仍待完成，如何合理组织下一轮而不在小训练集无休止修补？
请区分当前可执行的有限实验与真正验收证据。64仍未运行，先不推广。
'''
 (dest/'request.txt').write_text(text);write(dest/'request_receipt.json',dict(sources={k:digest(p) for k,p in sources.items()},request_sha256=digest(dest/'request.txt'),source_sha256=digest(__file__),status='PREPARED_NOT_SENT'))
 print(dest,flush=True)
if __name__=='__main__':main()
