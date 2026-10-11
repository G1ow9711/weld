"""Read-only failure diagnosis and local continuation packet for run272."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.world_set_evaluation_285 import reference_segment


def main():
    run=OUT/'run272';r=read(run/'receipt.json');evaluation=read(run/'evaluation.json')['summary']
    assert digest(run/'prediction.json')==r['prediction_sha256']
    prediction=read(run/'prediction.json');choices=read(run/'choices.json')
    case=next(c for c in read(OUT/'run98/local_cohort.json')['cases'] if c['source_key']=='21')
    assert digest(OUT/case['target_file'])==case['target_sha256']
    target=native_targets(read(OUT/case['target_file']))['parents'][0]['children'][0]
    reference=reference_segment(target);rows=[]
    for parent,choice in zip(prediction['parents'],choices):
        assert parent['predicted_parent_query']==choice['query']
        segment=parent['segments'][0]
        error=child_error(segment,target,reference,False,False)
        rows.append(dict(query=choice['query'],probability=choice['probability'],profile=segment['profile'],
            relation=segment['relation'],predicted_side=segment['predicted_side'],
            individually_geometry_compatible=error is not None,
            maximum_error_mm=None if error is None else max(error['bidirectional_sampled_maximum_mm'],
                *(error['ordered_endpoint_errors_after_allowed_whole_correspondence_mm'] or [0.]))))
    assert len(rows)==7
    diagnosis=dict(status='FROZEN_OUTPUT_DIAGNOSIS_ONLY',rows=rows,
        unique_selected_profiles=len({x['profile'] for x in rows}),
        individually_compatible_outputs=sum(x['individually_geometry_compatible'] for x in rows),
        original_targets=1,official_TP=1,official_FP=6,official_score_unchanged=True,
        side_predictions_not_certified=True,optimizer_updates=0,utc=utc())
    write(run/'duplicate_diagnosis.json',diagnosis)
    failures=dict(architecture='complete_parent_set_with_MLS',completed_train_test_rounds=1,failed_rounds=1,
        significant_local_improvement_over_previous_architecture=True,
        full_scope_acceptance_not_achieved=True,old_recursive_failed_rounds=3,old_local_potential_failed_rounds=3,
        train_runs=['run270','run271'],test_run='run272',updates_total=30,
        subsequent_cases_not_tested=['BM79A_native_local','10','64','4','25','36','42','24'],
        full_pilot_G=500,next_training_requires_completed_Pro_review=True,utc=utc())
    write(OUT/'parent_set_trial_ledger.json',failures)
    report=(run/'模型实测指标.md').read_text(encoding='utf-8')
    report+='\n\n本轮训练：21/BM/10三件共213条标签，10epoch/30更新，累计107.925秒。21号是训练内可行性测试；其余8件尚未提取测试，未扩至65件。\n'
    report+=f'\n只读诊断：7个输出均选中同一原始profile，其中{diagnosis["individually_compatible_outputs"]}个单独与同一条真值满足几何门。官方一对一计数仍为1TP、6FP。模型的重复输出抑制尚未学好；这不授权用几何合并或Top1删输出。\n'
    report+='\n与run246同一21号开发口径相比，误检从60,157降到6；命中仍为1/1。最大误差从0.404mm变为1.634mm，均仅对应命中的1条，不能推断整体数据集效果。\n'
    (run/'模型实测指标.md').write_text(report,encoding='utf-8')
    (OUT/'current_actual_training_status.md').write_text(report,encoding='utf-8')
    review=OUT/'pro_review_run272';review.mkdir(exist_ok=False)
    packet='''SESSION_CONTINUATION_V1
请继续同一项目的Pro246完整父对象集合方案复核。最后独立一行返回：GPT6_PRO_RESULT_SJG_RUN272_20261009

本轮未达到快速门，依用户要求先咨询再修改/训练。只发送我方实现和指标摘要；没有原始点云、STEP、客户文件、第三方源码或凭据。

1. 采纳与新增授权
已按你的上轮建议退休局部势TCN主预测器，A仅冻结独立诊断，不输入B。新B完整原profile有限域对象、669上下文+16有序长度加权分箱，256父queries/宽64/2层4头全对象cross-attention、每父最多5子、每子独立profile指针和完整锚点区间指针+连续fraction。共享节点单一权威，正gap独立，支持跨载体线/弧。
用户随后明确要求加入其所提供项目的几何预处理思路。我们独立实现纯XYZ双尺度10/30mm二阶局部拟合残差、平面/曲率/可靠性掩码，最多64原近邻，增加54通道，分箱188变242。未移动/删除XYZ、未作互相近邻补集/SOR硬删。不能把有效拟合比例当精度。新增阶段21/BM/10耗时3.797/7.153/37.943秒，不是合同全流程计时，原10基础输入准备335.279秒。

2. 输入和控制
21/BM/10全原始profiles=5250/12141/116216，无TopK候选筛选。7TRAIN原始G363的表达控制363/363，集合解码17解释→1父、同几何两侧→2父、短缝、共享节点、微gap、混合线弧均控制通过。不是模型准确率。
重要修正：7TRAIN过去统计的唯一POSITIVE_GAP在25号，其原标签端点差仅约1.14e-13mm，不能证明真实工艺间隔覆盖；保留原始字段/分母，没有偷偷改SHARED。固定科学三件G213不含真实正gap覆盖。
全候选softmax分母、注意力和分块编码器做稠密前向/梯度对照。匹配与反传实际成本差约3.8e-6或0。MLS通道有非零梯度。所有上游冻结、GT只用于训练监督和冻结后独立评分，推理不读GT/STEP。

3. 完整实际训练
固定seed1031，21→BM→10，10epoch×3=30更新，AdamW lr3e-4/wd1e-4/clip5，累计107.924835秒，180秒预算未放宽。初始一轮在3更新后因1.25GiB可用RAM门中断，随后保存模型/优化器/历史，从第4更新续训，累计30而非重跑。只改自身只读映射页回收，资源门保持，最终权重一次评分。旧中断checkpoint没有RNG，但本网络无dropout/随机训练层或后续随机采样；这一限制披露。GPU初始化和输入SHA核验墙钟时间另计，不能说总共只等了108秒。

4. 目标函数具体实现，请检查是否存在隐患
每父完整latent witness动态规划；共享见证转换成本为相邻原兼容端点距离/3，正gap要求原见证端点不完全相同。没有把多个profile见证复制为阳性。
每query/GT父正成本 = softplus(-existence) + count CE + parent_closed BCE + mean(child costs + transition costs)。child成本是：全profile categorical NLL + 全锚点interval NLL + fraction L1 + 3D有序控制点误差/3mm + type/direction/closed/wrap/已知connection NLL。端点分类和坐标误差在选中latent见证的完整原载体/目标锚点区间条件下监督；推理则自行argmax profile/interval。共享的直线起点不重复监督，弧保留中点控制；未知side没有伪造训练监督。
总loss = matched positive mean + unmatched query softplus(existence) mean。Q=256、P个真父的Hungarian cost = positive/P - background/(Q-P)，反传同一正成本与NO_OBJECT归一。匹配与反传中均保留全候选分母；无GT-count inference/NMS/union/Top1。阈值固定0.5。
父queries只各自cross-attend全对象，目前没有query之间的self-attention，也没有显式预测父数量损失。请重点检查稀疏P=1和密集P=124共同训练、两个均值归一、query缺少竞争是否会造成下面的重复检出。

5. 最新真实测试——失败，不扩测
三件训练后只测首门21号G1：TP1、FP6、FN0，共输出7条。TP/G=100%，TP/(TP+FP)=14.2857%，命中那1条端点/双向最大误差1.634249441mm。缓存上游后的下游推理1.771775秒，非合同无缓存时间。完整父结构开发评分，正式方向/侧别/E16尚未验收。
按登记fail-stop，BM/10/DEV64及其他原九件未继续提取测试，G500不改。不能用TRAIN单件100%声称达到合同>95%。本架构第1次完成训练/测试失败；旧递归3失败、局部势TCN3失败保留。
同一21旧run246：TP1/FP60157/max0.403607mm。误检明显减少但仍失败，精度误差反而增大但匹配条仍<3mm。未做只改MLS的消融，不能把全部改善归功于MLS。

6. 冻结输出只读诊断
''' + f'7个输出全部选中同一原始profile，{diagnosis["individually_compatible_outputs"]}个单独与同一真值满足3mm几何门；一对一官方计数仍只有1TP、6FP。\n'
    packet+='存在概率分别'+', '.join(f"{x['probability']:.9f}" for x in rows)+'。\n'
    packet+='7条都预测单子水平直线、同一两锚点区间，只有连续fraction微差；不存在这7条代表不同原profile/不同材料关系的证据。预测side头未得到已确认标签监督，不能用其A/UNKNOWN输出作真实侧别或作为合并依据。\n'
    packet+='''
7. 本地判断与请你作出的决定
局部判断：模型已选中正确几何位置，主要剩余首门错误是多query重复检出；粗暴Top1或按距离合并会损伤多焊缝/双侧，不能采用。两个均值归一可能在不可区分queries时偏向0.5附近；仅看几何位置损失会掩盖集合身份问题。另需检查teacher-interval条件训练与argmax推理差异。
请提出下一轮最小但真正针对原因的修改，不要再只增加审计或重复软件控制。比较并给出一个明确首选：query竞争/self-attention、计数/基数监督但运行时不用真数量、对最难重复背景的目标、目标成本重标定/存在性校准、或你认为更合理的替代。必须保留同几何不同关系/侧别可输出两个、完整父路径和所有原候选，不以GT阈值/计数筛出一条。说明为何不会全背景坍缩，也不能把推测写成已证实根因。
给下一轮可执行固定训练预算、只需的关键对照、21→BM→10→DEV64快否决门与退役条件。是否仍用30更新/180秒或需调整，请用本轮证据说明。不要把需要完整合同的数据真值缺口当成单TRAIN首门失败借口；正式>95%、FP/G<=5%、<=3mm、完整G500、分类/父子/方向/侧别/E16和90–110条工件三次独立无缓存<=10秒均保持。

请先给结论，再给最小实现/损失公式和预算，清楚区分证据与假设。最后独立一行：GPT6_PRO_RESULT_SJG_RUN272_20261009
'''
    (review/'packet.md').write_text(packet,encoding='utf-8')
    write(review/'local_evidence.json',dict(evaluation=evaluation,diagnosis=diagnosis,
        packet_sha256=digest(review/'packet.md'),sent=False,utc=utc()))
    print('DUPLICATE_DIAGNOSIS',diagnosis['individually_compatible_outputs'],'OF',len(rows),'ONE_PROFILE',flush=True)
    print('LOCAL_PRO_PACKET',review/'packet.md',flush=True)


if __name__=='__main__':main()
