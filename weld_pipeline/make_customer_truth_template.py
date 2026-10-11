"""Review template built only from existing labels; no inferred truth added."""
import csv
import json
from pathlib import Path
from weld_pipeline.contract_export import ATTRIBUTES


def main():
    root=Path('weld_pipeline/data/contract_child_v1')
    destination=Path('合同/客户补充真值_子段样表_20260930.csv')
    fields=['workpiece_id','parent_id','segment_index','geometry_type','weld_type',
        'start_x_mm','start_y_mm','start_z_mm','end_x_mm','end_y_mm','end_z_mm',
        'geometry_source','child_mapping_review','weld_side','board_1_id','board_2_id',
        'normal_sign_convention','segmentation_reason','truth_reviewer','truth_review_date']
    fields += ['normal_'+str(side)+'_'+axis for side in [1,2] for axis in ['x','y','z']]
    fields += [a+'_'+suffix for a in ATTRIBUTES for suffix in ['value','state','review_status']]
    count=0
    with destination.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for path in sorted(root.glob('*_children.json')):
            for child in json.loads(path.read_text('utf8')):
                row={'workpiece_id':child['workpiece_id'],'parent_id':child['parent_id'],
                    'segment_index':child['segment_index'],'geometry_type':child['kind'],'weld_type':child['type'],
                    **dict(zip(['start_x_mm','start_y_mm','start_z_mm','end_x_mm','end_y_mm','end_z_mm'],
                        [*child['points'][0],*child['points'][-1]])),
                    'geometry_source':'existing_customer_geometry','child_mapping_review':'pending'}
                for a in ATTRIBUTES:
                    attr=child['attributes'][a]
                    row[a+'_value']=attr['value']
                    row[a+'_state']=attr['state']
                    row[a+'_review_status']='legacy_label_requires_new_contract_review' if attr['truth_available'] else attr['reason']
                writer.writerow(row);count+=1
    note='''客户补充真值样表说明（2026年9月30日）

样表含25件已有原始工件、1605个几何子段。只预填现有客户几何和能正确对应的旧属性；不是新增审核真值。segment_index从0开始，与现有几何JSON段顺序一致。

请客户审核：
1. child_mapping_review：现有每个几何段是否就是合同的最小工艺子段；GR24A1的120mm父线分为40.2、40.2、39.6mm，需确认三段计数及断点原因。
2. weld_side、board_1_id、board_2_id、normal_sign_convention、segmentation_reason。板1/2须固定，并分别对应厚度和法向，不能交换。
3. thickness_1/2_value填毫米；normal_1/2_x/y/z填带方向单位向量。法向正负按材料侧约定填写，不用绝对点积消除方向差。
4. 内部子段首尾板高、孔高、挡板和外边距必须补测；不要复制父线两端值。父线Any字段不能直接视作每个子段的审核属性。
5. 挡板旧−1合并“无挡板”和“超过50mm”，须区分no_object与out_of_range；外边距旧51对应greater_than_50_mm，不按51mm回归；孔高0对应no_hole。
6. 各属性state使用value、no_object、no_hole、out_of_range、greater_than_50_mm、unknown或按已冻结业务规则not_applicable；unknown/missing_truth不能作为验收合格。不适用不能由程序自行决定。
7. 真值状态review_status填写reviewed、pending、disputed或经冻结规则确认的not_applicable；补审核人和日期。

新增数据仍须按独立工件分组划分800/100/200，并补真实单段立焊、独立曲线及难例。最终200件测试标签由客户封存，不回流训练。样表属于当前开发数据，不是最终测试集。
'''
    destination.with_suffix('.md').write_text(note,encoding='utf8')
    print('CUSTOMER_REVIEW_TEMPLATE',destination,'rows',count,flush=True)


if __name__=='__main__':main()
