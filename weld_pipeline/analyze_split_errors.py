"""Post-prediction error grouping using customer split metadata."""
import argparse,json
from pathlib import Path
from collections import defaultdict


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();root=Path(a.run)
    best=json.loads((root/'best_validation.json').read_text());epoch=best['epoch']
    records=[json.loads(s) for s in Path('weld_pipeline/reports/segment_labels/weld_supervision.jsonl').read_text().splitlines()]
    by_id={(r['group'],r['id']):r for r in records};report={}
    for name in best['validation']:
        score=json.loads((root/f'validation_epoch_{epoch:03d}'/(name+'_score.json')).read_text())
        groups=defaultdict(lambda:{'reference':0,'valid':0,'outer_endpoints_within_3mm':0})
        for item in score['per_truth']:
            row=by_id[name,item['truth_id']];flags=row['flags']
            categories=['all','multi_segment' if flags['SegmentCount']>1 else 'single_segment',
                'work_range' if flags['IsWorkRangeSplit (Any)'] else 'no_work_range',
                'dual_robot' if flags['IsSplitedForDualRobot (Any)'] else 'no_dual_robot',
                'vertical' if str(row['number']).lower().startswith('v') else 'horizontal']
            for key in categories:
                groups[key]['reference']+=1;groups[key]['valid']+=int(item['accepted'])
                groups[key]['outer_endpoints_within_3mm']+=int(max(item.get('start_mm',1e9),item.get('end_mm',1e9))<=3)
        report[name]=dict(groups)
    payload={'epoch':epoch,'scope':'Post-prediction reference audit, never inference input','groups':report}
    (root/'split_error_categories.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,groups in report.items():print(name,{k:groups[k] for k in ['all','work_range','multi_segment','vertical'] if k in groups},flush=True)


if __name__=='__main__':main()
