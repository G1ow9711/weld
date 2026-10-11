"""Audit all candidate sources before any new split; writes no model predictions."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.truth import load_truth,sample_weld


def main():
    root=Path.cwd()
    candidates=json.loads(Path('weld_pipeline/reports/candidate_inventory.json').read_text(encoding='utf-8'))
    out=[]
    for item in candidates:
        folder=root/item['folder'];pcds=list(folder.glob('*AllFaces.pcd'))
        if not pcds:pcds=list(folder.glob('*.pcd'))
        record={'name':folder.name,'folder':item['folder'],'declared_welds':item['welds']}
        try:
            welds,missing=load_truth(folder)
            points=read_xyz_pcd(pcds[0])*1000
            tree=cKDTree(points)
            audits=[]
            for w in welds:
                d=tree.query(sample_weld(w),workers=2)[0]
                r=w['metadata']
                audits.append({'id':w['id'],'name':w['name'],'type':w['type'],'source':w['source'],
                    'segment_count':len(w['segments']),'kind':[s['kind'] for s in w['segments']],
                    'max_cloud_distance_mm':float(d.max()),'p95_cloud_distance_mm':float(np.quantile(d,.95)),
                    'coverage_6mm':float(np.mean(d<=6)),
                    'work_range_split':bool(r['IsWorkRangeSplit (Any)']),
                    'front_range_split':bool(r['WorkRangeFrontSplit']),'back_range_split':bool(r['WorkRangeBackSplit']),
                    'dual_robot_split':bool(r['IsSplitedForDualRobot (Any)']),
                    'source_start_mm':w['segments'][0]['points'][0],
                    'source_end_mm':w['segments'][-1]['points'][-1]})
            hashes={str(p.relative_to(root)):sha256(p) for p in [pcds[0],next(folder.glob('*.xlsx')),*folder.glob('*welds.json')]}
            bad=[a['id'] for a in audits if a['max_cloud_distance_mm']>6]
            record.update(input=str(pcds[0].relative_to(root)),point_count=len(points),bbox_mm=[points.min(0).tolist(),points.max(0).tolist()],
                source_hashes=hashes,loaded_welds=len(welds),missing_curves=missing,bad_coverage_ids=bad,
                complete_truth=not missing,coverage_ok=not bad,weld_audit=audits,
                status='audited_candidate' if not bad and not missing else 'needs_label_or_pairing_review')
        except Exception as e:
            record.update(status='error',error=repr(e))
        out.append(record)
        Path('weld_pipeline/reports/source_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
        print(record['name'],record['status'],'points',record.get('point_count'),'welds',record.get('loaded_welds'),
              'bad',len(record.get('bad_coverage_ids',[])),'missing',len(record.get('missing_curves',[])),flush=True)
    summary={'candidate_groups':len(out),'complete_and_covered':sum(x.get('complete_truth',False) and x.get('coverage_ok',False) for x in out),
        'read_points':sum(x.get('point_count',0) for x in out),'loaded_reference_welds':sum(x.get('loaded_welds',0) for x in out),
        'work_range_split_welds':sum(a['work_range_split'] for x in out for a in x.get('weld_audit',[])),
        'dual_robot_split_welds':sum(a['dual_robot_split'] for x in out for a in x.get('weld_audit',[])),
        'note':'Coverage audit only; no predictions, no acceptance result, no frozen new split.'}
    Path('weld_pipeline/reports/source_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary)


if __name__=='__main__':main()
