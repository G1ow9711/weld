"""Inventory split supervision without inferring labels absent from the sources."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from weld_pipeline.truth import excel_rows,load_truth

FIELDS=['IsWorkRangeSplit (Any)','WorkRangeFrontSplit','WorkRangeBackSplit',
        'IsSplitedForDualRobot (Any)','HasMid (Any)','SegmentCount','HasEave (Any)']


def main():
    out=Path('weld_pipeline/reports/segment_labels');out.mkdir(parents=True,exist_ok=True)
    candidates=json.loads(Path('weld_pipeline/reports/candidate_inventory.json').read_text(encoding='utf-8'))
    records=[];summaries=[];mismatches=[]
    for item in candidates:
        folder=Path(item['folder']);rows=excel_rows(next(folder.glob('*.xlsx')))
        curves,_=load_truth(folder);curve_by_id={w['id']:w for w in curves}
        paths=list(folder.glob('*welds.json'))
        cur_summary={'group':folder.name,'scope':'main_dataset' if item['folder'].startswith('DataSet') else 'supplemental',
            'welds':len(rows),'json_available':bool(paths),'counts':{k:0 for k in FIELDS if k!='SegmentCount'},
            'segment_counts':dict(Counter(str(r['SegmentCount']) for r in rows))}
        for r in rows:
            wid=str(r['ObjectId']);w=curve_by_id.get(wid)
            flags={k:r[k] for k in FIELDS}
            for k in cur_summary['counts']:cur_summary['counts'][k]+=int(bool(r[k]))
            a=np.array([r['Source StartMatrix.Translation.'+d] for d in 'XYZ'],float)
            b=np.array([r['Source EndMatrix.Translation.'+d] for d in 'XYZ'],float)
            row={'group':folder.name,'scope':cur_summary['scope'],'id':wid,'number':r['WeldNumber (f/v + Name number)'],
                'name':r['Name'],'flags':flags,'start_world_mm':a.tolist(),'end_world_mm':b.tolist(),
                'curve_source':w['source'] if w else 'missing_internal_curve',
                'segment_geometry_available':bool(w),'segments':[]}
            if w:
                p=np.asarray(w['segments'][0]['points'][0]);q=np.asarray(w['segments'][-1]['points'][-1])
                direct=max(np.linalg.norm(a-p),np.linalg.norm(b-q));reverse=max(np.linalg.norm(a-q),np.linalg.norm(b-p))
                row['excel_curve_endpoint_max_mm']=float(min(direct,reverse));row['curve_reversed_to_excel']=bool(reverse<direct)
                row['segment_count_matches']=len(w['segments'])==r['SegmentCount']
                row['arc_flag_matches']=any(s['kind']=='arc' for s in w['segments'])==bool(r['HasMid (Any)'])
                if not row['segment_count_matches'] or not row['arc_flag_matches'] or min(direct,reverse)>.02:
                    mismatches.append({'group':folder.name,'id':wid,'endpoint_error_mm':row['excel_curve_endpoint_max_mm'],
                        'count_ok':row['segment_count_matches'],'arc_ok':row['arc_flag_matches']})
                segments=w['segments']
                if reverse<direct:
                    segments=[{**s,'points':s['points'][::-1]} for s in segments[::-1]]
                for i,s in enumerate(segments):
                    points=s['points'];endpoints=np.asarray(points)[[0,-1]]
                    gap=None if i==0 else float(np.linalg.norm(endpoints[0]-np.asarray(segments[i-1]['points'][-1])))
                    row['segments'].append({'index':i,'kind':s['kind'],'points_world_mm':points,
                        'start_world_mm':points[0],'end_world_mm':points[-1],'length_mm':s['length_mm'],
                        'gap_from_previous_mm':gap,
                        'start_work_range_split':bool(r['WorkRangeFrontSplit']) if i==0 else None,
                        'end_work_range_split':bool(r['WorkRangeBackSplit']) if i==len(segments)-1 else None,
                        'internal_split_cause':None})
            records.append(row)
        summaries.append(cur_summary)
    aggregate={}
    for scope in ['main_dataset','supplemental','all']:
        selected=[r for r in records if scope=='all' or r['scope']==scope]
        aggregate[scope]={'groups':len(set(r['group'] for r in selected)),'welds':len(selected),
            'flags':{k:sum(bool(r['flags'][k]) for r in selected) for k in FIELDS if k!='SegmentCount'},
            'multi_segment_welds':sum(r['flags']['SegmentCount']>1 for r in selected),
            'total_segments_excel':sum(int(r['flags']['SegmentCount']) for r in selected),
            'geometry_missing_welds':sum(not r['segment_geometry_available'] for r in selected)}
    report={'summary':aggregate,'mismatches':mismatches,'groups':summaries,
        'supervision_policy':{
            'weld_flags':'Any flags are weld-level aggregate properties; not broadcast as the cause of every internal segment break.',
            'global_endpoints':'Front/back flags attach to Excel source start/end, with curve direction checked.',
            'internal_boundaries':'Use actual JSON segment endpoints. Their cause stays unknown unless explicitly marked.',
            'inference':'Reference flags/curves may supervise training and score predictions, never serve as inference inputs.'}}
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'weld_supervision.jsonl').open('w',encoding='utf-8') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'summary':aggregate,'mismatches':mismatches},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
