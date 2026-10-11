"""Read original DataSet workbook flags and native exported segment kinds."""
import argparse
import hashlib
import json
import traceback
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import openpyxl

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def utc():return datetime.now(timezone.utc).isoformat()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);data=ROOT/'DataSet'
    write(dest/'registered_protocol.json',dict(stage='ORIGINAL_DATASET_CURVE_LABEL_INVENTORY',directory=str(data),
        optimizer_updates=0,no_full65_model_evaluation=True,no_training_split_change=True,
        original_files_read_only=True,source_sha256=digest(__file__),start_utc=utc()))
    try:
        books=sorted(data.rglob('*.xlsx'));exports=sorted(data.rglob('*_weld_welds.json'));book_rows=[];export_rows=[]
        for number,path in enumerate(books,1):
            workbook=openpyxl.load_workbook(path,read_only=True,data_only=False);sheets=[]
            for sheet in workbook:
                rows=iter(sheet.iter_rows(values_only=True));header=next(rows,());fields=[str(x or '') for x in header]
                flags=[i for i,v in enumerate(fields) if 'HasMid' in v]
                mids=[next((i for i,v in enumerate(fields) if 'MidMatrix.Translation.'+axis in v),None) for axis in 'XYZ']
                examples=[];flag_values=Counter();count=mid_count=flag_count=0
                for ri,row in enumerate(rows,2):
                    if not any(v is not None and v!='' for v in row):continue
                    count+=1;has_mid=all(i is not None and i<len(row) and row[i] is not None and row[i]!='' for i in mids)
                    positive=False
                    for i in flags:
                        value=row[i] if i<len(row) else None;flag_values[repr(value)]+=1
                        positive|=value is True or (isinstance(value,str) and value.strip().lower() in ('true','yes','是','1')) or (type(value) in (int,float) and value==1)
                    flag_count+=int(positive);mid_count+=int(has_mid)
                    if positive or has_mid:examples.append(dict(row=ri,has_mid_flag=bool(positive),mid_coordinates_present=bool(has_mid)))
                sheets.append(dict(sheet=sheet.title,rows=count,has_mid_columns=flags,flag_values=dict(flag_values),
                    positive_has_mid_rows=flag_count,complete_mid_coordinate_rows=mid_count,curve_rows=examples))
            workbook.close();book_rows.append(dict(file=str(path.relative_to(ROOT)),sha256=digest(path),sheets=sheets))
            if number%10==0 or number==len(books):print('ORIGINAL_WORKBOOK_AUDIT',number,'/',len(books),flush=True)
        for number,path in enumerate(exports,1):
            payload=read(path);kinds=Counter();parents=[]
            for weld in payload.get('welds',[]):
                segments=weld.get('segments',[])
                for segment in segments:kinds[str(segment.get('kind','MISSING'))]+=1
                curves=[dict(index=s.get('index'),kind=s.get('kind'),points=len(s.get('points',[]))) for s in segments if s.get('kind')!='line']
                if curves:parents.append(dict(id=weld.get('id'),segments=curves))
            export_rows.append(dict(file=str(path.relative_to(ROOT)),sha256=digest(path),parents=len(payload.get('welds',[])),
                segment_kinds=dict(kinds),nonline_parents=parents,coordinate_frame=payload.get('coordinateFrame')))
            if number%10==0 or number==len(exports):print('ORIGINAL_NATIVE_CURVE_AUDIT',number,'/',len(exports),flush=True)
        inventory=read(OUT/'inventory.json');processed=[]
        for row in inventory:
            path=ROOT/row['target']['target_path'];assert digest(path)==row['target']['target_sha256'];target=read(path)
            kinds=Counter(c['original_kind'] for p in target['parents'] for c in p['children'])
            processed.append(dict(source=row['index'],name=row['name'],original_G=target['original_children'],kinds=dict(kinds),target_sha256=digest(path)))
        write(dest/'workbooks.json',book_rows);write(dest/'native_exports.json',export_rows);write(dest/'processed_target_kinds.json',processed)
        counts=dict(workbooks=len(books),native_exports=len(exports),processed_sources=len(processed),
            workbook_positive_has_mid_rows=sum(s['positive_has_mid_rows'] for b in book_rows for s in b['sheets']),
            workbook_complete_mid_rows=sum(s['complete_mid_coordinate_rows'] for b in book_rows for s in b['sheets']),
            native_non_line_segments=sum(v for r in export_rows for k,v in r['segment_kinds'].items() if k!='line'),
            processed_arc_sources=[r['source'] for r in processed if r['kinds'].get('arc',0)])
        write(dest/'receipt.json',dict(status='ORIGINAL_CURVE_LABEL_INVENTORY_COMPLETE',counts=counts,
            files={name:digest(dest/name) for name in ['workbooks.json','native_exports.json','processed_target_kinds.json']},
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
        print('CURVE_LABEL_INVENTORY',counts,flush=True)
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
