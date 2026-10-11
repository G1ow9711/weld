"""CSV, plain workbook and CloudCompare layer from predictions alone."""
import csv,json,shutil,zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from plyfile import PlyData,PlyElement

XLSX_TEMPLATE=Path(r'C:\Users\Administrator\.codex\skills\minimax-xlsx\templates\minimal_xlsx')
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='http://schemas.openxmlformats.org/officeDocument/2006/relationships'


def curve_points(weld,step=2.):
    out=[]
    for i,s in enumerate(weld['segments'],1):
        p=np.asarray(s['points'],dtype=float)
        if not np.isfinite(p).all() or p.ndim!=2 or p.shape[1]!=3:raise ValueError('Invalid predicted coordinates')
        if s['kind']=='line':p=np.linspace(p[0],p[-1],max(2,int(np.ceil(np.linalg.norm(p[-1]-p[0])/step))+1))
        for j,q in enumerate(p):out.append((i,j,s['kind'],q,float(np.linalg.norm(q-p[j-1])) if j else 0.))
    return out


def save_csv(welds,path):
    ids=[w['id'] for w in welds]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate prediction identifiers')
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['weld_id','type','start_x_mm','start_y_mm','start_z_mm','end_x_mm','end_y_mm','end_z_mm','length_mm','segment_count','ordered_segments_json'])
        for w in welds:
            length=sum(float(np.linalg.norm(np.diff(np.array(s['points']),axis=0),axis=1).sum()) for s in w['segments'])
            writer.writerow([w['id'],w['type'],*w['segments'][0]['points'][0],*w['segments'][-1]['points'][-1],length,len(w['segments']),json.dumps(w['segments'],separators=(',',':'))])


def cell(address,value,style=0,formula=None):
    if formula:return f'<c r="{address}" s="{style}"><f>{escape(formula)}</f><v>{value:.12g}</v></c>'
    if isinstance(value,(float,int,np.floating,np.integer)):return f'<c r="{address}" s="{style}"><v>{float(value):.12g}</v></c>'
    return f'<c r="{address}" s="{style}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def sheet_xml(rows,widths):
    last=chr(64+len(widths))
    cols=''.join(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>' for i,w in enumerate(widths,1))
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="{NS}"><dimension ref="A1:{last}{len(rows)}"/><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetFormatPr defaultRowHeight="18"/><cols>{cols}</cols><sheetData>'+''.join(f'<row r="{i}">{r}</row>' for i,r in enumerate(rows,1))+f'</sheetData><autoFilter ref="A1:{last}{len(rows)}"/></worksheet>'


def save_workbook(welds,path,work):
    # Begin with the skill's OOXML skeleton. User's explicit black/white style
    # overrides the skill's financial font colors.
    shutil.copytree(XLSX_TEMPLATE,work,dirs_exist_ok=True)
    headers=['焊缝编号','类型','起点X/mm','起点Y/mm','起点Z/mm','终点X/mm','终点Y/mm','终点Z/mm','长度/mm','段数']
    pheaders=['焊缝编号','段编号','点序号','曲线类型','X/mm','Y/mm','Z/mm','步长/mm']
    rows=[''.join(cell(f'{chr(65+i)}1',h,1) for i,h in enumerate(headers))]
    points=[''.join(cell(f'{chr(65+i)}1',h,1) for i,h in enumerate(pheaders))]
    for wi,w in enumerate(welds,2):
        begin=len(points)+1;length=0.
        for si,pi,kind,p,distance in curve_points(w):
            r=len(points)+1;values=[w['id'],si,pi+1,kind,*p]
            row=''.join(cell(f'{chr(65+i)}{r}',v,2 if i>=4 else 0) for i,v in enumerate(values))
            formula=f'SQRT((E{r}-E{r-1})^2+(F{r}-F{r-1})^2+(G{r}-G{r-1})^2)' if pi else '0'
            row+=cell(f'H{r}',distance,2,formula);points.append(row);length+=distance
        end=len(points)
        values=[w['id'],'平焊' if w['type']=='horizontal' else '立焊',*w['segments'][0]['points'][0],*w['segments'][-1]['points'][-1]]
        row=''.join(cell(f'{chr(65+i)}{wi}',v,2 if i>=2 else 0) for i,v in enumerate(values))
        row+=cell(f'I{wi}',length,2,f"SUM('曲线点序列'!H{begin}:H{end})")
        row+=cell(f'J{wi}',len(w['segments']),0,f"MAX('曲线点序列'!B{begin}:B{end})")
        rows.append(row)
    (work/'xl/worksheets/sheet1.xml').write_text(sheet_xml(rows,[21,10,*([16]*7),10]),encoding='utf-8')
    (work/'xl/worksheets/sheet2.xml').write_text(sheet_xml(points,[21,10,10,12,16,16,16,16]),encoding='utf-8')
    (work/'xl/workbook.xml').write_text(f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{NS}" xmlns:r="{REL}"><sheets><sheet name="焊缝汇总" sheetId="1" r:id="rId1"/><sheet name="曲线点序列" sheetId="2" r:id="rId4"/></sheets><calcPr calcId="191029" fullCalcOnLoad="1"/></workbook>',encoding='utf-8')
    (work/'xl/_rels/workbook.xml.rels').write_text(f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="{REL}/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="{REL}/styles" Target="styles.xml"/><Relationship Id="rId3" Type="{REL}/sharedStrings" Target="sharedStrings.xml"/><Relationship Id="rId4" Type="{REL}/worksheet" Target="worksheets/sheet2.xml"/></Relationships>',encoding='utf-8')
    ct=work/'[Content_Types].xml';content=ct.read_text(encoding='utf-8');content=content.replace('</Types>','<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>');ct.write_text(content,encoding='utf-8')
    styles=f'''<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="{NS}"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts><fonts count="2"><font><sz val="11"/><color rgb="FF000000"/><name val="Microsoft YaHei"/></font><font><b/><sz val="11"/><color rgb="FF000000"/><name val="Microsoft YaHei"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border><left style="thin"><color rgb="FF000000"/></left><right style="thin"><color rgb="FF000000"/></right><top style="thin"><color rgb="FF000000"/></top><bottom style="thin"><color rgb="FF000000"/></bottom><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    (work/'xl/styles.xml').write_text(styles,encoding='utf-8')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for file in work.rglob('*'):
            if file.is_file():z.write(file,file.relative_to(work).as_posix())


def save_ply(welds,path):
    values=[(wi,si,p) for wi,w in enumerate(welds,1) for si,_,_,p,_ in curve_points(w)]
    array=np.empty(len(values),dtype=[('x','f8'),('y','f8'),('z','f8'),('red','u1'),('green','u1'),('blue','u1'),('weld_id','i4'),('segment_id','i4')])
    for i,(wi,si,p) in enumerate(values):array[i]=(*np.asarray(p)/1000,255,0,0,wi,si)
    with Path(path).open('wb') as f:PlyData([PlyElement.describe(array,'vertex')],text=False,comments=['Prediction only; coordinates in original PCD numerical units (m); weld/segment IDs']).write(f)


def save_all(welds,outdir):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    save_csv(welds,outdir/'predicted_welds.csv')
    save_workbook(welds,outdir/'预测焊缝.xlsx',outdir/'xlsx_source')
    save_ply(welds,outdir/'预测焊缝.ply')
