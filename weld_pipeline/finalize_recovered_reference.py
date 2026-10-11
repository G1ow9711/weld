"""Write a derived reference package after independent frame audit succeeds."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from training.labels import sha256
from weld_pipeline.recover_single_weld_reference import read_reference_ply,fit_primitive
from weld_pipeline.truth import excel_rows,load_truth


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',required=True);ap.add_argument('--auditdir',required=True);ap.add_argument('--outdir',required=True)
    args=ap.parse_args();folder=Path(args.folder);auditdir=Path(args.auditdir);out=Path(args.outdir)
    audits=[auditdir/'recovery_audit.json',auditdir/'surface_frame_audit_v2.json']
    records=[json.loads(p.read_text(encoding='utf-8')) for p in audits]
    for record in records:
        for source,wanted in record['source_hashes'].items():
            if sha256(source)!=wanted:raise ValueError('Audit source has changed: '+source)
    surface=records[1]
    if surface['status']!='FRAME_SUPPORTED':raise ValueError('Frame has not passed')
    xlsx=next(folder.glob('*.xlsx'));rows=excel_rows(xlsx)
    if len(rows)!=1:raise ValueError('Requires a single identified weld')
    row=rows[0];ply=next(folder.glob('*welds.ply'));points=read_reference_ply(ply)
    cuts=np.flatnonzero(np.linalg.norm(np.diff(points,axis=0),axis=1)<1e-8)+1
    segments=[dict(fit_primitive(p),points=p.tolist()) for p in np.split(points,cuts)]
    if len(segments)!=row['SegmentCount'] or abs(sum(s['lengthMm'] for s in segments)-row['Length'])>.01:
        raise ValueError('Reference segment/length mismatch')
    provenance={'origin':'Ordered customer reference PLY and Excel, not algorithm predictions.',
                'status':'DERIVED_REFERENCE_FRAME_CHECKED','usage':'Additional development validation; not a blinded customer test.',
                'source_hashes':{str(p):sha256(p) for p in [*audits,xlsx,ply,Path(__file__)]}}
    reference={'unit':'millimeter','matrixConvention':'row_major_column_vector','coordinateFrame':'reference_workpiece',
               'exportToWorld':surface['checks'][surface['selected_transform']]['exportToWorld'],'provenance':provenance,
               'welds':[{'id':row['ObjectId'],'name':row['Name'],'segments':segments}]}
    out.mkdir(parents=True,exist_ok=True);dest=out/(folder.name+'_recovered_welds.json')
    if dest.exists():raise ValueError('Existing recovered reference; preserve immutable package')
    dest.write_text(json.dumps(reference,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(xlsx,out/xlsx.name)
    truth,missing=load_truth(out)
    if len(truth)!=1 or missing:raise ValueError('Recovered reference loader failed')
    world_end=np.array([truth[0]['segments'][0]['points'][0],truth[0]['segments'][-1]['points'][-1]])
    expected=np.array([[row['Source '+side+'Matrix.Translation.'+c] for c in 'XYZ'] for side in ['Start','End']])
    if np.linalg.norm(world_end-expected,axis=1).max()>1e-5:raise ValueError('World endpoint check failed')
    (out/'manifest.json').write_text(json.dumps({**provenance,'outputs':{str(p):sha256(p) for p in [dest,out/xlsx.name]},
        'segments':len(segments),'kinds':[s['kind'] for s in segments],'world_endpoints_error_mm':np.linalg.norm(world_end-expected,axis=1).tolist()},ensure_ascii=False,indent=2),encoding='utf-8')
    print('RECOVERED',out,'one weld, four preserved segments',flush=True)


if __name__=='__main__':main()
