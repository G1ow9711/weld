"""Actual mixed-success batch, isolated raw input and CSV/workbook/error checks."""
import builtins
import csv
import io
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from training.labels import sha256
from weld_pipeline.structured_candidate_batch import run_batch
from weld_pipeline.verify_structured_height_candidate_v3 import same_cell

OUT=Path('weld_pipeline/reports/structured_candidate_batch_20261001')
CHECKPOINT=Path('weld_pipeline/runs/contract_structured_height_candidate_v5r1/candidate.pt')


def main():
    if OUT.exists():raise ValueError('Preserve actual batch verification')
    OUT.mkdir(parents=True);fixtures=OUT/'fixtures';fixtures.mkdir()
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    inputs={}
    for name in ('207-FR79M','224-BX1A'):
        manifest=json.loads((Path(split['piece_roots'][name])/'manifest.json').read_text('utf8'))
        inputs[name]=Path(manifest['record']['input']).resolve()
    bad=fixtures/'damaged_count.pcd'
    bad.write_text('FIELDS x y z\nPOINTS 3\nDATA ascii\n0 0 0\n1 1 1\n',encoding='ascii')
    nan=fixtures/'nonfinite.pcd'
    nan.write_text('FIELDS x y z\nPOINTS 3\nDATA ascii\n0 0 0\n1 1 1\nnan 0 1\n',encoding='ascii')
    u,v=np.meshgrid(np.arange(-30.,30.1,3.),np.arange(-30.,30.1,3.))
    xyz=np.c_[u.ravel(),v.ravel(),(.3*u+.2*v).ravel()]/1000.
    no_contact=fixtures/'synthetic_no_contact.pcd'
    with no_contact.open('w',encoding='ascii') as stream:
        stream.write('FIELDS x y z\nPOINTS '+str(len(xyz))+'\nDATA ascii\n')
        np.savetxt(stream,xyz,fmt='%.10f')
    missing=fixtures/'not_provided.pcd'
    items=[dict(workpiece_id='损坏点数',pcd=str(bad.resolve())),
           dict(workpiece_id='FR79M正常件',pcd=str(inputs['207-FR79M'])),
           dict(workpiece_id='缺失文件',pcd=str(missing.resolve())),
           dict(workpiece_id='非有限坐标',pcd=str(nan.resolve())),
           dict(workpiece_id='重复工件',pcd=str(inputs['207-FR79M'])),
           dict(workpiece_id='重复工件',pcd=str(inputs['224-BX1A'])),
           dict(workpiece_id='BX1正常件',pcd=str(inputs['224-BX1A'])),
           dict(workpiece_id='无接触平面合成例',pcd=str(no_contact.resolve())),
           dict(workpiece_id=[],pcd=str(no_contact.resolve()))]
    original_hashes={str(p):sha256(p) for p in [*inputs.values(),bad,nan,no_contact]}
    model_hash=sha256(CHECKPOINT);allowed={Path(i['pcd']).resolve() for i in items}
    runtime_root=(OUT/'runtime_output').resolve()
    forbidden=[Path(p).resolve() for p in ('DataSet','weld_pipeline/data','weld_pipeline/reports')]
    openers=(builtins.open,io.open,Path.open);reads=[];controls=[]
    def guard(opener):
        def checked(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,bytes,Path)) and ('r' in str(mode) or '+' in str(mode)):
                file=Path(path).resolve()
                if file in allowed:reads.append(str(file))
                elif file.is_relative_to(runtime_root):pass
                elif any(file.is_relative_to(r) for r in forbidden) or file.suffix.lower() in ('.step','.stp','.xlsx','.xls') or file.name.endswith('_welds.json'):
                    raise RuntimeError('Forbidden batch runtime reference/cache: '+str(file))
            return opener(path,mode,*args,**kwargs)
        return checked
    builtins.open,io.open,Path.open=tuple(guard(opener) for opener in openers)
    try:
        for opener in (builtins.open,io.open,Path.open):
            try:opener(Path('weld_pipeline/data/batch_negative_control.step'),'r')
            except RuntimeError:controls.append(True)
            else:raise ValueError('Batch reference/cache negative control failed')
        summary=run_batch(items,CHECKPOINT,runtime_root)
    finally:builtins.open,io.open,Path.open=openers
    codes={r['index']:r['reason_code'] for r in summary['items']}
    expected={0:'INPUT_INVALID_PCD',2:'INPUT_NOT_FOUND',3:'INPUT_INVALID_PCD',4:'DUPLICATE_WORKPIECE_ID',
              5:'DUPLICATE_WORKPIECE_ID',8:'BATCH_ITEM_INVALID'}
    errors_correct=all(codes[i]==code for i,code in expected.items())
    continued=all(summary['items'][i]['status'].startswith('SUCCESS') for i in (1,6))
    no_weld=summary['items'][7]['status']=='SUCCESS_NO_WELDS' and summary['items'][7]['child_rows']==0
    exports=[]
    for receipt in summary['items']:
        if not receipt['status'].startswith('SUCCESS'):continue
        folder=Path(receipt['output_directory'])
        with (folder/'predicted_children.csv').open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream);rows=list(reader);fields=reader.fieldnames
        book=load_workbook(folder/'predicted_children.xlsx',read_only=True,data_only=True)
        excel=list(book.active.values);book.close()
        equal=bool(excel and list(excel[0])==fields and len(excel)==len(rows)+1 and all(
            all(same_cell(row[field],value) for field,value in zip(fields,line)) for row,line in zip(rows,excel[1:])))
        identities=all(row['workpiece_id']==receipt['workpiece_id'] for row in rows)
        unknowns=all(row[attr+'_value']=='' for row in rows for attr in ('front_wall','back_wall','start_base_edge','end_base_edge'))
        exports.append(dict(index=receipt['index'],rows=len(rows),CSV_workbook_equal=equal,caller_workpiece_id_correct=identities,
                            unknown_values_empty=unknowns,empty_table_has_header=bool(fields)))
    unchanged=sha256(CHECKPOINT)==model_hash and all(sha256(p)==digest for p,digest in original_hashes.items())
    passed=errors_correct and continued and no_weld and unchanged and len(controls)==3 and all(
        all(r[k] for k in ('CSV_workbook_equal','caller_workpiece_id_correct','unknown_values_empty','empty_table_has_header')) for r in exports)
    result=dict(actual_items=9,successful=summary['successful'],failed=summary['failed'],
        errors_correct=errors_correct,success_after_bad_inputs=continued,synthetic_no_contact_empty_success=no_weld,
        exports=exports,raw_PCD_reads=len(reads),negative_controls=len(controls),input_and_checkpoint_unchanged=unchanged,
        all_checks_passed=passed,runtime_STEP_used=False,runtime_truth_used=False,runtime_preprocessing_cache_used=False,
        synthetic_no_contact_is_not_a_formal_no_weld_acceptance_piece=True,
        source_sha256={p:sha256(p) for p in (__file__,'weld_pipeline/structured_candidate_batch.py')},
        model_sha256=model_hash,batch_summary_sha256=sha256(runtime_root/'batch_summary.json'),full_contract_pass=False)
    (OUT/'complete.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('STRUCTURED_BATCH_CHECKS',json.dumps(result,ensure_ascii=False),flush=True)
    if not passed:raise ValueError('Actual batch verification failed; preserve receipts')


if __name__=='__main__':main()
