"""Raw-only checkpoint/export integration, then fixed development scoring."""
import builtins
import csv
import io
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from training.labels import sha256
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.predict_structured_height_candidate import predict
from weld_pipeline.verify_local_height_joint_raw_v3 import score


def same_cell(csv_value,excel_value):
    if csv_value=='' and excel_value in ('',None):
        return True
    if str(csv_value)==str(excel_value):
        return True
    try:
        return bool(np.isclose(float(csv_value),float(excel_value),rtol=0.,atol=1e-8))
    except (TypeError,ValueError):
        return False


def verify_one(name,split):
    root=Path('weld_pipeline/reports/structured_height_v5_integration_20261001')/name
    if root.exists():
        raise ValueError('Frozen v5 integration check exists')
    root.mkdir(parents=True);out=(root/'exported_output').resolve()
    piece=load_piece(split['piece_roots'][name],verify=True);pcd=Path(piece['record']['input']).resolve()
    checkpoint=Path('weld_pipeline/runs/contract_structured_height_candidate_v5/candidate.pt').resolve()
    frozen=sha256(checkpoint);input_digest=sha256(pcd)
    forbidden=[Path(path).resolve() for path in ('DataSet','weld_pipeline/data','weld_pipeline/reports')]
    original=(builtins.open,io.open,Path.open);reads=[];output_reads=[];controls=[]
    def guard(opener):
        def checked(path,mode='r',*args,**kwargs):
            if isinstance(path,(str,bytes,Path)) and ('r' in str(mode) or '+' in str(mode)):
                file=Path(path).resolve()
                if file==pcd:
                    reads.append(str(file))
                elif file.is_relative_to(out):
                    output_reads.append(str(file))
                elif any(file.is_relative_to(folder) for folder in forbidden) or file.suffix.lower() in ('.step','.stp','.xlsx','.xls','.cbf') or file.name.endswith('_welds.json'):
                    raise RuntimeError('Forbidden inference reference/cache: '+str(file))
            return opener(path,mode,*args,**kwargs)
        return checked
    builtins.open,io.open,Path.open=tuple(guard(opener) for opener in original)
    try:
        for opener in (builtins.open,io.open,Path.open):
            try:
                opener(Path('weld_pipeline/data/v5_negative_control.step'),'r')
            except RuntimeError:
                controls.append(True)
            else:
                raise ValueError('Isolation negative control failed')
        predictions,heights,inference=predict(pcd,checkpoint,out)
    finally:
        builtins.open,io.open,Path.open=original
    if not reads or len(controls)!=3:
        raise ValueError('Fresh PCD and three negative controls required')
    previous=Path('weld_pipeline/reports/continuous_raw_v2_20261001')/('raw_'+name)
    geometry_equal=predictions==json.loads((previous/'predictions.json').read_text('utf8'))
    height_equal=heights==json.loads((previous/'height_candidates.json').read_text('utf8'))
    with (out/'predicted_children.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);csv_rows=list(reader);fields=reader.fieldnames
    workbook=load_workbook(out/'predicted_children.xlsx',read_only=True,data_only=True)
    excel=list(workbook.active.values);workbook.close()
    excel_equal=(list(excel[0])==fields and len(excel)==len(csv_rows)+1 and all(
        same_cell(row[field],value) for row,values in zip(csv_rows,excel[1:]) for field,value in zip(fields,values)))
    profiles=json.loads((out/'height_profiles.json').read_text('utf8'))['children']
    plate=json.loads((out/'attribute_profiles.json').read_text('utf8'))['children']
    curves=json.loads((out/'child_curves.json').read_text('utf8'))
    keys={row['weld_id'] for row in csv_rows}
    keyed=(keys==set(profiles)==set(plate)==set(curves) and len(keys)==len(csv_rows))
    unknown_clear=all(row[field+'_value']=='' for row in csv_rows for field in ('start_height','end_height') if row[field+'_status']=='unknown')
    unit_normals=all(abs(np.linalg.norm(n)-1.)<=1e-3 for row in plate.values() if row.get('state')=='estimated_candidate'
                     for sample in row['samples'] for n in sample['material_outward_normals'])
    receipt=dict(name=name,role='FIT' if name in split['train'] else 'OBSERVED_DEVELOPMENT_NOT_BLIND',
        geometry_equal_to_fixed_raw_v2=geometry_equal,height_equal_to_fixed_raw_v2=height_equal,
        CSV_and_child_workbook_equal=excel_equal,profile_and_curve_child_keys_complete=keyed,
        unknown_height_values_empty=unknown_clear,estimated_normals_are_unit=unit_normals,
        negative_controls=len(controls),guarded_fresh_PCD_reads=len(reads),own_generated_output_reads=len(output_reads),
        input_unchanged=sha256(pcd)==input_digest,checkpoint_unchanged=sha256(checkpoint)==frozen,
        runtime_STEP_used=False,runtime_truth_used=False,runtime_preprocessing_cache_used=False,
        inference=inference,full_contract_pass=False)
    passed=all(receipt[key] for key in ('geometry_equal_to_fixed_raw_v2','height_equal_to_fixed_raw_v2','CSV_and_child_workbook_equal',
        'profile_and_curve_child_keys_complete','unknown_height_values_empty','estimated_normals_are_unit','input_unchanged','checkpoint_unchanged'))
    receipt['integration_passed']=passed
    (root/'integration_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf8')
    (root/'joint_development_score.json').write_text(json.dumps(score(piece,predictions,heights,name),indent=2),encoding='utf8')
    print('V5_INTEGRATION',name,passed,'CHILDREN',len(csv_rows),flush=True)
    if not passed:
        raise ValueError('v5 integration failed; preserve receipt and outputs')
    return receipt


def main():
    split=json.loads(Path('weld_pipeline/data/full_available_fit_v1/split.json').read_text('utf8'))
    receipts=[verify_one(name,split) for name in ('207-FR79M','232-TT1A')]
    path=Path('weld_pipeline/reports/structured_height_v5_integration_20261001/complete.json')
    path.write_text(json.dumps(dict(integration_passed=all(r['integration_passed'] for r in receipts),
        receipts=receipts,source_sha256=sha256(__file__),full_contract_pass=False),indent=2,ensure_ascii=False),encoding='utf8')


if __name__=='__main__':
    main()
