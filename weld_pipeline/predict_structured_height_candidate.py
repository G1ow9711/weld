"""One-checkpoint raw PCD development candidate with explicit height states."""
import argparse
import csv
import json
import time
from pathlib import Path

from openpyxl import Workbook
import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.contract_export import save_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.continuous_contextual_height_model import estimate_children as estimate_height
from weld_pipeline.continuous_contact_sampling import ContinuousJointBases
from weld_pipeline.export import save_all
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates
from weld_pipeline.fresh_point_context import point_structure
from weld_pipeline.full_fit_hybrid import infer_from_sequences
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.predict_hybrid_with_attributes import export_attributes
from weld_pipeline.structured_plate_attributes import estimate_children as estimate_plate

MODEL_NAME='full_fit_structured_height_candidate_v5'


def export_height(out,heights):
    indexed={(row['parent_id'],row['child_index']):row for row in heights}
    profile={row['parent_id']+'_'+str(row['child_index']+1):row for row in heights}
    (out/'height_profiles.json').write_text(json.dumps(dict(children=profile,
        reference='endpoint values use point-derived contact/base hypotheses and a shared learned residual',
        customer_height_definition='实际接触板及局部底板基准',reviewed_customer_truth=False,
        along_line_complete_field_acceptance=False,full_contract_pass=False),indent=2,ensure_ascii=False),encoding='utf8')
    path=out/'predicted_children.csv'
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);rows=list(reader);fields=list(reader.fieldnames)
    fields.append('height_profile_reference');candidates=0
    for row in rows:
        record=indexed.get((row['parent_id'],int(row['segment_index'])))
        row['height_profile_reference']='height_profiles.json#'+row['weld_id']
        for field in ('start_height','end_height'):
            value=record['fields'][field] if record else dict(state='unknown',value=None,reason='HEIGHT_CONTACT_NOT_ASSOCIATED')
            row[field+'_value']=value.get('value')
            row[field+'_status']=value.get('state','unknown')
            row[field+'_source']=value.get('source')
            row[field+'_confidence']=value.get('confidence')
            row[field+'_reason']=value.get('reason')
            candidates+=int(value.get('value') is not None)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    # This workbook mirrors the child CSV, including unknown and candidate states.
    book=Workbook();sheet=book.active;sheet.title='子段结果';sheet.append(fields)
    for row in rows:
        sheet.append([row.get(field) for field in fields])
    sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
    book.save(out/'predicted_children.xlsx');book.close()
    return candidates,fields


def predict(pcd,checkpoint,outdir):
    torch.set_num_threads(2);started=time.perf_counter()
    state=torch.load(checkpoint,map_location='cpu')
    if state['model_name']!=MODEL_NAME:
        raise ValueError('Structured-height candidate checkpoint required')
    verify_cached_inference_sources(state,checkpoint);verify_cached_inference_sources(state['horizontal'],checkpoint)
    for source,digest in state['height_heads']['config']['sources'].items():
        if sha256(source)!=digest:
            raise ValueError('Frozen height inference source changed: '+source)
    out=Path(outdir)
    if out.exists():
        raise ValueError('Use a new output directory; existing predictions preserved')
    out.mkdir(parents=True)
    model=from_checkpoint(state['horizontal']).cuda().eval()
    xyz=read_xyz_pcd(pcd)*1000.;sequences,geometry=make_mixed_sequences(xyz)
    structural,process=point_structure(xyz,sequences)
    predictions,diagnostic=infer_from_sequences(state,model,sequences,structural,process)
    plate=estimate_plate(state['attribute_heads'],sequences,predictions)
    height=estimate_height(state['height_heads'],FiniteContactPlates(xyz),ContinuousJointBases(xyz),
                           sequences,predictions,structural,process)
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf8')
    (out/'height_candidates.json').write_text(json.dumps(height,indent=2,ensure_ascii=False),encoding='utf8')
    (out/'vertical_diagnostic.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf8')
    children=save_children(predictions,out,'input_'+sha256(pcd)[:12]);save_all(predictions,out)
    plate_count=export_attributes(out,plate)
    profile_path=out/'attribute_profiles.json';profile=json.loads(profile_path.read_text('utf8'))
    profile['children']={row['parent_id']+'_'+str(row['child_index']+1):row for row in profile['children']}
    profile_path.write_text(json.dumps(profile,indent=2,ensure_ascii=False),encoding='utf8')
    height_count,fields=export_height(out,height)
    schema_path=out/'output_schema.json';schema=json.loads(schema_path.read_text('utf8'))
    schema.update(output_schema_version='structured_height_candidate_v5',columns=fields,
        attribute_profile_reference='attribute_profiles.json#<weld_id>',height_profile_reference='height_profiles.json#<weld_id>',
        csv_and_child_workbook_have_same_rows=True,customer_direction_and_side_convention_frozen=False,
        reviewed_customer_attribute_truth=False,along_line_fields_complete=False,full_contract_output_compliant=False)
    schema_path.write_text(json.dumps(schema,indent=2,ensure_ascii=False),encoding='utf8')
    torch.cuda.synchronize()
    receipt=dict(input=str(Path(pcd).resolve()),input_sha256=sha256(pcd),points=len(xyz),children=children,
        plate_attribute_candidate_children=plate_count,height_endpoint_candidate_values=height_count,
        checkpoint_sha256=sha256(checkpoint),single_checkpoint=True,runtime_STEP_used=False,
        runtime_truth_used=False,runtime_preprocessing_cache_used=False,geometry=geometry,
        elapsed_seconds_diagnostic_only=time.perf_counter()-started,speed_gate='DEFERRED_BY_USER',
        full_contract_pass=False,customer_truth_reviewed=False,source_sha256=sha256(__file__))
    (out/'inference_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf8')
    print('STRUCTURED_HEIGHT_RAW',json.dumps({key:receipt[key] for key in ('children','height_endpoint_candidate_values','elapsed_seconds_diagnostic_only')}),flush=True)
    return predictions,height,receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pcd',required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--outdir',required=True)
    args=parser.parse_args();predict(args.pcd,args.checkpoint,args.outdir)
