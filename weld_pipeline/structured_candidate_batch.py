"""Single-checkpoint local batch API with explicit per-item errors and receipts."""
import argparse
import collections
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from openpyxl import Workbook

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.predict_structured_height_candidate_v2 import predict,MODEL_NAME

VERSION='structured_height_candidate_batch_v1'
ERROR_CODES=dict(BATCH_ITEM_INVALID='Each item needs nonempty string workpiece_id and pcd fields.',
    DUPLICATE_WORKPIECE_ID='Duplicate workpiece identifiers in the same batch are rejected.',
    INPUT_NOT_FOUND='Input path is not an existing file.',INPUT_INVALID_PCD='PCD parsing, point count, schema or finite-value validation failed.',
    INPUT_EMPTY='Zero-point input is invalid; it is not a successful no-weld result.',
    INPUT_IO_FAILED='The input file could not be read or hashed.',MODEL_NOT_FOUND='Checkpoint is not an existing file.',
    MODEL_INVALID='Checkpoint type or frozen inference sources failed validation.',ACCELERATOR_UNAVAILABLE='This candidate requires CUDA.',
    INFERENCE_FAILED='Preprocessing, model or geometry/attribute inference raised an exception.',
    OUTPUT_FAILED='Output generation or CSV/workbook identity validation failed.')


class BatchError(Exception):
    def __init__(self,code,message):
        super().__init__(message);self.code=code


def validate_model(checkpoint):
    path=Path(checkpoint)
    if not path.is_file():raise BatchError('MODEL_NOT_FOUND',str(path))
    try:
        state=torch.load(path,map_location='cpu')
        if state['model_name']!=MODEL_NAME:raise ValueError('Expected v5r1 development candidate')
        verify_cached_inference_sources(state,path);verify_cached_inference_sources(state['horizontal'],path)
        for source,digest in state['height_heads']['config']['sources'].items():
            if sha256(source)!=digest:raise ValueError('Frozen height inference source changed: '+source)
        digest=sha256(path)
    except Exception as error:raise BatchError('MODEL_INVALID',str(error)) from error
    if not torch.cuda.is_available():raise BatchError('ACCELERATOR_UNAVAILABLE','CUDA not available')
    return digest


def caller_identity(out,workpiece_id):
    path=out/'predicted_children.csv'
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);rows=list(reader);fields=list(reader.fieldnames or [])
    if 'workpiece_id' not in fields or 'weld_id' not in fields:raise ValueError('Child identifier fields missing')
    seen=set()
    for row in rows:
        row['workpiece_id']=workpiece_id;key=(workpiece_id,row['weld_id'])
        if key in seen:raise ValueError('Duplicate child key')
        seen.add(key)
    with path.open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    book=Workbook();sheet=book.active;sheet.title='子段结果';sheet.append(fields)
    for row in rows:
        sheet.append([row.get(field,'') for field in fields])
        # Caller identifiers and other CSV strings remain text in the workbook.
        for cell in sheet[sheet.max_row]:
            if isinstance(cell.value,str):cell.data_type='s'
    sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
    book.save(out/'predicted_children.xlsx');book.close()
    return len(rows)


def run_batch(items,checkpoint,outdir):
    if not isinstance(items,list):raise ValueError('Batch input must be a JSON list')
    root=Path(outdir)
    if root.exists():raise ValueError('Use a new batch output directory')
    root.mkdir(parents=True)
    model_hash=None;model_error=None;preflight_started=time.perf_counter()
    try:model_hash=validate_model(checkpoint)
    except BatchError as error:model_error=error
    model_preflight_seconds=time.perf_counter()-preflight_started
    counts=collections.Counter(item.get('workpiece_id') for item in items
        if isinstance(item,dict) and isinstance(item.get('workpiece_id'),str))
    receipts=[];started=time.perf_counter()
    for index,item in enumerate(items):
        workpiece=item.get('workpiece_id') if isinstance(item,dict) else None
        pcd_value=item.get('pcd') if isinstance(item,dict) else None
        identifier=hashlib.sha256(str(workpiece).encode('utf8')).hexdigest()[:12]
        piece_root=root/(f'{index:04d}_'+identifier);piece_root.mkdir()
        out=piece_root/'output';piece_started=time.perf_counter()
        receipt=dict(index=index,workpiece_id=workpiece,input_path=pcd_value,model_sha256=model_hash,
            status='FAILED',reason_code=None,message=None,child_rows=None,input_sha256=None,
            attribute_acceptance_complete=False,full_contract_pass=False)
        stage='input'
        try:
            if not isinstance(workpiece,str) or not workpiece.strip() or not isinstance(pcd_value,str) or not pcd_value.strip():
                raise BatchError('BATCH_ITEM_INVALID',ERROR_CODES['BATCH_ITEM_INVALID'])
            if counts[workpiece]>1:raise BatchError('DUPLICATE_WORKPIECE_ID',workpiece)
            if model_error:raise model_error
            pcd=Path(pcd_value)
            if not pcd.is_file():raise BatchError('INPUT_NOT_FOUND',str(pcd))
            try:
                receipt['input_sha256']=sha256(pcd)
                xyz=read_xyz_pcd(pcd)
            except OSError as error:raise BatchError('INPUT_IO_FAILED',str(error)) from error
            except Exception as error:raise BatchError('INPUT_INVALID_PCD',str(error)) from error
            if not len(xyz):raise BatchError('INPUT_EMPTY',ERROR_CODES['INPUT_EMPTY'])
            receipt['input_points']=len(xyz);del xyz
            receipt['input_validation_seconds']=time.perf_counter()-piece_started
            stage='inference'
            predictions,heights,inference=predict(pcd,checkpoint,out)
            stage='output'
            rows=caller_identity(out,workpiece)
            if rows!=sum(len(p['segments']) for p in predictions):raise ValueError('Export child count differs')
            if sha256(pcd)!=receipt['input_sha256'] or sha256(checkpoint)!=model_hash:
                raise ValueError('Frozen input or model changed during inference')
            receipt.update(status='SUCCESS_NO_WELDS' if rows==0 else 'SUCCESS_WITH_CANDIDATES',
                reason_code='NO_PREDICTED_WELDS' if rows==0 else 'DEVELOPMENT_ATTRIBUTES_AND_ACCEPTANCE_INCOMPLETE',
                child_rows=rows,output_directory=str(out.resolve()),input_unchanged=True,model_unchanged=True,
                runtime_STEP_used=False,runtime_truth_used=False,runtime_preprocessing_cache_used=False,
                model_name=inference.get('model_name',MODEL_NAME))
        except BatchError as error:receipt.update(reason_code=error.code,message=str(error),exception_type=type(error).__name__)
        except Exception as error:
            receipt.update(reason_code='OUTPUT_FAILED' if stage=='output' else 'INFERENCE_FAILED',
                           message=str(error),exception_type=type(error).__name__)
        receipt['elapsed_seconds']=time.perf_counter()-piece_started
        receipt['run_record']=str((piece_root/'run_record.json').resolve())
        (piece_root/'run_record.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf8')
        receipts.append(receipt)
        print('BATCH_ITEM',index,workpiece,receipt['status'],receipt['reason_code'],'CHILDREN',receipt['child_rows'],flush=True)
    source_files=[__file__,'weld_pipeline/predict_structured_height_candidate_v2.py','training/dataio.py']
    summary=dict(version=VERSION,model_path=str(Path(checkpoint).resolve()),model_sha256=model_hash,
        model_name=MODEL_NAME,single_shared_checkpoint=True,items=receipts,total=len(items),
        successful=sum(r['status'].startswith('SUCCESS') for r in receipts),failed=sum(r['status']=='FAILED' for r in receipts),
        no_weld_results=sum(r['status']=='SUCCESS_NO_WELDS' for r in receipts),
        elapsed_seconds=time.perf_counter()-started,model_preflight_seconds=model_preflight_seconds,
        source_sha256={p:sha256(p) for p in source_files},error_codes=ERROR_CODES,
        environment=dict(python=platform.python_version(),OS=platform.platform(),torch=torch.__version__,numpy=np.__version__,
                         CUDA_runtime=torch.version.cuda,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        coordinate_units='mm',PCD_input_units='m',shared_unit_scale=1000.,
        inference_runtime_success_does_not_mean_accuracy_acceptance=True,full_contract_pass=False)
    path=root/'batch_summary.json'
    path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    (root/'batch_items.json').write_text(json.dumps(items,indent=2,ensure_ascii=False),encoding='utf8')
    return summary


def main():
    parser=argparse.ArgumentParser(description='v5r1 development candidate batch API; not contract acceptance')
    inputs=parser.add_mutually_exclusive_group(required=True);inputs.add_argument('--manifest');inputs.add_argument('--pcd')
    parser.add_argument('--workpiece-id');parser.add_argument('--checkpoint',required=True);parser.add_argument('--outdir',required=True)
    args=parser.parse_args()
    if args.manifest:items=json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'))
    else:
        if not args.workpiece_id:parser.error('--workpiece-id required with --pcd')
        items=[dict(workpiece_id=args.workpiece_id,pcd=args.pcd)]
    summary=run_batch(items,args.checkpoint,args.outdir)
    print('BATCH_COMPLETE','SUCCESS',summary['successful'],'FAILED',summary['failed'],flush=True)
    raise SystemExit(1 if summary['failed'] else 0)


if __name__=='__main__':main()
