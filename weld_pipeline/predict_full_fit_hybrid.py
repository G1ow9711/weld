"""Fresh raw-PCD inference; no STEP/labels/preprocessing/prediction cache inputs."""
import argparse
import json
import time
from pathlib import Path

import torch

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.contract_export import save_children
from weld_pipeline.contract_horizontal_model import from_checkpoint
from weld_pipeline.export import save_all
from weld_pipeline.full_fit_hybrid import MODEL_NAME,infer_from_sequences,point_structure
from weld_pipeline.mixed_sequences import make_mixed_sequences
from weld_pipeline.point_plate_attributes import PointPlateAttributes


def predict(pcd,checkpoint,outdir,attribute_candidates=False):
    torch.set_num_threads(4)
    started=time.perf_counter()
    state=torch.load(checkpoint,map_location='cpu')
    if state['model_name']!=MODEL_NAME:
        raise ValueError('Wrong single-checkpoint model')
    verify_cached_inference_sources(state,checkpoint)
    verify_cached_inference_sources(state['horizontal'],checkpoint)
    model=from_checkpoint(state['horizontal']).cuda().eval()
    torch.cuda.synchronize()
    load_seconds=time.perf_counter()-started
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    xyz=read_xyz_pcd(pcd)*1000.
    read_seconds=time.perf_counter()-started
    before=time.perf_counter()
    sequences,geometry=make_mixed_sequences(xyz)
    structural,process=point_structure(xyz,sequences)
    feature_seconds=time.perf_counter()-before
    before=time.perf_counter()
    predictions,diagnostic=infer_from_sequences(state,model,sequences,structural,process)
    torch.cuda.synchronize()
    inference_seconds=time.perf_counter()-before
    (out/'predictions.json').write_text(json.dumps(predictions,indent=2),encoding='utf8')
    (out/'vertical_diagnostic.json').write_text(json.dumps(diagnostic,indent=2),encoding='utf8')
    count=save_children(predictions,out,'input_'+sha256(pcd)[:12])
    save_all(predictions,out)
    if attribute_candidates:
        estimator=PointPlateAttributes(xyz)
        attributes=[]
        for w in predictions:
            for i,seg in enumerate(w['segments']):
                result=estimator.line(seg['points']) if seg['kind']=='line' else {'state':'unknown','reason':'CURVED_PLATE_ESTIMATOR_NOT_IMPLEMENTED'}
                attributes.append({'parent_id':w['id'],'child_index':i,'candidate':result})
        (out/'point_attribute_candidates.json').write_text(json.dumps({'scope':'Development candidates; estimator has not met E2 and customer side convention is unconfirmed. Contract CSV remains explicit unknown.',
            'reference_inputs_used':False,'results':attributes},ensure_ascii=False,indent=2),encoding='utf8')
    torch.cuda.synchronize()
    receipt={'pcd':str(pcd),'pcd_sha256':sha256(pcd),'input_points':len(xyz),
             'checkpoint_sha256':sha256(checkpoint),'single_checkpoint':True,
             'runtime_step_used':False,'runtime_labels_used':False,'preprocessing_cache_used':False,
             'role':'Full available-label FIT model; requires independent acceptance',
             'model_load_seconds':load_seconds,'read_seconds':read_seconds,'feature_seconds':feature_seconds,
             'inference_seconds':inference_seconds,'raw_read_to_all_exports_seconds':time.perf_counter()-started,
             'parents':len(predictions),'children':count,'geometry':geometry,
             'speed_gate':'DEFERRED_BY_USER','attribute_candidates_written':attribute_candidates,
             'full_contract_pass':False,'source_sha256':sha256(__file__)}
    (out/'inference_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('RAW_HYBRID',json.dumps({k:receipt[k] for k in ['input_points','children','raw_read_to_all_exports_seconds','single_checkpoint','runtime_step_used']}),flush=True)
    return predictions,receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pcd',required=True);parser.add_argument('--checkpoint',required=True);parser.add_argument('--outdir',required=True)
    parser.add_argument('--attribute-candidates',action='store_true')
    args=parser.parse_args()
    predict(args.pcd,args.checkpoint,args.outdir,args.attribute_candidates)
