"""Standalone pure-PCD inference for new-contract development checkpoints."""
import argparse
import json
import time
from pathlib import Path
import torch
from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.checkpoint_sources import verify_cached_inference_sources
from weld_pipeline.candidate_extent_model import from_checkpoint as extent_factory
from weld_pipeline.contract_horizontal_model import from_checkpoint as horizontal_factory
from weld_pipeline.mixed_sequences import make_mixed_sequences,FEATURE_VERSION
from weld_pipeline.typed_contact_model import infer_probabilities
from weld_pipeline.event_gap_decoder import decode_intervals
from weld_pipeline.contract_export import save_children
from weld_pipeline.export import save_all


def factory(state):
    if state['model_name']=='candidate_extent_branch_v1':return extent_factory(state)
    if state['model_name']=='contract_horizontal_extent_v1':return horizontal_factory(state)
    raise ValueError('Unknown development model: '+state['model_name'])


def predict(pcd,checkpoint,outdir,device='auto'):
    torch.set_num_threads(4)
    start_load=time.perf_counter()
    with Path(checkpoint).open('rb') as f:state=torch.load(f,map_location='cpu')
    verify_cached_inference_sources(state,checkpoint)
    if state['feature_version']!=FEATURE_VERSION or state['features']!=84:raise ValueError('Feature version mismatch')
    chosen=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
    model=factory(state).to(chosen).eval()
    if chosen=='cuda':torch.cuda.synchronize()
    load_seconds=time.perf_counter()-start_load
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();xyz=read_xyz_pcd(pcd)*1000.
    read_seconds=time.perf_counter()-start
    before=time.perf_counter();seq,geometry=make_mixed_sequences(xyz)
    feature_seconds=time.perf_counter()-before
    before=time.perf_counter();probs=infer_probabilities(model,seq)
    pred=decode_intervals(seq,probs,threshold=.5,weld_penalty=2.,mask_weight=.25,event_gap_mm=0.)
    if chosen=='cuda':torch.cuda.synchronize()
    inference_seconds=time.perf_counter()-before
    (out/'predictions.json').write_text(json.dumps(pred,indent=2),encoding='utf8')
    count=save_children(pred,out,'input_'+sha256(pcd)[:12])
    save_all(pred,out)
    if chosen=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    receipt={'input':str(pcd),'input_sha256':sha256(pcd),'input_points':len(xyz),'device':chosen,
        'checkpoint_sha256':sha256(checkpoint),'checkpoint':str(checkpoint),
        'reference_inputs_used':False,'preprocessing_cache_used':False,'model_load_seconds':load_seconds,
        'read_seconds':read_seconds,'feature_seconds':feature_seconds,'inference_decoder_seconds':inference_seconds,
        'raw_read_to_all_exports_seconds':elapsed,'child_predictions':count,'parent_predictions':len(pred),
        'decoder':{'threshold':.5,'weld_penalty':2.,'mask_weight':.25,'event_gap_mm':0.},
        'timing_status':'DIAGNOSTIC_ONLY_USER_DEFERRED_SPEED_GATE','full_contract_pass':False,
        'source_hashes':{str(p):sha256(p) for p in [Path(__file__),Path('weld_pipeline/contract_export.py'),
            Path('weld_pipeline/export.py'),Path('weld_pipeline/mixed_sequences.py')]},'geometry':geometry}
    (out/'inference_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print('RAW',json.dumps({k:receipt[k] for k in ['input_points','child_predictions','raw_read_to_all_exports_seconds','full_contract_pass']}),flush=True)
    return pred,receipt


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pcd',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--outdir',required=True);ap.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    args=ap.parse_args();predict(args.pcd,args.checkpoint,args.outdir,args.device)
