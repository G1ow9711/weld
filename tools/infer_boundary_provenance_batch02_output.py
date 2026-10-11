"""Frozen detector queries -> input event endpoints -> actual output and D3."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_boundary_event_provenance_selector import BoundaryEventProvenanceSelector
from weld_pipeline.server_fixed_fragment_queries import FixedQueries,GRID
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.world_set_runtime_285 import line_segment
from weld_pipeline.server_array_cached_d3_evaluation import score_native_parents_cached
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def run(key,comparison):
 arm="balanced"
 start=time.perf_counter();torch.set_num_threads(2);folder=BASE/('boundary_provenance_batch02_'+comparison+'_v1');r=read(folder/'receipt.json');assert digest(folder/'model.pt')==r['model_sha256'];state=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=BoundaryEventProvenanceSelector().cuda().eval();model.load_state_dict(state['state_dict']);cache=BASE/('reference_detector_'+arm+'_event_cache_v1')/key;cr=read(cache/'receipt.json');assert digest(cache/'events.npz')==cr['events_sha256'];assert cr['event_generator_sha256']==read(BASE/'sampled_boundary_anchor_training_v1/protocol.json')['event_generator_sha256'];ev=np.load(cache/'events.npz');provfolder=cache;pr=read(provfolder/'receipt.json');assert digest(provfolder/'provenance.npy')==pr['features_sha256'];provenance=np.load(provfolder/'provenance.npy');positions=ev['positions_mm'];indices=ev['feature_indices'];offsets=ev['offsets'];ids=ev['selected_query_ids'];source=BASE/'frozen_features/legacy_runtime'/('source_'+key);inp=FixedQueries(source,key);assert inp.bindings==cr['input_bindings'];previous=BASE/('reference_detector_'+arm+'_full_output_v1')/key;freeze=read(previous/'prediction_freeze.json');assert digest(previous/'prediction_freeze.json')==cr['detector_freeze_sha256'];assert digest(previous/'predictions.npz')==freeze['predictions_sha256'];pred=np.load(previous/'predictions.npz');chosen=np.flatnonzero(pred['valid']&pred['line_eligible']&(pred['reference_occupancy_score']>=.5));assert np.array_equal(ids,chosen);pids=inp.pid[ids//4];bounds=np.empty((len(ids),2),np.float64);event_ids=np.empty((len(ids),2),np.int64);confidence=np.empty((len(ids),2),np.float32);seen=np.zeros(len(ids),bool);tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32);mean,std=state['mean'],state['std'];model.eval()
 with torch.no_grad():
  for index,pid in enumerate(ev['profile_ids']):
   a,b=offsets[index:index+2];q=positions[a:b];length=float(ev['profile_length_mm'][index]);rows=indices[a:b];raw=np.asarray(inp.raw[np.column_stack((2*rows,2*rows+1))],np.float64);features=np.column_stack((raw.mean(1),raw.min(1),raw.max(1),q/length)).astype(np.float32);features=tensor(np.clip((features-mean)/std,-20,20))[None];er=tensor(q/length)[None];qp=torch.as_tensor(q,device='cuda',dtype=torch.float64)[None];matching=np.flatnonzero(pids==pid)
   for begin in range(0,len(matching),16):
    local=matching[begin:begin+16];query_ids=ids[local];x=inp.features(query_ids);spans=inp.spans[inp.canonical[query_ids//4],:2];query_mm=spans[:,0]+GRID[query_ids%4]*(spans[:,1]-spans[:,0]);qm=torch.as_tensor(query_mm,device='cuda',dtype=torch.float64);legal=torch.stack((qp<=qm[:,None],qp>=qm[:,None]),2);B=len(local)
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(tensor(np.clip((x-mean)/std,-20,20)),features.expand(B,-1,-1),tensor(x[:,456]),er.expand(B,-1),tensor(np.full(B,length)),tensor(np.clip((provenance[a:b]-state['provenance_mean'])/state['provenance_std'],-20,20))[None].expand(B,-1,-1))
    assert torch.isfinite(logits).all();masked=logits.float().masked_fill(~legal,-torch.inf);take=masked.argmax(1);prob=masked.softmax(1).gather(1,take[:,None,:]).squeeze(1);take=take.cpu().numpy();event_ids[local]=a+take;bounds[local]=q[take];confidence[local]=prob.cpu().numpy();seen[local]=True
 assert seen.all();out=BASE/('boundary_provenance_batch02_'+comparison+'_full_output_v1')/key;out.mkdir(parents=True,exist_ok=False);np.savez(out/'predictions.npz',selected_query_ids=ids,profile_ids=pids,interval_mm=bounds,event_ids=event_ids,endpoint_probability=confidence)
 from weld_pipeline.server_compact_line_predictions import build_compact_lines
 controls=read(BASE/'compact_line_output_controls_v1/receipt.json');assert controls['status']=='PASS';assert controls['compact_module_sha256']==digest('weld_pipeline/server_compact_line_predictions.py')
 decoded,compact=build_compact_lines(inp.profiles,pids,bounds,pred['weld_type'][ids],ids,event_ids);np.savez(out/'compact_predictions.npz',**compact)
 write(out/'prediction_freeze.json',dict(model_sha256=r['model_sha256'],detector_freeze_sha256=digest(previous/'prediction_freeze.json'),input_event_receipt_sha256=digest(cache/'receipt.json'),queries=len(ids),unique_intervals=decoded['total_prediction_children'],invalid_intervals=len(decoded['invalid_predictions']),GT_reads=0,STEP_reads=0,storage='Lossless compact endpoints with unchanged original1mm D3 samples regenerated in memory',files={name:digest(out/name) for name in ['predictions.npz','compact_predictions.npz']},compact_decoder_sha256=digest('weld_pipeline/server_compact_line_predictions.py'),equivalence_control_sha256=digest(BASE/'compact_line_output_controls_v1/receipt.json'),source_sha256=digest(__file__)))
 old=read(source/'run379'/('test_'+key)/'final_decoded.json');union=dict(old,parents=old['parents']+decoded['parents'],invalid_predictions=old['invalid_predictions']+decoded['invalid_predictions'],total_prediction_children=old['total_prediction_children']+decoded['total_prediction_children']);target=read(BASE/'labels'/key/'targets.json');reports=[]
 for name,output in [('new',decoded),('union',union)]:
  score=score_native_parents_cached(output,target,[False]*len(target['parents']));write(out/(name+'_score.json'),score);reports.append(dict(branch=name,**score['categories']['overall']))
 write(out/'receipt.json',dict(status='REFERENCE_DETECTOR_AND_FIXED_ENDPOINT_MODEL_FULL_OUTPUT_SCORED_NOT_PROMOTED',key=key,arm=arm,comparison=comparison,rows=reports,fit_calls=0,DEV_reads=0,seconds=time.perf_counter()-start));inp.close();print(key,arm,reports,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--comparison',required=True,choices=['control6','expanded8']);a=p.parse_args();run(a.key,a.comparison)
