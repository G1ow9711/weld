"""Freeze attribute predictions on all actual expanded8 queries and compact outputs."""
from pathlib import Path
import argparse
import numpy as np
import torch
from weld_pipeline.server_transverse_attribute_head import TransverseAttributeHead
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';PARSER=argparse.ArgumentParser();PARSER.add_argument('--transverse-context',action='store_true');ARGS=PARSER.parse_args();OUT=BASE/('transverse_attribute_context_predictions_v1' if ARGS.transverse_context else 'transverse_attribute_control_predictions_v1')
def main():
 torch.set_num_threads(2);folder=BASE/('transverse_attribute_head_v1' if ARGS.transverse_context else 'transverse_attribute_control_v1');r=read(folder/'receipt.json');assert digest(folder/'model.pt')==r['model_sha256'];s=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=TransverseAttributeHead().eval();model.load_state_dict(s['state_dict']);features=BASE/'runtime_endpoint_attribute_features_v1';assert read(features/'receipt.json')['status']=='TWO_HELD_PREDICTED_ENDPOINT_ATTRIBUTE_INPUTS_FROZEN';proto=read(features/'protocol.json');assert proto['endpoint_model_sha256']==s['source_endpoint_model_sha256'];OUT.mkdir(exist_ok=False);reports=[]
 for key in proto['keys']:
  f=features/key;fr=read(f/'receipt.json');assert digest(f/'features.npy')==fr['features_sha256'] and digest(f/'index.npz')==fr['index_sha256'];geometry=BASE/'boundary_provenance_batch02_expanded8_full_output_v1'/key;gf=read(geometry/'prediction_freeze.json');assert digest(geometry/'prediction_freeze.json')==fr['input_prediction_freeze_sha256'];assert digest(geometry/'compact_predictions.npz')==gf['files']['compact_predictions.npz'];x=np.load(f/'features.npy');context_sha=None
  if ARGS.transverse_context:
   context=BASE/'transverse_endpoint_runtime_context_v1'/key;cr=read(context/'receipt.json');assert cr['source_feature_receipt_sha256']==digest(f/'receipt.json') and digest(context/'features.npy')==cr['features_sha256'];extra=np.load(context/'features.npy');context_sha=digest(context/'receipt.json');tp=read(BASE/'transverse_endpoint_training_context_v1/protocol.json');rp=read(BASE/'transverse_endpoint_runtime_context_v1/protocol.json')
   for field in ['columns','per_endpoint','per_face','plane_search_band_mm','sampled_boundary_rule','source_endpoint_model_sha256','module_sha256']:assert tp[field]==rp[field],field
  else:extra=np.zeros((len(x),86),np.float32)
  assert extra.shape==(len(x),86);x=np.column_stack((x,extra));index=np.load(f/'index.npz');chosen=np.load(geometry/'compact_predictions.npz')['selected_prediction_indices'];probs=[];nums=[]
  with torch.no_grad():
   for begin in range(0,len(x),1024):
    batch=torch.from_numpy(np.clip((x[begin:begin+1024]-s['mean'])/s['std'],-20,20));logits,mm=model(batch);probs.append(logits.softmax(-1).numpy());nums.append(mm.numpy())
  probability=np.concatenate(probs);numeric=np.concatenate(nums);state=probability.argmax(-1).astype(np.int8);reported=np.where(state==0,numeric,np.nan);assert np.isfinite(numeric).all() and np.all(numeric>=0);out=OUT/key;out.mkdir();np.savez(out/'query_predictions.npz',query_ids=index['query_ids'],state=state,state_probability=probability,regression_mm=numeric,reported_exact_mm=reported);np.savez(out/'compact_predictions.npz',selected_query_indices=chosen,query_ids=index['query_ids'][chosen],state=state[chosen],state_probability=probability[chosen],regression_mm=numeric[chosen],reported_exact_mm=reported[chosen]);report=dict(key=key,queries=len(x),compact_children=len(chosen),model_sha256=r['model_sha256'],endpoint_model_sha256=s['source_endpoint_model_sha256'],feature_receipt_sha256=digest(f/'receipt.json'),transverse_context_receipt_sha256=context_sha,geometry_prediction_freeze_sha256=digest(geometry/'prediction_freeze.json'),files={name:digest(out/name) for name in ['query_predictions.npz','compact_predictions.npz']},GT_reads=0,DEV_reads=0,scope='Typed attributes for all existing compact expanded8 candidates in increasing carrier direction; nonexact numeric export isNaN. No selection by reference, no pose/frame or customer CSV.');write(out/'prediction_freeze.json',report);reports.append(report);print(key,len(x),len(chosen),flush=True)
 write(OUT/'receipt.json',dict(status='TWO_HELD_ATTRIBUTE_PREDICTIONS_FROZEN_NOT_SCORED',cases=reports,fit_calls=0,GT_reads=0,DEV_reads=0))
if __name__=='__main__':main()
