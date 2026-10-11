"""Freeze actual TRAIN query selections for later prediction-quality supervision."""
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.server_joint_reference_query_model import JointReferenceModel
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 torch.set_num_threads(2);source=BASE/'positive_query_head_v1/joint_reference_v1';r=read(source/'receipt.json');assert digest(source/'model.pt')==r['model_sha256'];state=torch.load(source/'model.pt',map_location='cpu',weights_only=False);model=JointReferenceModel().cuda().eval();model.load_state_dict(state['state_dict']);detector=read(BASE/'reference_occupancy_only_balanced_v1/receipt.json');keys=read(BASE/'reference_query_task_v1/protocol.json')['training_keys'];summaries=[]
 for key in keys:
  cached=BASE/'reference_occupancy_only_balanced_training_fit_v1'/key;fr=read(cached/'prediction_freeze.json');assert fr['model_sha256']==detector['model_sha256'];assert digest(cached/'scores.npy')==fr['scores_sha256'];inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key);assert inp.bindings==fr['input_bindings'];scores=np.load(cached/'scores.npy');assert len(scores)==inp.n and np.isfinite(scores).all();line=np.repeat(np.array([inp.profiles[pid]['representation']['kind']=='line' for pid in inp.pid]),4);ids=np.flatnonzero(line&(scores>=.5));roles=np.full(inp.n,-1,np.int8)
  with torch.no_grad():
   for a in range(0,len(ids),4096):
    use=ids[a:a+4096];x=inp.features(use);z=torch.as_tensor(np.clip((x-state['mean'])/state['std'],-20,20),device='cuda');_,typ,_=model(z,torch.as_tensor(x[:,456],device='cuda'));roles[use]=typ.argmax(1).cpu().numpy()
  assert np.all(roles[ids]>=0);out=BASE/'reference_detector_balanced_full_output_v1'/key;out.mkdir(exist_ok=False);np.savez(out/'predictions.npz',reference_occupancy_score=scores,weld_type=roles,line_eligible=line,valid=np.ones(inp.n,bool));report=dict(key=key,arm='balanced',total_queries=inp.n,selected_queries=len(ids),selected_profiles=len(np.unique(inp.pid[ids//4])),model_sha256=detector['model_sha256'],type_model_sha256=r['model_sha256'],predictions_sha256=digest(out/'predictions.npz'),input_bindings=inp.bindings,reused_prelabel_score_freeze_sha256=digest(cached/'prediction_freeze.json'),threshold=.5,GT_reads=0,STEP_reads=0,DEV_reads=0,fit_calls=0,scope='TRAIN actual prediction pool, not label-selected queries; unselected type=-1 never decoded; downstream quality labels still pending',source_sha256=digest(__file__));write(out/'prediction_freeze.json',report);summaries.append(report);inp.close();print(key,len(ids),report['selected_profiles'],flush=True)
 write(BASE/'reference_detector_balanced_full_output_v1/training_inputs_receipt.json',dict(status='SIX_TRAIN_ACTUAL_QUERY_POOLS_FROZEN',cases=summaries,fit_calls=0,DEV_reads=0))
if __name__=='__main__':main()
