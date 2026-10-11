"""Freeze complete input-query detector predictions without reading target labels."""
from pathlib import Path
import argparse,time
import numpy as np
import torch
from weld_pipeline.server_joint_reference_query_model import JointReferenceModel
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def run(arm,key):
 start=time.perf_counter();torch.set_num_threads(2);source=BASE/('reference_occupancy_only_'+arm+'_v1');r=read(source/'receipt.json');assert digest(source/'model.pt')==r['model_sha256'];s=torch.load(source/'model.pt',map_location='cpu',weights_only=False);assert s['reference_only'];model=JointReferenceModel().cuda().eval();model.load_state_dict(s['state_dict']);inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key);scores=np.empty(inp.n,np.float32)
 with torch.no_grad():
  for a in range(0,inp.n,4096):
   ids=np.arange(a,min(inp.n,a+4096));x=inp.features(ids);z=torch.as_tensor(np.clip((x-s['mean'])/s['std'],-20,20),device='cuda');scores[ids]=model.reference(model.trunk(z)).squeeze(1).sigmoid().cpu().numpy()
 previous=BASE/'joint_reference_full_output'/key;fr=read(previous/'prediction_freeze.json');assert digest(previous/'predictions.npz')==fr['predictions_sha256'];old=np.load(previous/'predictions.npz');roles=old['weld_type'];assert len(roles)==inp.n;line=np.repeat(np.array([inp.profiles[pid]['representation']['kind']=='line' for pid in inp.pid]),4);valid=np.isfinite(scores);assert valid.all();selected=valid&line&(scores>=.5);out=BASE/('reference_detector_'+arm+'_full_output_v1')/key;out.mkdir(parents=True,exist_ok=False);np.savez(out/'predictions.npz',reference_occupancy_score=scores,weld_type=roles,line_eligible=line,valid=valid);report=dict(key=key,arm=arm,total_queries=inp.n,selected_queries=int(selected.sum()),selected_profiles=int(len(np.unique(inp.pid[np.flatnonzero(selected)//4]))),model_sha256=r['model_sha256'],predictions_sha256=digest(out/'predictions.npz'),input_bindings=inp.bindings,type_predictions_frozen_from=digest(previous/'prediction_freeze.json'),eligibility='Finite input query on line carrier; no gating by obsolete continuous interval prediction',threshold=.5,endpoint_prediction_performed=False,GT_reads=0,STEP_reads=0,DEV_reads=0,fit_calls=0,not_full_extraction=True,source_sha256=digest(__file__),seconds=time.perf_counter()-start);write(out/'prediction_freeze.json',report);inp.close();print(report,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['empirical','balanced']);p.add_argument('--key',required=True,choices=['new_cfc732e5f068','new_ceeae49d6777']);a=p.parse_args();run(a.arm,a.key)
