"""Complete fixed-grid inference and frozen .5 reference-task line selection."""
import argparse,time
import numpy as np
import torch
from tools.query_reference_utility_runtime import Inputs,BASE
from weld_pipeline.server_joint_reference_query_model import JointReferenceModel
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_native_d3_adapter_1056 import score_native_parents

def run(key):
 start=time.perf_counter();torch.set_num_threads(2);folder=BASE/'positive_query_head_v1/joint_reference_v1';r=read(folder/'receipt.json');assert digest(folder/'model.pt')==r['model_sha256'];s=torch.load(folder/'model.pt',map_location='cpu',weights_only=False);model=JointReferenceModel().cuda().eval();model.load_state_dict(s['state_dict']);inp=Inputs(key);out=BASE/'joint_reference_full_output'/key;out.mkdir(parents=True,exist_ok=False);n=len(inp.valid);ratio=np.empty((n,2),np.float64);role=np.empty(n,np.int8);score=np.empty(n,np.float64);tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
 with torch.no_grad():
  for begin in range(0,n,4096):
   ids=np.arange(begin,min(begin+4096,n));x=inp.features(ids)[:,:457];p,t,o=model(tensor(np.clip((x-s['mean'])/s['std'],-20,20)),tensor(x[:,456]));ratio[ids]=p.cpu().numpy();role[ids]=t.argmax(1).cpu().numpy();score[ids]=torch.sigmoid(o).cpu().numpy()
 valid=np.isfinite(ratio).all(1)&(ratio[:,0]>=0)&(ratio[:,1]<=1)&(ratio[:,0]<ratio[:,1]);line=np.repeat([inp.profiles[int(pid)]['representation']['kind']=='line' for pid in inp.pid],4);np.savez(out/'predictions.npz',interval_ratios=ratio,weld_type=role,reference_occupancy_score=score,valid=valid,line_eligible=line);inp.ratio=ratio;inp.role=role;inp.valid=valid;ids=np.flatnonzero(valid&line&(score>=.5));decoded=inp.decode(ids,score[ids]);write(out/'new_decoded.json',decoded)
 oldpath=BASE/'frozen_features/legacy_runtime'/('source_'+key)/'run379'/('test_'+key)/'final_decoded.json';old=read(oldpath);union=dict(old,parents=old['parents']+decoded['parents'],total_prediction_children=old['total_prediction_children']+decoded['total_prediction_children']);write(out/'union_decoded.json',union)
 write(out/'prediction_freeze.json',dict(model_sha256=r['model_sha256'],queries=n,selected_queries=len(ids),exported_singletons=len(decoded['parents']),threshold=.5,arc_reference_selection_excluded=True,GT_reads=0,STEP_reads=0,predictions_sha256=digest(out/'predictions.npz'),new_output_sha256=digest(out/'new_decoded.json'),union_output_sha256=digest(out/'union_decoded.json'),source_geometry_receipt_sha256=digest(inp.folder/'receipt.json'),not_parent_assembly=True))
 target=read(BASE/'labels'/key/'targets.json');rows=[]
 for name,output in [('new',decoded),('union',union)]:
  result=score_native_parents(output,target,[False]*len(target['parents']));write(out/(name+'_score.json'),result);rows.append(dict(branch=name,**result['categories']['overall']))
 write(out/'receipt.json',dict(status='EXPERIMENTAL_COMPLETE_OUTPUT_SCORED_NOT_PROMOTED',key=key,rows=rows,fit_calls=0,DEV_reads=0,seconds=time.perf_counter()-start));print(key,rows,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);run(p.parse_args().key)
