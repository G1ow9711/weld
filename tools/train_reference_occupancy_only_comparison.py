"""Isolated reference-query detector: empirical versus balanced TRAIN sampling."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
from weld_pipeline.server_joint_reference_query_model import JointReferenceModel
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['empirical','balanced'],required=True);arm=parser.parse_args().arm;dest=BASE/('reference_occupancy_only_'+arm+'_v1');dest.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1031);source=BASE/'positive_query_head_v1/joint_reference_v1';r=read(source/'receipt.json');assert digest(source/'model.pt')==r['model_sha256'];state=torch.load(source/'model.pt',map_location='cpu',weights_only=False);keys=state['presence_training_keys'];xx=[];yy=[];ww=[];bindings={}
 for key in keys:
  folder=BASE/'reference_query_task_v1'/key;rr=read(folder/'receipt.json');assert digest(folder/'receipt.json')==r['presence_bindings'][key];assert digest(folder/'sampled_presence.npz')==rr['files']['sampled_presence.npz'];z=np.load(folder/'sampled_presence.npz');x=z['features'];y=z['labels'];w=z['weights'].astype(np.float64);assert np.all((y==0)|(y==1)) and np.isclose(w.sum(),1);bindings[key]=digest(folder/'receipt.json')
  if arm=='balanced':
   for label in (0,1):mask=y==label;assert mask.any();w[mask]*=.5/w[mask].sum()
  xx.append(np.clip((x-state['mean'])/state['std'],-20,20));yy.append(y);ww.append(w/len(keys))
 tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32);x=tensor(np.concatenate(xx));y=tensor(np.concatenate(yy));w=tensor(np.concatenate(ww));model=JointReferenceModel().cuda();model.load_state_dict(state['state_dict']);model.interval.requires_grad_(False);parameters=list(model.trunk.parameters())+list(model.reference.parameters());optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,16000,eta_min=1e-5);history=[];write(dest/'protocol.json',dict(arm=arm,training_keys=keys,initial_model_sha256=r['model_sha256'],steps=16000,batch_size=512,seed=1031,loss='Reference query occupancy BCE only; no geometry loss',sampling='Equal family; '+('empirical population via inverse inclusion weights' if arm=='empirical' else 'half reference positive / half reference background within family, inverse inclusion weighting within class'),score_not_calibrated_to_physical_weld_probability=True,reference_background_not_physical_negative=True,geometry_head_not_deployable_after_trunk_update=True,old_endpoint_selector_unchanged=True,threshold_for_fit_audit=.5,heldout_reads=0,DEV_reads=0))
 for step in range(16000):
  ids=torch.multinomial(w,512,replacement=True);logits=model.reference(model.trunk(x[ids])).squeeze(1);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[ids]);assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step()
  if step%1000==0:history.append(dict(step=step,BCE=float(loss.detach())));print(arm,history[-1],flush=True)
 model.eval();torch.save(dict(state_dict=model.cpu().state_dict(),mean=state['mean'],std=state['std'],presence_training_keys=keys,training_keys=keys,initial_shared_trunk_training_keys=state['training_keys'],reference_only=True),dest/'model.pt');write(dest/'receipt.json',dict(status='EXPERIMENTAL_REFERENCE_DETECTOR_ONLY_NOT_PROMOTED',model_sha256=digest(dest/'model.pt'),presence_bindings=bindings,training_keys=keys,history=history,fit_calls=1,optimizer_steps=16000,heldout_reads=0,DEV_reads=0,seconds=time.perf_counter()-start,source_sha256=digest(__file__)))
if __name__=='__main__':main()
