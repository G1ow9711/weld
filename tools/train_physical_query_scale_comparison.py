"""Fixed eleven-family geometry / six-family reference-occupancy experiment."""
from pathlib import Path
import time,argparse
import numpy as np
import torch
from torch import nn
from weld_pipeline.server_metric_query_model import MetricQueryModel
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DATA=BASE/'reference_query_task_v1';SRC=BASE/'positive_query_head_v1/joint_reference_v1'
parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['control457','metric460'],required=True);ARM=parser.parse_args().arm;DEST=BASE/'physical_query_scale_comparison_v1'/ARM
def extend(x,length):
 if ARM=='control457':return x
 q=x[:,456].astype('float64');return np.column_stack((x,length/1000,q*length/1000,(1-q)*length/1000)).astype(np.float32)
def main():
 start=time.perf_counter();DEST.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(1031);r=read(SRC/'receipt.json');state=torch.load(SRC/'model.pt',map_location='cpu',weights_only=False);assert digest(SRC/'model.pt')==r['model_sha256'];presence_keys=read(DATA/'protocol.json')['training_keys'];old=['21','BM79A_native_local','10','25','36'];train=old+presence_keys;held=r['heldout_keys'];keys=train+held;paths={k:(BASE/'exhaustive_carrier_supervision_v1/positive_query'/k if k in presence_keys else Path(r['binding_paths'][k])) for k in keys}
 write(DEST/'protocol.json',dict(training_keys=train,presence_training_keys=presence_keys,heldout_keys=held,initial_checkpoint_sha256=r['model_sha256'],presence_protocol_sha256=digest(DATA/'protocol.json'),arm=ARM,explicit_input_features='carrier_length_mm, query_from_start_mm, query_to_end_mm divided by1000; all frozen input geometry',matched_optimization_budget=True,steps=8000,geometry_batch=256,presence_batch=256,lr=.0001,lr_end=.00001,seed=1031,loss='Same physical endpoint SmoothL1 plus .1 type CE plus .5 BCE for projected-reference interval occupancy',presence_sampling='Equal family, empirical line reference query population via inverse inclusion weights; no physical negative assertion',geometry_sampling='Equal family and child positive witness weights',heldout_not_used_for_training_or_threshold_selection=True,output_threshold=.5,arc_reference_objective_masked=True,no_parent_assembly=True,not_contract_model=True))
 data={};bindings={}
 for k in keys:
  f=paths[k];rr=read(f/'receipt.json')
  for name,sha in rr['files'].items():assert digest(f/name)==sha
  records=read(f/'records.json');z=np.load(f/'targets.npz');src=(ROOT/'server_rebuild/best_baseline_restore/legacy_runtime' if k in old else BASE/'frozen_features/legacy_runtime')/('source_'+k);inp=FixedQueries(src,k);length=np.array([inp.spans[inp.offsets[v['profile']+1]-1,1] for v in records]);inp.close();data[k]=dict(x=extend(np.load(f/'features.npy'),length),y=z['interval_ratios'],t=z['weld_type'],w=z['weights'],length=length);bindings[k]=digest(f/'receipt.json')
 mean,std=state['mean'],state['std']
 if ARM=='metric460':
  extra=np.concatenate([data[k]['x'][:,457:] for k in train]);mean=np.r_[mean,extra.mean(0,dtype=np.float64)].astype(np.float32);std=np.r_[std,np.maximum(extra.std(0,dtype=np.float64),1e-5)].astype(np.float32)
 tensor=lambda a:torch.as_tensor(a,device='cuda',dtype=torch.float32);x=np.concatenate([data[k]['x'] for k in train]);xx=tensor(np.clip((x-mean)/std,-20,20));q=tensor(x[:,456]);y=tensor(np.concatenate([data[k]['y'] for k in train]));length=tensor(np.concatenate([data[k]['length'] for k in train]));types=torch.as_tensor(np.concatenate([data[k]['t'] for k in train]),device='cuda',dtype=torch.long);w=tensor(np.concatenate([data[k]['w']/len(train) for k in train]));del x
 samples=[];presence_bindings={}
 for k in presence_keys:
  f=DATA/k;rr=read(f/'receipt.json');assert digest(f/'sampled_presence.npz')==rr['files']['sampled_presence.npz'];z=np.load(f/'sampled_presence.npz');inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+k),k);samples.append(dict(features=extend(z['features'],inp.length[z['query_ids']//4]),labels=z['labels'],weights=z['weights']));inp.close();presence_bindings[k]=digest(f/'receipt.json')
 x=np.concatenate([z['features'] for z in samples]);ox=tensor(np.clip((x-mean)/std,-20,20));oy=tensor(np.concatenate([z['labels'] for z in samples]));ow=tensor(np.concatenate([z['weights']/len(samples) for z in samples]));del x
 model=MetricQueryModel(len(mean)).cuda();initial={k:v.clone() for k,v in state['state_dict'].items()}
 if ARM=='metric460':initial['trunk.0.weight']=torch.cat((initial['trunk.0.weight'],torch.zeros(256,3)),1)
 model.load_state_dict(initial);opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,8000,eta_min=.00001);history=[]
 for step in range(8000):
  ids=torch.multinomial(w,256,replacement=True);p,t,_=model(xx[ids],q[ids]);geometry=nn.functional.smooth_l1_loss((p-y[ids])*length[ids,None]/100,torch.zeros_like(p),beta=.03)+.1*nn.functional.cross_entropy(t,types[ids]);oi=torch.multinomial(ow,256,replacement=True);h=model.trunk(ox[oi]);presence=nn.functional.binary_cross_entropy_with_logits(model.reference(h).squeeze(1),oy[oi]);loss=geometry+.5*presence;assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();sched.step()
  if step%1000==0:history.append(dict(step=step,geometry=float(geometry.detach()),reference_BCE=float(presence.detach())));print(history[-1],flush=True)
 model.eval();torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,training_keys=train,presence_training_keys=presence_keys,columns=len(mean),arm=ARM),DEST/'model.pt');model.cuda()
 with torch.no_grad():
  for k in keys:
   x=data[k]['x'];bounds=[];roles=[];scores=[]
   for i in range(0,len(x),4096):
    raw=x[i:i+4096];p,t,o=model(tensor(np.clip((raw-mean)/std,-20,20)),tensor(raw[:,456]));bounds.append(p.cpu().numpy());roles.append(t.argmax(1).cpu().numpy());scores.append(torch.sigmoid(o).cpu().numpy())
   np.savez(DEST/(k+('_training.npz' if k in train else '_heldout.npz')),interval_ratios=np.concatenate(bounds),weld_type=np.concatenate(roles),reference_occupancy_score=np.concatenate(scores))
 write(DEST/'receipt.json',dict(status='JOINT_REFERENCE_EXPERIMENT_NOT_PROMOTED',bindings=bindings,binding_paths={k:str(v) for k,v in paths.items()},presence_bindings=presence_bindings,training_keys=train,heldout_keys=held,model_sha256=digest(DEST/'model.pt'),files={p.name:digest(p) for p in DEST.glob('*') if p.is_file()},fit_calls=1,optimizer_arm=ARM,explicit_input_features='carrier_length_mm, query_from_start_mm, query_to_end_mm divided by1000; all frozen input geometry',matched_optimization_budget=True,steps=8000,history=history,DEV_reads=0,seconds=time.perf_counter()-start))
if __name__=='__main__':main()
