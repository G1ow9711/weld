"""Continue the underfit local neural proposal head with a fixed longer schedule."""
import time
import numpy as np
import torch
from torch import nn
import tools.train_query_context_neural_comparison as base_training
from tools.train_query_context_neural_comparison import Model,BASE,DATA
TRAIN=base_training.TRAIN+('new_034230635b7a','new_07380fe2eac4','new_0d355d5d0cbe')
HELD=base_training.HELD
KEYS=TRAIN+HELD
EXPANDED={'new_ac2008a03607','new_fcaf2af1abee'}
def data_folder(key):
 return (BASE/'exhaustive_carrier_supervision_v1/positive_query' if key in EXPANDED else DATA)/key
def prepare(key):
 previous=base_training.DATA
 try:
  base_training.DATA=data_folder(key).parent
  return base_training.prepare(key)
 finally:base_training.DATA=previous
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
SRC=DATA/'neural_local_continuation_v1';DEST=DATA/'neural_expanded_ten_v1'
def main():
 DEST.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(1031);start=time.perf_counter();r=read(SRC/'receipt.json');assert digest(SRC/'model.pt')==r['model_sha256'];s=torch.load(SRC/'model.pt',map_location='cpu',weights_only=False);assert tuple(s['training_keys'])==base_training.TRAIN;write(DEST/'protocol.json',dict(initial_checkpoint_sha256=r['model_sha256'],training_keys=TRAIN,heldout_keys=HELD,additional_steps=18000,total_steps_with_predecessor=38000,batch_size=512,seed=1031,learning_rate_start=.0003,learning_rate_end=.00003,optimizer_state='Restarted AdamW; predecessor did not save optimizer',weight_decay=.0001,same_model_and_physical_loss=True,reason='Add three completed TRAIN families and all compatible carrier positive supervision for first two expansion families; held families unchanged',supervision_paths={k:str(data_folder(k)) for k in KEYS},no_physical_negative_labels=True,not_controlled_single_factor_ablation=True,no_heldout_model_selection=True,proposal_only=True,not_full_chain_OOF=True));data={k:prepare(k) for k in KEYS};mean,std=s['mean'],s['std'];model=Model(457).cuda();model.load_state_dict(s['state_dict']);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=18000,eta_min=.00003);tensor=lambda v:torch.as_tensor(v,device='cuda',dtype=torch.float32);x=np.concatenate([data[k]['x'] for k in TRAIN]);xx=tensor(np.clip((x-mean)/std,-20,20));q=tensor(x[:,456]);y=tensor(np.concatenate([data[k]['y'] for k in TRAIN]));length=tensor(np.concatenate([data[k]['length'] for k in TRAIN]));typ=torch.as_tensor(np.concatenate([data[k]['t'] for k in TRAIN]),device='cuda',dtype=torch.long);w=tensor(np.concatenate([data[k]['w']/len(TRAIN) for k in TRAIN]));history=[]
 for step in range(18000):
  ids=torch.multinomial(w,512,replacement=True);pred,role=model(xx[ids],q[ids]);loss=nn.functional.smooth_l1_loss((pred-y[ids])*length[ids,None]/100.,torch.zeros_like(pred),beta=.03)+.1*nn.functional.cross_entropy(role,typ[ids]);assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();scheduler.step()
  if step%1000==0:history.append(dict(additional_step=step,loss=float(loss.detach()),lr=opt.param_groups[0]['lr']));print(history[-1],flush=True)
 model.eval();torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,columns=457,training_keys=TRAIN,protocol_sha256=digest(DEST/'protocol.json')),DEST/'model.pt');model.cuda()
 with torch.no_grad():
  for key in KEYS:
   x=data[key]['x'];pred=[];types=[]
   for first in range(0,len(x),4096):
    raw=x[first:first+4096];p,t=model(tensor(np.clip((raw-mean)/std,-20,20)),tensor(raw[:,456]));pred.append(p.cpu().numpy());types.append(t.argmax(1).cpu().numpy())
   np.savez(DEST/(key+('_training.npz' if key in TRAIN else '_heldout.npz')),interval_ratios=np.concatenate(pred),weld_type=np.concatenate(types))
 write(DEST/'receipt.json',dict(status='CONTINUED_EXPERIMENTAL_NEURAL_PROPOSAL_HEAD_NOT_PROMOTED',bindings={k:digest(data_folder(k)/'receipt.json') for k in KEYS},binding_paths={k:str(data_folder(k)) for k in KEYS},training_keys=TRAIN,heldout_keys=HELD,model_sha256=digest(DEST/'model.pt'),files={p.name:digest(p) for p in DEST.glob('*') if p.is_file()},fit_calls=1,optimizer_steps=18000,history=history,DEV_reads=0,seconds=time.perf_counter()-start))
if __name__=='__main__':main()
