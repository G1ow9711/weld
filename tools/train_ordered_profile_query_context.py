"""Fixed TRAIN-only attention experiment on complete input profile sequences."""
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from weld_pipeline.server_ordered_profile_context import ProfileSequences,OrderedContextModel
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';SRC=BASE/'positive_query_head_v1/neural_expanded_ten_v1';DEST=BASE/'positive_query_head_v1/ordered_context_v1'
def main():
 start=time.perf_counter();DEST.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(1031);receipt=read(SRC/'receipt.json');state=torch.load(SRC/'model.pt',map_location='cpu',weights_only=False);assert digest(SRC/'model.pt')==receipt['model_sha256'];train=receipt['training_keys'];held=receipt['heldout_keys'];keys=train+held
 write(DEST/'protocol.json',dict(training_keys=train,heldout_keys=held,initial_checkpoint_sha256=receipt['model_sha256'],steps=12000,batch=128,learning_rate=.0001,learning_rate_end=.00001,seed=1031,context='Every input fragment in finite carrier, side symmetric mean/min/max152, ordered arclength span ratios and log1p physical length; no truncation, no GT endpoints in context',model='Frozen-definition 4-head query-to-token attention128 + residual4 on warm local457; residual initially zero, local weights trainable',positive_only=True,physical_loss_unchanged=True,no_heldout_checkpoint_selection=True,not_parameter_matched_ablation=True,not_full_chain_OOF=True,not_contract_model=True))
 data={};sequences=[];train_sequences=[];input_bindings={}
 for key in keys:
  folder=Path(receipt['binding_paths'][key]);assert digest(folder/'receipt.json')==receipt['bindings'][key];records=read(folder/'records.json');x=np.load(folder/'features.npy');z=np.load(folder/'targets.npz');source=(ROOT/'server_rebuild/best_baseline_restore/legacy_runtime' if key in ('21','BM79A_native_local','10','25','36') else BASE/'frozen_features/legacy_runtime')/('source_'+key);provider=ProfileSequences(source,key);mapping={};lengths={}
  for pid in sorted({r['profile'] for r in records}):
   token,length=provider.tokens(pid);mapping[pid]=len(sequences);lengths[pid]=length;sequences.append(token)
   if key in train:train_sequences.append(token)
  input_bindings[key]=provider.bindings;provider.close();data[key]=dict(x=x,y=z['interval_ratios'],t=z['weld_type'],w=z['weights'],seq=np.array([mapping[r['profile']] for r in records]),length=np.array([lengths[r['profile']] for r in records]))
 raw=np.concatenate(train_sequences);token_mean=raw.mean(0,dtype=np.float64).astype(np.float32);token_std=np.maximum(raw.std(0,dtype=np.float64),1e-5).astype(np.float32);del raw,train_sequences
 maximum=max(map(len,sequences));padded=np.zeros((len(sequences),maximum,459),np.float32);valid=np.zeros((len(sequences),maximum),bool)
 for i,s in enumerate(sequences):padded[i,:len(s)]=np.clip((s-token_mean)/token_std,-20,20);valid[i,:len(s)]=True
 del sequences
 tensor=lambda v:torch.as_tensor(v,device='cuda',dtype=torch.float32)
 tokens=tensor(padded);mask=torch.as_tensor(valid,device='cuda');del padded,valid
 mean,std=state['mean'],state['std'];model=OrderedContextModel().cuda();model.local.load_state_dict({k.removeprefix('net.'):v for k,v in state['state_dict'].items()});model.eval()
 # Warm residual must exactly preserve old model independently of padding/context.
 from tools.train_query_context_neural_comparison import Model
 baseline=Model(457).cuda();baseline.load_state_dict(state['state_dict']);baseline.eval();k=train[0];test=tensor(np.clip((data[k]['x']-mean)/std,-20,20));q=tensor(data[k]['x'][:,456]);ids=torch.as_tensor(data[k]['seq'],device='cuda')
 with torch.no_grad():
  a=model(test,q,tokens[ids],mask[ids]);b=baseline(test,q)
  assert all(torch.equal(x,y) for x,y in zip(a,b))
 del baseline
 write(DEST/'input_controls.json',dict(status='PASS',initial_predictions_bitwise_equal_to_predecessor=True,complete_profile_tokens=True,maximum_tokens=maximum,profiles=len(tokens),token_normalization_only_TRAIN=True,input_bindings=input_bindings))
 x=np.concatenate([data[k]['x'] for k in train]);xx=tensor(np.clip((x-mean)/std,-20,20));q=tensor(x[:,456]);y=tensor(np.concatenate([data[k]['y'] for k in train]));length=tensor(np.concatenate([data[k]['length'] for k in train]));typ=torch.as_tensor(np.concatenate([data[k]['t'] for k in train]),device='cuda',dtype=torch.long);seq=torch.as_tensor(np.concatenate([data[k]['seq'] for k in train]),device='cuda');w=tensor(np.concatenate([data[k]['w']/len(train) for k in train]));opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12000,eta_min=.00001);history=[];model.train()
 for step in range(12000):
  ids=torch.multinomial(w,128,replacement=True);s=seq[ids];p,t=model(xx[ids],q[ids],tokens[s],mask[s]);loss=nn.functional.smooth_l1_loss((p-y[ids])*length[ids,None]/100,torch.zeros_like(p),beta=.03)+.1*nn.functional.cross_entropy(t,typ[ids]);assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();scheduler.step()
  if step%1000==0:history.append(dict(step=step,loss=float(loss.detach())));print(history[-1],flush=True)
 model.eval();torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,token_mean=token_mean,token_std=token_std,training_keys=train),DEST/'model.pt');model.cuda()
 with torch.no_grad():
  for key in keys:
   pred=[];roles=[];v=data[key]
   for begin in range(0,len(v['x']),128):
    raw=v['x'][begin:begin+128];s=torch.as_tensor(v['seq'][begin:begin+128],device='cuda');p,t=model(tensor(np.clip((raw-mean)/std,-20,20)),tensor(raw[:,456]),tokens[s],mask[s]);pred.append(p.cpu().numpy());roles.append(t.argmax(1).cpu().numpy())
   np.savez(DEST/(key+('_training.npz' if key in train else '_heldout.npz')),interval_ratios=np.concatenate(pred),weld_type=np.concatenate(roles))
 write(DEST/'receipt.json',dict(status='ORDERED_CONTEXT_EXPERIMENT_NOT_PROMOTED',bindings=receipt['bindings'],binding_paths=receipt['binding_paths'],training_keys=train,heldout_keys=held,model_sha256=digest(DEST/'model.pt'),files={p.name:digest(p) for p in DEST.glob('*') if p.is_file()},fit_calls=1,optimizer_steps=12000,history=history,DEV_reads=0,seconds=time.perf_counter()-start))
if __name__=='__main__':main()
