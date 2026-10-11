"""Fixed-budget positive-query local/context comparison in physical endpoint units."""
import json,time,pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';DATA=BASE/'positive_query_head_v1';DEST=DATA/'neural_context_comparison_v1'
TRAIN=('21','BM79A_native_local','10','25','36','new_ac2008a03607','new_fcaf2af1abee');HELD=('new_cfc732e5f068','new_ceeae49d6777');KEYS=TRAIN+HELD
class Model(nn.Module):
 def __init__(self,n):
  super().__init__();self.net=nn.Sequential(nn.Linear(n,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,4))
 def forward(self,x,q):
  y=self.net(x);d=torch.sigmoid(y[:,:2]);return torch.stack((q*(1-d[:,0]),q+(1-q)*d[:,1]),1),y[:,2:]
def prepare(key):
 f=DATA/key;r=read(f/'receipt.json')
 for name,sha in r['files'].items():assert digest(f/name)==sha
 records=read(f/'records.json');x=np.load(f/'features.npy');z=np.load(f/'targets.npz');original=key in TRAIN[:5];src=(ROOT/'server_rebuild/best_baseline_restore/legacy_runtime' if original else BASE/'frozen_features/legacy_runtime')/('source_'+key);fr=src/('run344' if key in ('25','36') else 'run329')/('source_'+key);binding=read(fr/'receipt.json');assert digest(fr/'features.npy')==binding['features_sha256'];raw=np.load(fr/'features.npy',mmap_mode='r');alias=src/'run288'/('source_'+key+'_alias_map.npz')
 with np.load(alias) as a:canonical=a['canonical_fragments']
 g=next(v for v in read(src/'run285/all_geometry_frozen_before_labels.json')['cases'] if v['source']==key)
 with np.load(src/'run285'/g['geometry_file']) as a:offsets=a['offsets'];spans=a['spans']
 pid=np.searchsorted(offsets,canonical,side='right')-1;assert np.all(np.diff(pid)>=0);unique,first,count=np.unique(pid,return_index=True,return_counts=True);wanted={r['profile'] for r in records};context={}
 for p,start,n in zip(unique,first,count):
  if p not in wanted:continue
  vals=np.asarray(raw[start*2:(start+n)*2],np.float64).reshape(n,2,152);weights=spans[canonical[start:start+n],1]-spans[canonical[start:start+n],0];context[int(p)]=np.r_[np.average(vals.mean(1),axis=0,weights=weights),vals.min((0,1)),vals.max((0,1))]
 ctx=np.array([context[r['profile']] for r in records],np.float32);length=np.array([spans[offsets[r['profile']+1]-1,1] for r in records]);raw._mmap.close();return dict(x=x,context=ctx,length=length,y=z['interval_ratios'],t=z['weld_type'],w=z['weights'])
def main():
 DEST.mkdir(exist_ok=False);torch.set_num_threads(2);assert torch.cuda.is_available();write(DEST/'protocol.json',dict(training_keys=TRAIN,heldout_keys=HELD,steps_per_arm=2000,batch_size=512,seed=1031,hidden=[256,128],learning_rate=.001,weight_decay=.0001,arms=['local457','context913'],same_training_rows=True,same_queries=True,same_physical_loss=True,endpoint_loss='SmoothL1 on decoded endpoint error /100mm, beta=.03 (=3mm); type CE weight .1',query_conditioned_positive_only=True,no_absence_or_count_labels=True,GT_only_supervision=True,context='All canonical fragments of the same input profile: length weighted side mean, min and max; no GT boundaries used',not_full_chain_OOF=True,upstream_frozen=True,not_final_weld_model=True));data={k:prepare(k) for k in KEYS};bindings={k:digest(DATA/k/'receipt.json') for k in KEYS}
 for arm in ('local457','context913'):
  start=time.perf_counter();dest=DEST/arm;dest.mkdir();torch.manual_seed(1031);np.random.seed(1031);arrays={k:np.column_stack((v['x'],v['context'])) if arm=='context913' else v['x'] for k,v in data.items()};x=np.concatenate([arrays[k] for k in TRAIN]);mean=x.mean(0,dtype=np.float64).astype(np.float32);std=x.std(0,dtype=np.float64).astype(np.float32);std=np.maximum(std,1e-5);model=Model(x.shape[1]).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);tensor=lambda v:torch.as_tensor(v,device='cuda',dtype=torch.float32);xx=tensor(np.clip((x-mean)/std,-20,20));q=tensor(np.concatenate([data[k]['x'][:,456] for k in TRAIN]));y=tensor(np.concatenate([data[k]['y'] for k in TRAIN]));length=tensor(np.concatenate([data[k]['length'] for k in TRAIN]));typ=torch.as_tensor(np.concatenate([data[k]['t'] for k in TRAIN]),device='cuda',dtype=torch.long);w=tensor(np.concatenate([data[k]['w']/len(TRAIN) for k in TRAIN]));history=[]
  for step in range(2000):
   ids=torch.multinomial(w,512,replacement=True);pred,role=model(xx[ids],q[ids]);loss=nn.functional.smooth_l1_loss((pred-y[ids])*length[ids,None]/100.,torch.zeros_like(pred),beta=.03)+.1*nn.functional.cross_entropy(role,typ[ids]);opt.zero_grad();loss.backward();opt.step()
   if step%200==0:history.append(dict(step=step,loss=float(loss.detach())));print(arm,history[-1],flush=True)
  model.eval();torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,columns=x.shape[1],training_keys=TRAIN,protocol_sha256=digest(DEST/'protocol.json')),dest/'model.pt');model.cuda()
  with torch.no_grad():
   for key in KEYS:
    split='training' if key in TRAIN else 'heldout';predictions=[];roles=[]
    for first in range(0,len(arrays[key]),4096):
     raw=arrays[key][first:first+4096];p,t=model(tensor(np.clip((raw-mean)/std,-20,20)),tensor(data[key]['x'][first:first+4096,456]));predictions.append(p.cpu().numpy());roles.append(t.argmax(1).cpu().numpy())
    np.savez(dest/(key+'_'+split+'.npz'),interval_ratios=np.concatenate(predictions),weld_type=np.concatenate(roles))
  write(dest/'receipt.json',dict(status='EXPERIMENTAL_POSITIVE_QUERY_NEURAL_HEAD',bindings=bindings,training_keys=TRAIN,heldout_keys=HELD,fit_calls=1,optimizer_steps=2000,history=history,model_sha256=digest(dest/'model.pt'),files={p.name:digest(p) for p in dest.glob('*') if p.is_file()},seconds=time.perf_counter()-start,DEV_reads=0,full_extraction_not_measured=True));del model,opt,xx,q,y,length,typ,w;torch.cuda.empty_cache()
if __name__=='__main__':main()
