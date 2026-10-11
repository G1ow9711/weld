"""Fixed-budget grid proposal-head adaptation; not a full-output promotion."""
from pathlib import Path
from collections import defaultdict, OrderedDict
import time
import numpy as np
import torch
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.server_grid_feature_stream import grid_feature_chunks
from weld_pipeline.server_grid_endpoint_selector import GridEndpointSelector
from weld_pipeline.server_grid_owner_loss import owner_pair_loss
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1031);rng=np.random.default_rng(1031)
 source=BASE/'grid_endpoint_training_v1';assert read(source/'receipt.json')['status']=='EIGHT_TRAIN_GRID_OWNER_SUPERVISION_READY'
 assert read(BASE/'grid_owner_loss_controls_v1/receipt.json')['module_sha256']==digest('weld_pipeline/server_grid_owner_loss.py')
 keys=read(source/'protocol.json')['keys'];out=BASE/'grid_endpoint_head_v1';out.mkdir(exist_ok=False)
 checkpoint=BASE/'bounded_raw_selector_v1/model.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 model=GridEndpointSelector().cuda();model.load_raw_state_dict(state['state_dict']);model.query.requires_grad_(False);model.event.requires_grad_(False)
 mean,std=state['mean'],state['std'];tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
 write(out/'protocol.json',dict(keys=keys,steps=1000,queries_per_step=4,seed=1031,lr=.0001,lr_end=.00001,sampling='Equal family then represented child then query; union of same-owner interpretations',frozen='Original query/event encoders',trained='Endpoint score MLP and explicit grid offset',precision='float32 model,float64 ordered pair mass',source_checkpoint_sha256=digest(checkpoint),new_event_provenance='Normalized zeros plus explicit grid flag; no asserted observed provenance',scope='Conditional line proposal adaptation, not null/parent/type model, not a matched old-training ablation',modules={n:digest('weld_pipeline/'+n+'.py') for n in ['server_grid_feature_stream','server_grid_endpoint_selector','server_grid_owner_loss']},DEV_reads=0))
 banks=[];bindings={}
 for key in keys:
  folder=source/key;r=read(folder/'receipt.json');assert digest(folder/'supervision.npz')==r['supervision_sha256']
  with np.load(folder/'supervision.npz') as z:s={k:z[k] for k in z.files}
  with np.load(folder/'events.npz') as z:pids=z['profile_ids'].copy()
  data=BASE/'exhaustive_carrier_supervision_v1/positive_query'/key;features=np.load(data/'features.npy',mmap_mode='r')
  inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key);assert inp.bindings==read(folder/'input_freeze.json')['input_bindings']
  groups=defaultdict(list)
  for i,(a,b) in enumerate(zip(s['interpretation_offsets'][:-1],s['interpretation_offsets'][1:])):
   for owner in set(map(tuple,s['owners'][a:b])):groups[owner].append(i)
  banks.append(dict(key=key,inp=inp,pids=pids,s=s,features=features,groups=list(groups.values())))
  bindings[key]=digest(folder/'receipt.json');print('LOADED',key,len(groups),flush=True)
 cache=OrderedDict();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.0001);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,1000,eta_min=.00001);history=[]
 for step in range(1000):
  opt.zero_grad();loss_sum=0.
  for micro in range(4):
   bi=int(rng.integers(len(banks)));bank=banks[bi];group=bank['groups'][int(rng.integers(len(bank['groups'])))];row=int(rng.choice(group));s=bank['s'];pi=int(s['query_profile_indices'][row]);pid=int(bank['pids'][pi]);key=(bi,pid)
   if key not in cache:
    inp=bank['inp'];a,b=inp.offsets[pid:pid+2];spans=inp.spans[a:b,:2];length=float(spans[-1,1]);reps=inp.representative[np.arange(a,b)];fr=np.searchsorted(inp.canonical,reps)
    chunks=list(grid_feature_chunks(length,spans,fr,inp.raw));position=np.concatenate([c['positions_mm'] for c in chunks]);features=np.concatenate([c['features'] for c in chunks]);x=tensor(np.clip((features-mean)/std,-20,20))[None];p=torch.as_tensor(position,device='cuda',dtype=torch.float64);ratio=tensor(position/length)[None]
    cache[key]=(x,p,ratio,length)
    if len(cache)>4:cache.popitem(last=False)
   cache.move_to_end(key);x,p,ratio,length=cache[key];query=float(s['query_mm'][row]);qx=tensor(np.clip((bank['features'][s['query_rows'][row]]-mean)/std,-20,20))[None]
   logits=model(qx,x,tensor([query/length]),ratio,tensor([length]),torch.zeros((1,len(p),13),device='cuda'),torch.ones((1,len(p)),device='cuda',dtype=torch.bool))[0]
   legal=torch.stack((p<=query,p>=query),-1);a,b=s['interpretation_offsets'][row:row+2];loss=owner_pair_loss(logits,legal,s['endpoint_ranges'][a:b]);assert torch.isfinite(loss);(loss/4).backward();loss_sum+=float(loss.detach())/4
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10.);opt.step();scheduler.step()
  if step%100==0:entry=dict(step=step,loss=loss_sum,seconds=time.perf_counter()-start);history.append(entry);print(entry,flush=True)
  if (step+1)%250==0:
   torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},mean=mean,std=std,completed_steps=step+1,protocol_sha256=digest(out/'protocol.json')),out/'last.tmp.pt');(out/'last.tmp.pt').replace(out/'last.pt')
 for bank in banks:bank['inp'].close();bank['features']._mmap.close()
 write(out/'receipt.json',dict(status='GRID_ENDPOINT_HEAD_TRAINED_NOT_PROMOTED',optimizer_steps=1000,fit_calls=1,DEV_reads=0,bindings=bindings,history=history,model_sha256=digest(out/'last.pt'),seconds=time.perf_counter()-start,full_output_evaluated=False))
if __name__=='__main__':main()
