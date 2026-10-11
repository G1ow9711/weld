"""Masked typed-attribute baseline on eight TRAIN frozen predicted-endpoint inputs."""
from pathlib import Path
import time,argparse
import numpy as np
import torch
from weld_pipeline.server_transverse_attribute_head import TransverseAttributeHead
from weld_pipeline.server_endpoint_attribute_supervision import masked_endpoint_loss
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1';PARSER=argparse.ArgumentParser();PARSER.add_argument('--transverse-context',action='store_true');ARGS=PARSER.parse_args();OUT=BASE/('transverse_attribute_head_v1' if ARGS.transverse_context else 'transverse_attribute_control_v1')
def main():
 controls=read(BASE/"transverse_attribute_initialization_controls_v1/receipt.json");assert controls["status"]=="PASS" and controls["module_sha256"]==digest("weld_pipeline/server_transverse_attribute_head.py")
 predecessor=BASE/"predicted_endpoint_attribute_head_v1";previous=read(predecessor/"receipt.json");assert digest(predecessor/"model.pt")==previous["model_sha256"];warm=torch.load(predecessor/"model.pt",map_location="cpu",weights_only=False)
 if ARGS.transverse_context:assert read(BASE/"transverse_endpoint_training_context_v1/receipt.json")["status"]=="EIGHT_TRAIN_TRANSVERSE_ENDPOINT_INPUT_CONTEXT_READY"
 start=time.perf_counter();OUT.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(1031);labelroot=BASE/'predicted_endpoint_attribute_labels_v1';assert read(labelroot/'receipt.json')['status']=='EIGHT_TRAIN_PREDICTED_ENDPOINT_ATTRIBUTE_LABELS_READY';keys=read(labelroot/'protocol.json')['keys'];banks=[];states=[];values=[];masks=[];numbers=[];weights=[];spans={};bindings={};n=0
 for key in keys:
  f=BASE/'predicted_endpoint_attribute_features_v1'/key;l=labelroot/key;r=read(l/'receipt.json');assert digest(f/'receipt.json')==r['feature_receipt_sha256'];fr=read(f/'receipt.json');assert digest(f/'features.npy')==fr['features_sha256'] and digest(l/'labels.npz')==r['labels_sha256'];x=np.load(f/'features.npy');context_receipt=None
  if ARGS.transverse_context:
   context=BASE/'transverse_endpoint_training_context_v1'/key;ctx=read(context/'receipt.json');assert ctx['source_feature_receipt_sha256']==digest(f/'receipt.json') and digest(context/'features.npy')==ctx['features_sha256'];extra=np.load(context/'features.npy');context_receipt=digest(context/'receipt.json')
  else:extra=np.zeros((len(x),86),np.float32)
  assert extra.shape==(len(x),86);x=np.column_stack((x,extra));z=np.load(l/'labels.npz');assert len(x)==len(z['state']);banks.append(x);states.append(z['state']);values.append(z['exact_mm']);masks.append(z['state_mask']);numbers.append(z['numeric_mask']);weights.append(z['row_weight']/len(keys));spans[key]=(n,n+len(x));n+=len(x);bindings[key]=dict(features=digest(f/'receipt.json'),labels=digest(l/'receipt.json'),transverse_context=context_receipt)
 x=np.concatenate(banks);y=np.concatenate(states);v=np.concatenate(values);sm=np.concatenate(masks);nm=np.concatenate(numbers);w=np.concatenate(weights);assert abs(w.sum()-1)<1e-8 and np.isfinite(v[nm]).all();mean=np.sum(x*w[:,None],axis=0).astype(np.float32);std=np.maximum(np.sqrt(np.sum((x-mean)**2*w[:,None],axis=0)),1e-6).astype(np.float32);mean[:411]=warm['mean'];std[:411]=warm['std'];normal=np.clip((x-mean)/std,-20,20).astype(np.float32);tensor=lambda a,dtype=torch.float32:torch.as_tensor(a,device='cuda',dtype=dtype);xx=tensor(normal);yy=tensor(y,torch.long);vv=tensor(v);ss=tensor(sm,torch.bool);nn=tensor(nm,torch.bool);ww=tensor(w);model=TransverseAttributeHead().cuda();model.load_predecessor_state_dict(warm['state_dict']);means=[]
 for field in range(8):
  use=nm[:,field];mass=w[use].sum();assert mass>0;means.append(float(np.sum(w[use]*v[use,field])/mass))
 protocol=dict(keys=keys,steps=8000,batch_size=256,seed=1031,learning_rate=.001,learning_rate_end=.0001,input_columns=497,transverse_context=ARGS.transverse_context,warm_attribute_model_sha256=previous['model_sha256'],old411_normalization_frozen=True,encoder_frozen=True,source_input_protocol_sha256=digest(BASE/'predicted_endpoint_attribute_features_v1/protocol.json'),sampling='Equal family then equal known-attribute child among geometry-matched TRAIN queries',loss='Existing independently masked per-field state CE plus exact numeric SmoothL1(mm)/50; sampling handles family/child weights once',numeric_output='Positive softplus, warm state inherited unchanged; new86 input weights start zero',numeric_TRAIN_means_mm=means,required_unknown_masked=True,optional_masked=True,scope='Conditional TRAIN attribute baseline on frozen expanded8 endpoints; not full-chain acceptance, pose/frame absent',heldout_reads=0,DEV_reads=0);write(OUT/'protocol.json',protocol);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,8000,eta_min=.0001);history=[]
 for step in range(8000):
  ids=torch.multinomial(ww,256,replacement=True);logits,pred=model(xx[ids]);state_loss,numeric_loss,_=masked_endpoint_loss(logits,pred,yy[ids],vv[ids],ss[ids],nn[ids]);loss=state_loss+numeric_loss/50.;assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();schedule.step()
  if step%500==0:history.append(dict(step=step,state_loss=float(state_loss.detach()),numeric_loss_mm=float(numeric_loss.detach())));print(history[-1],flush=True)
 model.eval();torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,keys=keys,source_endpoint_model_sha256=read(BASE/'predicted_endpoint_attribute_features_v1/protocol.json')['endpoint_model_sha256'],protocol_sha256=digest(OUT/'protocol.json')),OUT/'model.pt');model.cuda();reports=[]
 with torch.no_grad():
  for key,(a,b) in spans.items():
   logits=[];predictions=[]
   for i in range(a,b,1024):
    l,p=model(xx[i:min(i+1024,b)]);logits.append(l.cpu().numpy());predictions.append(p.cpu().numpy())
   scores=np.concatenate(logits);pred=np.concatenate(predictions);np.savez(OUT/(key+'_training_predictions.npz'),state_logits=scores,exact_mm=pred);chosen=scores.argmax(2);fields=[]
   for j in range(8):
    known=sm[a:b,j];exact=nm[a:b,j];wk=w[a:b]*known;wn=w[a:b]*exact;error=abs(pred[:,j]-v[a:b,j]);fields.append(dict(field=j,known=int(known.sum()),exact=int(exact.sum()),weighted_state_accuracy=float(np.sum(wk*(chosen[:,j]==y[a:b,j]))/wk.sum()) if wk.sum() else None,weighted_exact_MAE_mm=float(np.sum(wn[exact]*error[exact])/wn.sum()) if wn.sum() else None,exact_error_quantiles_mm=np.quantile(error[exact],[.5,.95,1]).tolist() if exact.any() else []))
   reports.append(dict(key=key,fields=fields))
 write(OUT/'receipt.json',dict(status='EXPERIMENTAL_ATTRIBUTE_HEAD_TRAIN_FIT_ONLY_NOT_PROMOTED',model_sha256=digest(OUT/'model.pt'),bindings=bindings,training=reports,history=history,fit_calls=1,optimizer_steps=8000,heldout_reads=0,DEV_reads=0,source_sha256=digest(__file__),module_sha256=digest('weld_pipeline/server_transverse_attribute_head.py'),seconds=time.perf_counter()-start))
if __name__=='__main__':main()
