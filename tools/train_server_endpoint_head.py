"""Rebuild and fit an endpoint proposal head; NOT a weld extraction evaluation."""
import hashlib,json,pathlib,sys,time,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.server_geometry_events import event_features,geometry,controls
from weld_pipeline.server_finite_candidates import guard
KEYS=['21','BM79A_native_local','10','25','36'];OUT=ROOT/'server_rebuild/endpoint_head_v1'
OUT.mkdir(parents=True,exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=dict(component='geometry endpoint presence head',keys=KEYS,epochs=30,batch=1024,seed=20261010,lr=.001,weight_decay=.0001,threshold=.5,endpoint_label_radius_mm=3.,no_DEV_or_holdout_access=True,model_selection='final epoch only',full_weld_extraction=False,controls=controls())
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copyfile(ROOT/'weld_pipeline/server_geometry_events.py',OUT/'features_snapshot.py');shutil.copyfile(__file__,OUT/'trainer_snapshot.py')
torch.manual_seed(protocol['seed']);np.random.seed(protocol['seed']);torch.set_num_threads(2)
allx=[];ally=[];groups=[];bindings={};preparation=time.perf_counter()
with threadpool_limits(limits=2):
 for ki,key in enumerate(KEYS):
  base=ROOT/'server_rebuild';d=base/'candidates_v7'/key;r=json.loads((d/'receipt.json').read_text())
  for n,h in r['files'].items():assert sha(d/n)==h
  xf=base/'inputs_v1'/key/'xyz_mm.npy';assert sha(xf)==r['input_sha256'];xyz=np.load(xf,mmap_mode='r');tree=cKDTree(xyz)
  faces=json.loads((base/'candidates_v5'/key/'faces.json').read_text());cs=json.loads((d/'candidates.json').read_text());tf=base/'labels_v1'/key/'targets.json';targets=json.loads(tf.read_text());endpoints={'line':[],'arc':[]}
  for parent in targets['parents']:
   for child in parent['children']:
    length,point,_,_=geometry(child['representation']);endpoints[child['original_kind']].extend(point(np.array([0.,length])).tolist())
  gt={k:cKDTree(v) for k,v in endpoints.items() if v};xs=[];ys=[];meta=[]
  for ci,c in enumerate(cs):
   s,q,x=event_features(c,faces,tree);kind=c['representation']['kind'];y=(gt[kind].query(q)[0]<=3.).astype(np.float32) if kind in gt else np.zeros(len(q),np.float32)
   xs.append(x);ys.append(y);meta.extend([[ci,float(t)] for t in s])
  x=np.concatenate(xs);y=np.concatenate(ys);np.savez_compressed(OUT/(key+'.npz'),features=x,labels=y,candidate_positions=np.array(meta));allx.append(x);ally.append(y);groups.extend([ki]*len(y));bindings[key]=dict(input_sha256=sha(xf),candidates_sha256=sha(d/'candidates.json'),targets_sha256=sha(tf),rows=len(y),positive=int(y.sum()))
  print('PREPARED',key,len(y),'POSITIVE',int(y.sum()),flush=True);guard()
x=np.concatenate(allx);y=np.concatenate(ally);groups=np.array(groups);mean=x.mean(0);scale=np.maximum(x.std(0),.01);np.savez(OUT/'normalization.npz',mean=mean,scale=scale)
X=torch.tensor((x-mean)/scale,device='cuda');Y=torch.tensor(y,device='cuda');assert Y.sum()>0
model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.SiLU(),torch.nn.Linear(128,64),torch.nn.SiLU(),torch.nn.Linear(64,1)).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay']);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=(len(y)-Y.sum())/Y.sum());fit=time.perf_counter();updates=0
with (OUT/'epochs.jsonl').open('w') as log:
 for epoch in range(protocol['epochs']):
  model.train();order=torch.randperm(len(y),device='cuda');total=0.
  for ids in order.split(protocol['batch']):
   optimizer.zero_grad();loss=lossfn(model(X[ids]).squeeze(1),Y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(ids);updates+=1
  row=dict(epoch=epoch+1,loss=total/len(y),updates=updates);log.write(json.dumps(row)+'\n');log.flush();print('FIT',row,flush=True)
torch.cuda.synchronize();fit_seconds=time.perf_counter()-fit;torch.save(dict(state_dict=model.state_dict(),input_dim=x.shape[1],protocol=protocol),OUT/'model.pt');model.eval()
with torch.no_grad():scores=torch.sigmoid(model(X).squeeze(1)).cpu().numpy()
np.save(OUT/'training_event_scores.npy',scores);metrics={}
for ki,key in enumerate(KEYS):
 mask=groups==ki;truth=y[mask]>0;pred=scores[mask]>=.5;tp=int((truth&pred).sum());fp=int((~truth&pred).sum());fn=int((truth&~pred).sum());metrics[key]=dict(event_TP=tp,event_FP=fp,event_FN=fn,event_G=int(truth.sum()),rows=int(mask.sum()))
r=dict(status='ENDPOINT_COMPONENT_FIT_COMPLETE',optimizer_updates=updates,fit_seconds=fit_seconds,preparation_and_fit_seconds=time.perf_counter()-preparation,bindings=bindings,metrics=metrics,checkpoint_sha256=sha(OUT/'model.pt'),files={n:sha(OUT/n) for n in ['model.pt','normalization.npz','features_snapshot.py','trainer_snapshot.py','protocol.json']},limitations=['TRAIN event classification only; duplicated endpoints are not weld counts','No material contact classifier, segment pairing, parent assembly, orientation or attribute validation','No independent accuracy or runtime acceptance'],full_contract_pass=False)
(OUT/'receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
