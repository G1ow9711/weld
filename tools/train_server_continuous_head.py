"""Continuous endpoint/occupancy component, all supervision confined to TRAIN."""
import hashlib,json,pathlib,sys,time,shutil
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from weld_pipeline.server_geometry_events import event_features,geometry,events
from weld_pipeline.server_finite_candidates import guard
from weld_pipeline.world_set_evaluation_285 import reference_segment,distance_to_primitive
KEYS=['21','BM79A_native_local','10','25','36'];OUT=ROOT/'server_rebuild/continuous_head_v1';OUT.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=dict(component='continuous endpoint offset and interval occupancy',TRAIN=KEYS,epochs=24,batch=2048,lr=.001,weight_decay=.0001,seed=20261010,query_spacing_mm=2.,positive_query_radius_mm=1.,endpoint_roles=['H_start','H_end','V_start','V_end'],loss='weighted BCE four endpoint roles + positive-only SmoothL1 offsets + weighted BCE H/V occupancy',endpoint_positive_weight_cap=100.,occupancy_positive_weight_cap=20.,fixed_threshold=.5,final_epoch_only=True,DEV_access=False,full_weld_extraction=False)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copyfile(__file__,OUT/'trainer_snapshot.py');shutil.copyfile(ROOT/'weld_pipeline/server_geometry_events.py',OUT/'features_snapshot.py')
start=time.perf_counter();bindings={};allx=[];ally=[];allo=[];allw=[];groups=[]
with threadpool_limits(limits=2):
 for ki,key in enumerate(KEYS):
  base=ROOT/'server_rebuild';src=base/'candidates_v8'/key;r=json.loads((src/'receipt.json').read_text())
  for n,h in r['files'].items():assert sha(src/n)==h
  xf=base/'inputs_v1'/key/'xyz_mm.npy';assert sha(xf)==r['input_sha256'];tree=cKDTree(np.load(xf,mmap_mode='r'));faces=json.loads((base/'candidates_v5'/key/'faces.json').read_text());cs=json.loads((src/'candidates.json').read_text());tf=base/'labels_v1'/key/'targets.json';targets=json.loads(tf.read_text());truth=[]
  for p in targets['parents']:
   for child in p['children']:
    L,pt,_,arc=geometry(child['representation']);truth.append((child['original_kind'],child['weld_type'],pt(np.array([0,L])),reference_segment(dict(world_representation=child['representation']))))
  xs=[];ys=[];offsets=[];ws=[];metadata=[]
  for ci,c in enumerate(cs):
   length,point,tangent,arc=geometry(c['representation']);s=np.unique(np.r_[events(c,faces),np.linspace(0,length,int(np.ceil(length/2))+1)]);_,q,x=event_features(c,faces,tree,s)
   support=np.array(c.get('support_intervals_mm',[[0,length]]));ds=np.maximum(support[:,0,None]-s,0)+np.maximum(s-support[:,1,None],0);d=ds.min(0);x=np.c_[x,(d<=1e-6).astype(float),np.log1p(d)].astype(np.float32)
   y=np.zeros((len(s),4),np.float32);off=np.zeros_like(y);best=np.full_like(y,np.inf);occ=np.zeros((len(s),2),np.float32)
   pred=reference_segment(dict(world_representation=c['representation']))
   for kind,wt,ends,ref in truth:
    if kind!=c['representation']['kind']:continue
    typ=int(wt=='vertical')
    # Supervised occupancy lives here, never in input feature construction.
    bounds=np.array(ref['points'])[[0,-1]]
    if kind=='line':
     direction=tangent(0.);tt=(ends-point(0.))@direction;perp=np.linalg.norm(ends-(point(0.)+tt[:,None]*direction),axis=1)
     if perp.max()>3:continue
     occ[:,typ]=np.maximum(occ[:,typ],((s>=min(tt))&(s<=max(tt))).astype(np.float32))
    else:
     radial=ends-arc.center_mm;angles=np.arctan2(radial@np.cross(arc.plane_normal,arc.start_direction),radial@arc.start_direction)*np.sign(arc.sweep_rad);tt=(angles%(2*np.pi))*arc.radius_mm
     candidates=np.stack([tt-2*np.pi*arc.radius_mm,tt,tt+2*np.pi*arc.radius_mm]);outside=np.maximum(-candidates,0)+np.maximum(candidates-length,0);tt=candidates[outside.argmin(0),np.arange(2)]
     if np.min(distance_to_primitive(ends,pred))>3:continue
     occ[:,typ]=np.maximum(occ[:,typ],(distance_to_primitive(q,ref)<=3).astype(np.float32))
    for ei,t in enumerate(tt):
     if distance_to_primitive(ends[ei:ei+1],pred)[0]>3:continue
     role=2*typ+ei;diff=t-s;valid=(abs(diff)<=1.)&(abs(diff)<best[:,role]);y[valid,role]=1;off[valid,role]=diff[valid];best[valid,role]=abs(diff[valid])
   xs.append(x);ys.append(y);offsets.append(off);ws.append(occ);metadata.extend([[ci,float(t)] for t in s])
  x=np.concatenate(xs);y=np.concatenate(ys);off=np.concatenate(offsets);occ=np.concatenate(ws);np.savez_compressed(OUT/(key+'.npz'),features=x,labels=y,offsets=off,occupancy=occ,candidate_positions=np.array(metadata));allx.append(x);ally.append(y);allo.append(off);allw.append(occ);groups.extend([ki]*len(x));bindings[key]=dict(input_sha256=sha(xf),candidate_sha256=sha(src/'candidates.json'),targets_sha256=sha(tf),rows=len(x),endpoint_positive=y.sum(0).tolist(),occupancy_positive=occ.sum(0).tolist());print('PREPARED',key,bindings[key],flush=True);guard()
x=np.concatenate(allx);y=np.concatenate(ally);off=np.concatenate(allo);occ=np.concatenate(allw);mean=x.mean(0);scale=np.maximum(x.std(0),.01);np.savez(OUT/'normalization.npz',mean=mean,scale=scale);torch.manual_seed(protocol['seed']);torch.set_num_threads(2)
X=torch.tensor((x-mean)/scale,device='cuda');Y=torch.tensor(y,device='cuda');O=torch.tensor(off,device='cuda');W=torch.tensor(occ,device='cuda');model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.SiLU(),torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,10)).cuda();opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay']);pos=(len(y)-Y.sum(0))/Y.sum(0).clamp_min(1);wp=(len(occ)-W.sum(0))/W.sum(0).clamp_min(1);bce=torch.nn.BCEWithLogitsLoss(pos_weight=pos.clamp(max=100));wbce=torch.nn.BCEWithLogitsLoss(pos_weight=wp.clamp(max=20));fit=time.perf_counter();updates=0
with (OUT/'epochs.jsonl').open('w') as log:
 for epoch in range(protocol['epochs']):
  order=torch.randperm(len(y),device='cuda');tot=np.zeros(3)
  for ids in order.split(protocol['batch']):
   opt.zero_grad();p=model(X[ids]);a=bce(p[:,:4],Y[ids]);b=(torch.nn.functional.smooth_l1_loss(p[:,4:8],O[ids],reduction='none')*Y[ids]).sum()/Y[ids].sum().clamp_min(1);c=wbce(p[:,8:],W[ids]);loss=a+b+c;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();tot+=np.array([float(a.detach()),float(b.detach()),float(c.detach())])*len(ids);updates+=1
  row=dict(epoch=epoch+1,losses=(tot/len(y)).tolist(),updates=updates);log.write(json.dumps(row)+'\n');log.flush();print('FIT',row,flush=True)
torch.cuda.synchronize();fit_seconds=time.perf_counter()-fit;torch.save(dict(state_dict=model.state_dict(),input_dim=x.shape[1],protocol=protocol),OUT/'model.pt');model.eval()
with torch.no_grad():pred=torch.cat([model(b) for b in X.split(4096)]).cpu().numpy()
np.save(OUT/'TRAIN_query_outputs.npy',pred);groups=np.array(groups);metrics={}
for ki,key in enumerate(KEYS):
 mask=groups==ki;truth=y[mask]>0;pr=pred[mask,:4]>=0;delta=abs(pred[mask,4:8]-off[mask]);metrics[key]=dict(query_TP=int((truth&pr).sum()),query_FP=int((~truth&pr).sum()),query_FN=int((truth&~pr).sum()),query_G=int(truth.sum()),positive_offset_mae_mm=float(delta[truth].mean()),positive_offset_max_mm=float(delta[truth].max()))
r=dict(status='CONTINUOUS_COMPONENT_FIT_COMPLETE',bindings=bindings,fit_seconds=fit_seconds,total_seconds=time.perf_counter()-start,updates=updates,metrics=metrics,files={n:sha(OUT/n) for n in ['model.pt','normalization.npz','protocol.json','trainer_snapshot.py','features_snapshot.py']},full_weld_extraction_evaluated=False,limitations=['query statistics count duplicated carriers, not welds','pairing, parent grouping, material identity and attributes remain unvalidated'])
(OUT/'receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
