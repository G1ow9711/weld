"""Learn segment validity from predicted endpoints; supervision stays in TRAIN."""
import json,pathlib,sys,time,hashlib,shutil,numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from weld_pipeline.server_span_proposals import build
from weld_pipeline.world_set_evaluation_285 import reference_segment,distance_to_primitive
base=ROOT/'server_rebuild';src=base/'continuous_head_v1';out=base/'span_head_v1';out.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();r=json.loads((src/'receipt.json').read_text())
for n,h in r['files'].items():assert sha(src/n)==h
protocol=dict(source_checkpoint_sha256=sha(src/'model.pt'),epochs=60,batch=1024,seed=20261010,lr=.001,threshold=.5,final_epoch_only=True,one_mm_same_role_peak_suppression=True,no_short_weld_filter=True,DEV_reads=0,parent_assembly=False)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copyfile(__file__,out/'trainer_snapshot.py');shutil.copyfile(ROOT/'weld_pipeline/server_span_proposals.py',out/'features_snapshot.py');start=time.perf_counter();pred=np.load(src/'TRAIN_query_outputs.npy',mmap_mode='r');cursor=0;allx=[];ally=[];groups=[];bindings={}
for ki,(key,binding) in enumerate(r['bindings'].items()):
 z=np.load(src/(key+'.npz'));meta=z['candidate_positions'];p=pred[cursor:cursor+len(meta)];cursor+=len(meta);cfile=base/'candidates_v8'/key/'candidates.json';assert sha(cfile)==binding['candidate_sha256'];cs=json.loads(cfile.read_text());rows,x=build(cs,meta,z['features'],p);tfile=base/'labels_v1'/key/'targets.json';assert sha(tfile)==binding['targets_sha256'];targets=json.loads(tfile.read_text());truth=[]
 for pi,parent in enumerate(targets['parents']):
  for ci,child in enumerate(parent['children']):truth.append((pi,ci,child,reference_segment(dict(world_representation=child['representation']))))
 labels=np.zeros(len(rows),np.float32);matches=[]
 for i,row in enumerate(rows):
  pr=reference_segment(dict(world_representation=row['representation']));ends=np.array(pr['points'])[[0,-1]];valid=[]
  for pi,ci,t,ref in truth:
   if row['weld_type']!=t['weld_type'] or pr['kind']!=t['original_kind']:continue
   err=float(np.linalg.norm(ends-np.array(ref['points'])[[0,-1]],axis=1).max())
   if err>3:continue
   whole=max(float(distance_to_primitive(pr['points'],ref).max()),float(distance_to_primitive(ref['points'],pr).max()))
   if whole<=3:valid.append([pi,ci,whole])
  if valid:labels[i]=1
  matches.append(valid)
 (out/(key+'_proposals.json')).write_text(json.dumps(rows));(out/(key+'_supervision.json')).write_text(json.dumps(matches));np.savez_compressed(out/(key+'.npz'),features=x,labels=labels);allx.append(x);ally.append(labels);groups.extend([ki]*len(labels));bindings[key]=dict(rows=len(rows),positive=int(labels.sum()),source_data_sha256=sha(src/(key+'.npz')),candidates_sha256=sha(cfile),targets_sha256=sha(tfile));print('PREPARED',key,bindings[key],flush=True)
x=np.concatenate(allx);y=np.concatenate(ally);mean=x.mean(0);scale=np.maximum(x.std(0),.01);np.savez(out/'normalization.npz',mean=mean,scale=scale);torch.manual_seed(protocol['seed']);torch.set_num_threads(2);X=torch.tensor((x-mean)/scale,device='cuda');Y=torch.tensor(y,device='cuda');m=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.SiLU(),torch.nn.Linear(128,64),torch.nn.SiLU(),torch.nn.Linear(64,1)).cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=((len(y)-Y.sum())/Y.sum()).clamp(max=20));fit=time.perf_counter();updates=0
with (out/'epochs.jsonl').open('w') as log:
 for ep in range(protocol['epochs']):
  total=0
  for ids in torch.randperm(len(y),device='cuda').split(protocol['batch']):
   opt.zero_grad();loss=lossfn(m(X[ids]).squeeze(1),Y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();updates+=1;total+=float(loss.detach())*len(ids)
  row=dict(epoch=ep+1,loss=total/len(y),updates=updates);log.write(json.dumps(row)+'\n');log.flush()
  if (ep+1)%10==0:print('FIT',row,flush=True)
torch.cuda.synchronize();fit_seconds=time.perf_counter()-fit;torch.save(dict(state_dict=m.state_dict(),input_dim=x.shape[1],protocol=protocol),out/'model.pt')
with torch.no_grad():scores=torch.sigmoid(m(X).squeeze(1)).cpu().numpy()
np.save(out/'TRAIN_scores.npy',scores);groups=np.array(groups);metrics={}
for ki,key in enumerate(bindings):
 mask=groups==ki;t=y[mask]>0;p=scores[mask]>=.5;metrics[key]=dict(proposal_TP=int((t&p).sum()),proposal_FP=int((~t&p).sum()),proposal_FN=int((t&~p).sum()),proposal_G=int(t.sum()))
receipt=dict(status='SPAN_COMPONENT_FIT_COMPLETE',bindings=bindings,metrics=metrics,updates=updates,fit_seconds=fit_seconds,total_seconds=time.perf_counter()-start,files={n:sha(out/n) for n in ['model.pt','normalization.npz','protocol.json','trainer_snapshot.py','features_snapshot.py']},full_contract_pass=False,metrics_are_proposals_not_one_to_one_weld_counts=True)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
