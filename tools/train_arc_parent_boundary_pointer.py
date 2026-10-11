"""Conditional TRAIN-only parent boundary learning; no promotion or rejection claim."""
from pathlib import Path
import time
import numpy as np
import torch
from weld_pipeline.server_arc_boundary_pointer import ArcBoundaryPointer
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main():
    started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1031);rng=np.random.default_rng(1031);key='new_ac2008a03607'
    features=BASE/'training_arc_event_features_v1'/key;fr=read(features/'receipt.json');assert fr['status']=='INPUT_ARC_EVENT_FEATURES_FROZEN';assert digest(features/'protocol.json')==fr['protocol_sha256'];fp=read(features/'protocol.json')
    labels=BASE/'training_arc_parent_boundaries_v1';lr=read(labels/'receipt.json');assert digest(labels/'labels.json')==lr['labels_sha256'];supervision=read(labels/'labels.json')
    events=BASE/'training_arc_vertex_events_v1'/key;assert supervision['event_receipt_sha256']==fp['event_receipt_sha256']==digest(events/'receipt.json')
    fmap={f['face']:f for f in fr['faces']};banks={};cases=[];groups=[]
    for parent in supervision['rows']:
        group=[]
        for record in parent['labels']:
            if not record['paths'] or any(record['missing_boundaries']):continue
            face=record['face'];index=record['candidate']
            if face not in banks:
                path=features/fmap[face]['file'];assert digest(path)==fmap[face]['sha256'];ep=events/f'face_{face}.npz';assert digest(ep)==fmap[face]['event_file_sha256']
                with np.load(ep) as z:banks[face]=(np.load(path,mmap_mode='r'),z['offsets'],z['positions_mm'])
            x,offsets,q=banks[face];a,b=offsets[index:index+2];path=record['paths'][0];positive=record['boundary_event_indices'];teacher=path['event_indices']
            if path['direction_sign']<0:positive=positive[::-1];teacher=teacher[::-1]
            group.append(len(cases));cases.append(dict(parent=parent['parent'],face=face,candidate=index,x=np.array(x[a:b]),q=np.array(q[a:b]),positive=[(np.asarray(v)-a).tolist() for v in positive],teacher=(np.asarray(teacher)-a).tolist()))
        if group:groups.append(group)
    assert len(groups)==4 and cases
    weights=np.zeros(len(cases))
    for group in groups:weights[group]=1/len(groups)/len(group)
    mean=sum(w*c['x'].mean(0,dtype=np.float64) for w,c in zip(weights,cases));second=sum(w*np.square(c['x'].astype(np.float64)).mean(0) for w,c in zip(weights,cases));std=np.maximum(np.sqrt(np.maximum(second-mean**2,0)),1e-6)
    out=BASE/'arc_parent_boundary_pointer_v1';out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(key=key,parents=len(groups),labelled_carriers=len(cases),steps=4000,seed=1031,batch_size=1,sampling='Equal parent then equal compatible carrier',lr=.001,lr_end=.0001,feature_receipt_sha256=digest(features/'receipt.json'),label_receipt_sha256=digest(labels/'receipt.json'),module_sha256=digest('weld_pipeline/server_arc_boundary_pointer.py'),scope='Conditional one-TRAIN-family mechanics experiment. Carriers selected with TRAIN supervision; no negative/rejection training, runtime candidate selection, type/frame prediction or generalization claim.',DEV_reads=0,heldout_reads=0))
    for c in cases:c['features']=torch.as_tensor(np.clip((c['x']-mean)/std,-20,20),dtype=torch.float32,device='cuda');c['positions']=torch.as_tensor(c['q'],dtype=torch.float64,device='cuda')
    model=ArcBoundaryPointer().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000,eta_min=.0001);history=[]
    for step in range(4000):
        group=groups[int(rng.integers(len(groups)))];c=cases[group[int(rng.integers(len(group)))]]
        loss=model.set_teacher_loss(c['features'],c['positions'],c['positive'],c['teacher']);assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();schedule.step()
        if step%250==0:history.append(dict(step=step,loss=float(loss.detach())));print(history[-1],flush=True)
    model.eval();predictions=[]
    for c in cases:predictions.append(dict(parent=c['parent'],face=c['face'],candidate=c['candidate'],selected_local_event_indices=model.decode(c['features'],c['positions'])))
    torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,protocol_sha256=digest(out/'protocol.json')),out/'model.pt')
    write(out/'conditional_train_predictions.json',dict(rows=predictions,scope='Frozen predictions on supervised TRAIN carrier list only; no reference-based endpoint correction. Not deployment candidate selection.'))
    write(out/'receipt.json',dict(status='CONDITIONAL_TRAIN_POINTER_NOT_PROMOTED',model_sha256=digest(out/'model.pt'),prediction_sha256=digest(out/'conditional_train_predictions.json'),history=history,fit_calls=1,DEV_reads=0,heldout_reads=0,seconds=time.perf_counter()-started))

if __name__=='__main__':main()
