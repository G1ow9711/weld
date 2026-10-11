"""Conditional carrier boundary extension of preserved trained pointer."""
from pathlib import Path
import time
import numpy as np
import torch
from weld_pipeline.server_arc_boundary_feedback_pointer import ArcBoundaryFeedbackPointer
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1031);rng=np.random.default_rng(1031);key='new_fcaf2af1abee';features=BASE/'curved_probe_event_features_v1'/key;fr=read(features/'receipt.json');assert digest(features/'protocol.json')==fr['protocol_sha256'];assert digest(features/'features.npy')==fr['feature_sha256'];fp=read(features/'protocol.json');labels=BASE/'curved_probe_boundary_sequences_v1';lr=read(labels/'receipt.json');assert digest(labels/'labels.json')==lr['labels_sha256'];supervision=read(labels/'labels.json');events=BASE/'curved_probe_vertex_events_v1'/key;assert supervision['event_receipt_sha256']==fp['event_receipt_sha256']==digest(events/'receipt.json');er=read(events/'receipt.json');assert digest(events/'events.npz')==er['events_sha256']
    with np.load(events/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm']
    x=np.load(features/'features.npy',mmap_mode='r');cases=[]
    for row in supervision['rows']:
        a,b=offsets[row['profile']:row['profile']+2];cases.append(dict(profile=row['profile'],x=np.array(x[a:b]),q=np.array(positions[a:b]),positive=[(np.asarray(ids)-a).tolist() for ids in row['boundary_event_indices']],teacher=(np.asarray(row['teacher_event_indices'])-a).tolist()))
    warm_folder=BASE/'arc_parent_feedback_model_valid_v1';wr=read(warm_folder/'receipt.json');assert digest(warm_folder/'model.pt')==wr['model_sha256'];warm=torch.load(warm_folder/'model.pt',map_location='cpu',weights_only=False)
    extra_mean=np.mean([c['x'][:,76:].mean(0,dtype=np.float64) for c in cases],axis=0);extra_second=np.mean([np.square(c['x'][:,76:].astype(float)).mean(0) for c in cases],axis=0);extra_std=np.maximum(np.sqrt(np.maximum(extra_second-extra_mean**2,0)),1e-6);mean=np.r_[warm['mean'],extra_mean];std=np.r_[warm['std'],extra_std]
    out=BASE/'curved_probe_boundary_pointer_v1';out.mkdir(exist_ok=False);write(out/'protocol.json',dict(steps=4000,seed=1031,lr=.001,lr_end=.0001,sampling='Equal each of6 compatible TRAIN carriers; no reference negative/rejection labels',feature_receipt_sha256=digest(features/'receipt.json'),label_receipt_sha256=digest(labels/'receipt.json'),warm_model_sha256=wr['model_sha256'],module_sha256=digest('weld_pipeline/server_arc_boundary_feedback_pointer.py'),base_module_sha256=digest('weld_pipeline/server_arc_boundary_pointer.py'),trainer_sha256=digest(__file__),input_dim=83,old_normalization_preserved=True,new_encoder_columns_initially_zero=True,scope='Conditional one-family boundary prediction experiment on selected carriers with multiple known independent parents. No learned parent relation/type/contact/existence and no claim of preserved other-family performance. Original models retained.',DEV_reads=0))
    model=ArcBoundaryFeedbackPointer(feedback_mode='model_valid',input_dim=83).cuda();state=warm['state_dict'];oldweight=state['encoder.0.weight'];state['encoder.0.weight']=torch.cat((oldweight,torch.zeros(oldweight.shape[0],7,dtype=oldweight.dtype)),dim=1);model.load_state_dict(state)
    for c in cases:c['features']=torch.as_tensor(np.clip((c['x']-mean)/std,-20,20),dtype=torch.float32,device='cuda');c['positions']=torch.as_tensor(c['q'],dtype=torch.float64,device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,4000,eta_min=.0001);history=[]
    for step in range(4000):
        c=cases[int(rng.integers(len(cases)))];loss=model.set_teacher_loss(c['features'],c['positions'],c['positive'],c['teacher']);assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();schedule.step()
        if step%250==0:history.append(dict(step=step,loss=float(loss.detach())));print(history[-1],flush=True)
    model.eval();predictions=[]
    for c in cases:
        selected=model.decode(c['features'],c['positions']);correct=len(selected)==len(c['positive']) and all(i in ids for i,ids in zip(selected,c['positive']));predictions.append(dict(profile=c['profile'],selected_local_event_indices=selected,all_boundaries_within_label_sets=correct))
    torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,protocol_sha256=digest(out/'protocol.json')),out/'model.pt');write(out/'conditional_predictions.json',dict(rows=predictions,scope='Predicted boundaries on reference-selected TRAIN carriers, no endpoint snapping. No deployment selection or parent assembly.'));write(out/'receipt.json',dict(status='CONDITIONAL_CURVED_POINTER_NOT_PROMOTED',model_sha256=digest(out/'model.pt'),predictions_sha256=digest(out/'conditional_predictions.json'),history=history,fit_calls=1,seconds=time.perf_counter()-started,DEV_reads=0))

if __name__=='__main__':main()
