"""Matched raw vs numerically grouped event continuation from identical weights."""
from pathlib import Path
import time,argparse
import numpy as np
import torch
from weld_pipeline.server_arc_boundary_feedback_pointer import ArcBoundaryFeedbackPointer
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main(mode):
    started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1031);rng=np.random.default_rng(1031);key='new_fcaf2af1abee';warm_folder=BASE/'curved_probe_boundary_pointer_v1';wr=read(warm_folder/'receipt.json');assert digest(warm_folder/'model.pt')==wr['model_sha256'];warm=torch.load(warm_folder/'model.pt',map_location='cpu',weights_only=False);warm_protocol=read(warm_folder/'protocol.json');features=BASE/'curved_probe_event_features_v1'/key;fr=read(features/'receipt.json');assert digest(features/'receipt.json')==warm_protocol['feature_receipt_sha256']
    if mode=='raw':
        folder=BASE/'curved_probe_vertex_events_v1'/key;er=read(folder/'receipt.json');assert digest(folder/'events.npz')==er['events_sha256'];assert digest(features/'features.npy')==fr['feature_sha256'];x=np.load(features/'features.npy',mmap_mode='r');label_folder=BASE/'curved_probe_boundary_sequences_v1'
    else:
        folder=BASE/'curved_probe_event_groups_v1'/key;er=read(folder/'receipt.json');assert digest(folder/'events.npz')==er['files']['events.npz'];assert digest(folder/'mean.npy')==er['files']['mean.npy'];assert read(folder/'protocol.json')['feature_receipt_sha256']==digest(features/'receipt.json');x=np.load(folder/'mean.npy',mmap_mode='r');label_folder=BASE/'curved_group_boundary_sequences_v1'
    lr=read(label_folder/'receipt.json');assert digest(label_folder/'labels.json')==lr['labels_sha256'];labels=read(label_folder/'labels.json')
    with np.load(folder/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm']
    cases=[];mean,std=warm['mean'],warm['std']
    for row in labels['rows']:
        a,b=offsets[row['profile']:row['profile']+2];cases.append(dict(profile=row['profile'],features=torch.as_tensor(np.clip((x[a:b]-mean)/std,-20,20),dtype=torch.float32,device='cuda'),positions=torch.as_tensor(positions[a:b],dtype=torch.float64,device='cuda'),positive=[(np.asarray(ids)-a).tolist() for ids in row['boundary_event_indices']],teacher=(np.asarray(row['teacher_event_indices'])-a).tolist()))
    out=BASE/('curved_group_comparison_'+mode+'_v1');out.mkdir(exist_ok=False);write(out/'protocol.json',dict(mode=mode,steps=4000,seed=1031,lr=.001,lr_end=.0001,batch_size=1,sampling='Equal each of same6 carrier IDs, same RNG order',feature_receipt_sha256=digest(features/'receipt.json'),event_receipt_sha256=digest(folder/'receipt.json'),label_receipt_sha256=digest(label_folder/'receipt.json'),warm_model_sha256=wr['model_sha256'],normalization_frozen=True,input_dim=83,module_sha256=digest('weld_pipeline/server_arc_boundary_feedback_pointer.py'),base_module_sha256=digest('weld_pipeline/server_arc_boundary_pointer.py'),trainer_sha256=digest(__file__),scope='Matched conditional TRAIN continuation; raw vs mean pooled numerical groups. No added candidate rejection/contact/type labels or full acceptance claim.',DEV_reads=0))
    model=ArcBoundaryFeedbackPointer(feedback_mode='model_valid',input_dim=83).cuda();model.load_state_dict(warm['state_dict']);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000,eta_min=.0001);history=[]
    for step in range(4000):
        c=cases[int(rng.integers(len(cases)))];loss=model.set_teacher_loss(c['features'],c['positions'],c['positive'],c['teacher']);assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();schedule.step()
        if step%500==0:history.append(dict(step=step,loss=float(loss.detach())));print(mode,history[-1],flush=True)
    model.eval();predictions=[]
    for c in cases:
        path=model.decode(c['features'],c['positions']);predictions.append(dict(profile=c['profile'],selected_local_event_indices=path,all_boundaries_within_label_sets=len(path)==len(c['positive']) and all(i in ids for i,ids in zip(path,c['positive']))))
    torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,protocol_sha256=digest(out/'protocol.json')),out/'model.pt');write(out/'conditional_predictions.json',dict(rows=predictions));write(out/'receipt.json',dict(status='MATCHED_CURVED_GROUP_POINTER_NOT_PROMOTED',model_sha256=digest(out/'model.pt'),predictions_sha256=digest(out/'conditional_predictions.json'),history=history,seconds=time.perf_counter()-started,fit_calls=1,DEV_reads=0))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['raw','grouped'],required=True);main(parser.parse_args().mode)
