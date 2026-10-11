"""Two-family balanced rehearsal; retain source models and full diagnostics."""
from pathlib import Path
import time
import numpy as np
import torch
from weld_pipeline.server_arc_boundary_feedback_pointer import ArcBoundaryFeedbackPointer
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def first_family():
    key='new_ac2008a03607';features=BASE/'training_arc_event_features_v1'/key;events=BASE/'training_arc_vertex_events_v1'/key;labels=BASE/'training_arc_parent_boundaries_v1';fr=read(features/'receipt.json');assert digest(features/'protocol.json')==fr['protocol_sha256'];lr=read(labels/'receipt.json');assert digest(labels/'labels.json')==lr['labels_sha256'];supervision=read(labels/'labels.json');assert supervision['event_receipt_sha256']==digest(events/'receipt.json')==read(features/'protocol.json')['event_receipt_sha256'];fmap={r['face']:r for r in fr['faces']};banks={};cases=[];groups=[]
    for parent in supervision['rows']:
        group=[]
        for record in parent['labels']:
            if not record['paths'] or any(record['missing_boundaries']):continue
            face=record['face'];index=record['candidate']
            if face not in banks:
                path=features/fmap[face]['file'];assert digest(path)==fmap[face]['sha256'];eventpath=events/f'face_{face}.npz';assert digest(eventpath)==fmap[face]['event_file_sha256']
                with np.load(eventpath) as z:banks[face]=(np.load(path,mmap_mode='r'),z['offsets'],z['positions_mm'],z['vertex_indices'])
            x,offsets,positions,ids=banks[face];a,b=offsets[index:index+2];q=positions[a:b];extra=np.column_stack((np.zeros((b-a,2)),ids[a:b]>=0,np.ones(b-a),np.log1p(q),np.log1p(q[-1]-q),np.zeros(b-a)));path=record['paths'][0];positive=record['boundary_event_indices'];teacher=path['event_indices']
            if path['direction_sign']<0:positive=positive[::-1];teacher=teacher[::-1]
            group.append(len(cases));cases.append(dict(family='first',parent=parent['parent'],face=face,candidate=index,x=np.column_stack((x[a:b],extra)),q=np.array(q),positive=[(np.asarray(v)-a).tolist() for v in positive],teacher=(np.asarray(teacher)-a).tolist()))
        groups.append(group)
    return cases,groups,dict(features=digest(features/'receipt.json'),labels=digest(labels/'receipt.json'),events=digest(events/'receipt.json'))


def second_family():
    key='new_fcaf2af1abee';folder=BASE/'curved_probe_event_groups_v1'/key;fr=read(folder/'receipt.json');assert digest(folder/'mean.npy')==fr['files']['mean.npy'];assert digest(folder/'events.npz')==fr['files']['events.npz'];labels=BASE/'curved_group_boundary_sequences_v1';lr=read(labels/'receipt.json');assert digest(labels/'labels.json')==lr['labels_sha256'];supervision=read(labels/'labels.json');assert supervision['group_receipt_sha256']==digest(folder/'receipt.json');x=np.load(folder/'mean.npy',mmap_mode='r')
    with np.load(folder/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm']
    cases=[]
    for row in supervision['rows']:
        a,b=offsets[row['profile']:row['profile']+2];cases.append(dict(family='second',profile=row['profile'],x=np.array(x[a:b]),q=np.array(positions[a:b]),positive=[(np.asarray(ids)-a).tolist() for ids in row['boundary_event_indices']],teacher=(np.asarray(row['teacher_event_indices'])-a).tolist()))
    return cases,[[i] for i in range(len(cases))],dict(groups=digest(folder/'receipt.json'),labels=digest(labels/'receipt.json'))


def main():
    started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(1031);rng=np.random.default_rng(1031);first,fg,fb=first_family();second,sg,sb=second_family();cases=first+second;families=[fg,[[i+len(first) for i in g] for g in sg]];warm_folder=BASE/'curved_group_comparison_grouped_v1';wr=read(warm_folder/'receipt.json');assert digest(warm_folder/'model.pt')==wr['model_sha256'];warm=torch.load(warm_folder/'model.pt',map_location='cpu',weights_only=False);mean,std=warm['mean'],warm['std'];out=BASE/'joint_arc_boundary_pointer_v1';out.mkdir(exist_ok=False)
    write(out/'protocol.json',dict(steps=4000,seed=1031,lr=.001,lr_end=.0001,batch_size=1,sampling='Equal family; first family equal parent then compatible carrier; second equal6 multi-parent carriers',warm_model_sha256=wr['model_sha256'],feature_receipt_sha256=digest(BASE/'curved_probe_event_features_v1/new_fcaf2af1abee/receipt.json'),family_bindings=dict(first=fb,second=sb),input_dim=83,normalization_frozen=True,module_sha256=digest('weld_pipeline/server_arc_boundary_feedback_pointer.py'),base_module_sha256=digest('weld_pipeline/server_arc_boundary_pointer.py'),trainer_sha256=digest(__file__),scope='Joint conditional TRAIN boundary rehearsal. No learned candidate rejection/contact/type or parent relation. Both families previously trained; retention is NOT heldout generalization.',DEV_reads=0))
    model=ArcBoundaryFeedbackPointer(feedback_mode='model_valid',input_dim=83).cuda();model.load_state_dict(warm['state_dict'])
    for c in cases:c['features']=torch.as_tensor(np.clip((c['x']-mean)/std,-20,20),dtype=torch.float32,device='cuda');c['positions']=torch.as_tensor(c['q'],dtype=torch.float64,device='cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000,eta_min=.0001);history=[]
    for step in range(4000):
        family=families[int(rng.integers(2))];group=family[int(rng.integers(len(family)))];c=cases[group[int(rng.integers(len(group)))]];loss=model.set_teacher_loss(c['features'],c['positions'],c['positive'],c['teacher']);assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();schedule.step()
        if step%500==0:history.append(dict(step=step,family=c['family'],loss=float(loss.detach())));print(history[-1],flush=True)
    model.eval();predictions=dict(first=[],second=[])
    for c in cases:
        path=model.decode(c['features'],c['positions']);identity={k:c[k] for k in ['profile','face','candidate','parent'] if k in c};predictions[c['family']].append(dict(identity,selected_local_event_indices=path,all_boundaries_within_label_sets=len(path)==len(c['positive']) and all(i in ids for i,ids in zip(path,c['positive']))))
    torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,protocol_sha256=digest(out/'protocol.json')),out/'model.pt');write(out/'conditional_predictions.json',dict(rows=predictions['second']));write(out/'conditional_first_family_predictions.json',dict(rows=predictions['first']));write(out/'receipt.json',dict(status='JOINT_TRAIN_POINTER_NOT_PROMOTED',model_sha256=digest(out/'model.pt'),predictions_sha256=digest(out/'conditional_predictions.json'),first_predictions_sha256=digest(out/'conditional_first_family_predictions.json'),history=history,seconds=time.perf_counter()-started,fit_calls=1,DEV_reads=0))

if __name__=='__main__':main()
