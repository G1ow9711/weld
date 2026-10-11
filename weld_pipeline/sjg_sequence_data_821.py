"""Whole-profile TRAIN sequences from input-only grids; labels stay targets."""
import argparse
import gc
import traceback
import numpy as np
import torch
from scipy.spatial import cKDTree
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_joint_data_816 import load_inputs,point_queries
from weld_pipeline.sjg_joint_model_817 import JointEventModel


def sequence_features(profile_id,profiles,xyz,point,tree):
    p=profiles[profile_id]
    q=np.linspace(0,p['length'],int(np.ceil(p['length']))+1)
    values=[]
    for begin in range(0,len(q),2048):
        qq=q[begin:begin+2048]
        patch,query=point_queries(profiles,np.full(len(qq),profile_id,np.int64),qq,xyz,point,tree)
        values.append(np.c_[patch.mean(1),patch.std(1),query])
    x=np.concatenate(values).astype(np.float32)
    assert x.shape==(len(q),53) and np.isfinite(x).all()
    return q,x


def targets(q,events,type_by_child):
    y=np.zeros((len(q),3),np.float32);offset=np.zeros((len(q),2),np.float32)
    mask=np.zeros_like(offset);type_mask=np.zeros((len(q),2),bool)
    for event in events:
        inside=(q>=event['lo'])&(q<=event['hi']);y[inside,0]=1
        kind=type_by_child[(event['parent'],event['child'])]
        type_mask[inside,kind]=True
    if events:
        for head,key in enumerate(('lo','hi')):
            delta=np.asarray([e[key] for e in events])[None]-q[:,None]
            near=delta[np.arange(len(q)),np.abs(delta).argmin(1)]
            y[:,head+1]=np.exp(-.5*(near/2)**2)
            offset[:,head]=near/2;mask[:,head]=np.abs(near)<=4
    types=np.full(len(q),-1,np.int64)
    known=type_mask.sum(1)==1;types[known]=type_mask[known].argmax(1)
    return dict(targets=y,offsets=offset,offset_mask=mask,types=types)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    checkpoint=OUT/'run36/last.pt';model=JointEventModel().cuda()
    with checkpoint.open('rb') as h:state=torch.load(h,map_location='cuda')
    model.load_state_dict(state['model']);del state;model.eval()
    write(dest/'registered_protocol.json',dict(stage='FULL_PROFILE_SEQUENCE_TRAIN_DATA',train=TRAIN,
        development_not_read=VALID,arc_not_read=ARC,source_sha256=digest(__file__),
        selection='TRAIN predicted-positive profiles plus geometrically compatible TRAIN positives',
        selection_uses_TRAIN_labels=True,grid_coordinates_input_only=True,grid_step_mm=1.,
        sequence_features=53,global_condition_features=202,GT_not_feature=True,
        PTv3_and_joint_frozen=True,joint_checkpoint_sha256=digest(checkpoint),
        type_conflicts_unknown=True,material_and_parent_unresolved=True,full_contract_pass=False,start_utc=utc()))
    try:
        inventory=read(OUT/'inventory.json');records=[];rows=[]
        with torch.inference_mode():
            for si in TRAIN:
                check_resources();xyz,point,faces,profiles=load_inputs(si,inventory);tree=cKDTree(xyz)
                with np.load(OUT/f'run35/source_{si:02d}_inputs.npz') as z:
                    graph={k:torch.as_tensor(z[k],device='cuda') for k in z.files}
                h=model.encode_faces(graph['face_x'],graph['neighbors'],graph['relation'])
                probability=np.load(OUT/f'run36/after_source_{si:02d}_probability.npy')
                labels=np.load(OUT/f'run35/source_{si:02d}_labels.npy')
                selected=np.flatnonzero((probability>=.5)|(labels>0))
                truth=read(OUT/f'run35/source_{si:02d}_supervision.json')
                target_path=ROOT/inventory[si]['target']['target_path']
                assert digest(target_path)==inventory[si]['target']['target_sha256']
                target=read(target_path)
                type_by_child={(pi,ci):{'horizontal':0,'vertical':1}[c['weld_type']]
                    for pi,p in enumerate(target['parents']) for ci,c in enumerate(p['children'])}
                folder=dest/f'source_{si:02d}';folder.mkdir();points=0
                for index,ni in enumerate(selected,1):
                    ni=int(ni);q,x=sequence_features(ni,profiles,xyz,point,tree)
                    joint=model.encode_joints(h,graph['pairs'][ni:ni+1],graph['geometry'][ni:ni+1])[0].cpu().numpy()
                    condition=np.r_[joint,x.mean(0),x.std(0)].astype(np.float32)
                    y=targets(q,truth['by_profile'].get(str(ni),[]),type_by_child)
                    file=folder/f'profile_{ni:06d}.npz'
                    np.savez_compressed(file,x=x,condition=condition,q_mm=q,**y)
                    records.append(dict(source=si,profile=ni,file=str(file.relative_to(dest)),
                        sha256=digest(file),points=len(q),positive=bool(labels[ni]>0)))
                    points+=len(q)
                    if index==1 or index%40==0 or index==len(selected):
                        print('SEQUENCE_DATA',si,index,'/',len(selected),'GRID_POINTS',points,flush=True);check_resources()
                rows.append(dict(source=si,profiles=len(selected),grid_points=points,original_G=truth['original_G']))
                del xyz,point,faces,profiles,tree,graph,h;gc.collect();torch.cuda.empty_cache()
        write(dest/'manifest.json',dict(sequences=records,rows=rows,only_TRAIN=True))
        write(dest/'receipt.json',dict(status='WHOLE_PROFILE_TRAIN_SEQUENCES_COMPLETE',rows=rows,
            sequences=len(records),points=sum(r['points'] for r in records),end_utc=utc(),full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
