"""Freeze TRAIN-only negative supervision before matched training."""
import argparse
import numpy as np
import torch
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_train_ptv3_cohort_806 import load_block
from weld_pipeline.sjg_context_negative_867 import local_negative_pairs,context_negative_loss


def controls():
    xyz=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[9,0,0]],float)
    labels=np.array([8,8,4,4,-1]);core=np.array([1,1,0,0,1],bool)
    p=local_negative_pairs(xyz,labels,core)
    assert p['pairs'].tolist()==[[0,2],[1,2]] and p['group'].tolist()==[0,0]
    q=local_negative_pairs(xyz,np.where(labels==8,40,np.where(labels==4,99,-1)),core)
    assert np.array_equal(p['pairs'],q['pairs'])
    emb=torch.tensor([[0.,0],[.1,0],[.3,0],[2.,0],[.2,0]],requires_grad=True)
    loss=context_negative_loss(emb,torch.from_numpy(p['pairs']),torch.from_numpy(p['group']));loss.backward()
    assert torch.all(emb.grad[[0,1,2]].norm(dim=1)>0) and torch.all(emb.grad[[3,4]]==0)
    assert abs(float(loss)-(.7**2+.8**2)/2)<1e-6
    empty=local_negative_pairs(xyz,np.full(5,-1),core)
    assert len(empty['pairs'])==0
    assert context_negative_loss(emb,torch.from_numpy(empty['pairs']),torch.from_numpy(empty['group'])).item()==0
    return dict(context_only_negative_included=True,nearest_other_owner_verified=True,
        labels_renaming_invariant=True,unknown_gradient_zero=True,all_unknown_zero=True,optimizer_updates=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);rows=[];total=anchors=groups=context_only=0
    write(dest/'controls.json',controls())
    write(dest/'registered_protocol.json',dict(stage='CORE_TO_CONTEXT_LOCAL_NEGATIVE_PREPARATION',train=TRAIN,
        coordinates='model-visible voxel representative XYZ',unknown_is_not_negative=True,
        all_other_known_owners_in_input_block=True,source_sha256=digest(__file__),
        objective_sha256=digest(ROOT/'weld_pipeline/sjg_context_negative_867.py'),start_utc=utc()))
    for i,r in enumerate(read(OUT/'run67/manifest.json')['records']):
        if i%20==0:check_resources()
        assert r['source'] in TRAIN and digest(OUT/r['input_file'])==r['input_sha256']
        assert digest(OUT/'run67'/r['material_file'])==r['material_sha256']
        x,y=load_block(OUT/r['input_file']);labels=np.load(OUT/'run67'/r['material_file'])
        coord=x['coord'][x['inverse']];p=local_negative_pairs(coord,labels,y['core'])
        path=dest/f'negative_{i:04d}.npz';np.savez_compressed(path,**p)
        core_owners=np.unique(labels[y['core']&(labels>=0)]);all_owners=np.unique(labels[labels>=0])
        count=len(p['pairs']);n=int(len(np.unique(p['pairs'][:,0])));g=int(len(np.unique(p['group'])))
        only=len(core_owners)==1 and len(all_owners)>1;context_only+=int(only)
        total+=count;anchors+=n;groups+=g
        rows.append(dict(r,negative_file=path.name,negative_sha256=digest(path),negative_pairs=count,
            unique_core_anchors=n,ordered_owner_groups=g,context_only_negative=bool(only)))
        if i%40==0 or i+1==584:print('CONTEXT_NEGATIVE_PREPARE',i+1,'/584','PAIRS',total,flush=True)
    assert len(rows)==584 and groups==1742 and context_only==126
    write(dest/'manifest.json',dict(records=rows))
    write(dest/'receipt.json',dict(status='TRAIN_CONTEXT_NEGATIVE_PAIRS_FROZEN',blocks=len(rows),negative_pairs_per_epoch=total,
        unique_core_anchors_per_epoch=anchors,ordered_owner_groups=groups,context_only_negative_blocks=context_only,
        full_contract_G=496,full_contract_pass=False,optimizer_updates=0,end_utc=utc()))


if __name__=='__main__':main()
