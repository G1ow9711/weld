"""Freeze raw PTv3 features for all 8, then prepare TRAIN ambiguity labels only."""
import argparse
import gc
import numpy as np
import torch
from training.dataio import read_xyz_pcd
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
from weld_pipeline.sjg_material_ambiguity_874 import infer_backbone_features,targets_from_reference,ambiguity_loss


def confusion(prob,target):
    known=target>=0;pred=prob>=.5;positive=target==1
    return dict(TP=int((known&pred&positive).sum()),FP=int((known&pred&~positive).sum()),
        FN=int((known&~pred&positive).sum()),TN=int((known&~pred&~positive).sum()),unknown=int((~known).sum()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);checkpoint=OUT/'run74/dense_control/last.pt'
    write(dest/'registered_protocol.json',dict(stage='FROZEN_PTV3_FEATURES_AND_SEPARATE_MATERIAL_AMBIGUITY_TARGETS',sources=TRAIN+VALID+ARC,
        all_inputs_frozen_before_reference=True,only_TRAIN_targets=True,checkpoint_sha256=digest(checkpoint),
        surface_boundary_is_not_material_ambiguity=True,positive='offline multiple material memberships',
        negative='offline unique known reference owner',unknown='not covered by offline structure reference',
        reference_not_customer_approved_truth=True,source_sha256=digest(__file__),full_contract_pass=False,start_utc=utc()))
    target=targets_from_reference(np.array([2,-1,-1]),np.array([False,True,False]));assert target.tolist()==[0,1,-1]
    logit=torch.tensor([.2,-.2,1.],requires_grad=True);loss=ambiguity_loss(logit,torch.from_numpy(target));loss.backward();assert logit.grad[2]==0
    assert ambiguity_loss(logit,torch.full((3,),-1)).item()==0
    write(dest/'controls.json',dict(unique_negative_multi_positive_uncovered_unknown=True,unknown_gradient_zero=True,all_unknown_zero=True))
    model=SurfaceMaterialModel().cuda()
    with checkpoint.open('rb') as h:state=torch.load(h,map_location='cuda')
    model.load_state_dict(state['model']);del state;model.eval();inventory=read(OUT/'inventory.json');frozen=[]
    relations={r['source']:r for r in read(OUT/'run75/dense_control/all_predictions_frozen.json')['sources']}
    for source in TRAIN+VALID+ARC:
        check_resources();raw=ROOT/inventory[source]['raw']['original_pcd_path'];assert digest(raw)==relations[source]['raw_sha256']
        xyz=read_xyz_pcd(raw)*1000.;print('AMBIGUITY_RAW_FEATURE_START',source,len(xyz),flush=True)
        output=infer_backbone_features(model,xyz,resource_check=check_resources,
            progress=lambda done,total:print('AMBIGUITY_RAW_FEATURE_BLOCK',source,done,'/',total,flush=True) if done==1 or done%20==0 or done==total else None)
        prior=OUT/'run75/dense_control'/relations[source]['file'];assert digest(prior)==relations[source]['sha256']
        with np.load(prior) as p:
            normal_error=float(np.abs(output['normal']-p['normal']).max());boundary_error=float(np.abs(output['boundary']-p['boundary']).max())
        assert normal_error<1e-4 and boundary_error<1e-4
        path=dest/f'source_{source:02d}_features.npy';np.save(path,output['features'])
        frozen.append(dict(source=source,points=len(xyz),file=path.name,sha256=digest(path),raw_sha256=digest(raw),
            every_original_point_one_core=bool(np.all(output['core_coverage']==1)),normal_max_abs_difference=normal_error,boundary_max_abs_difference=boundary_error))
        print('AMBIGUITY_RAW_FEATURE_FROZEN',frozen[-1],flush=True);del xyz,output;gc.collect();torch.cuda.empty_cache()
    write(dest/'all_inputs_frozen.json',dict(sources=frozen,all_before_reference=True));del model;torch.cuda.empty_cache()
    teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')};rows=[];total=positive=negative=unknown=0
    for source in TRAIN:
        teacher=OUT/'run22'/teachers[source]['teacher_file'];assert digest(teacher)==teachers[source]['teacher_sha256']
        with np.load(teacher) as t:
            labels=targets_from_reference(t['reference_volume'],t['multiple_material']);boundary_truth=t['boundary_target']
            unique_reference_face_boundary=int(((labels==0)&(boundary_truth==1)).sum())
        with np.load(OUT/f'run75/dense_control/source_{source:02d}_material_relations.npz') as p:old=confusion(p['boundary'],labels)
        path=dest/f'source_{source:02d}_ambiguity_target.npy';np.save(path,labels)
        row=dict(source=source,points=len(labels),positive=int((labels==1).sum()),negative=int((labels==0).sum()),unknown=int((labels<0).sum()),
            file=path.name,sha256=digest(path),teacher_sha256=digest(teacher),face_boundary_as_material_ambiguity=old,
            unique_owner_points_at_reference_face_boundary=unique_reference_face_boundary)
        rows.append(row);total+=len(labels);positive+=row['positive'];negative+=row['negative'];unknown+=row['unknown']
        print('AMBIGUITY_TRAIN_TARGET',row,flush=True)
    write(dest/'training_targets.json',dict(rows=rows,unknown_retained=True))
    write(dest/'receipt.json',dict(status='AMBIGUITY_FEATURES_AND_TRAIN_TARGETS_READY',train_points=total,positive=positive,negative=negative,unknown=unknown,
        all8_points=sum(r['points'] for r in frozen),full_contract_G=496,optimizer_updates=0,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
