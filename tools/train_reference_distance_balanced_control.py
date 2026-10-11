"""Fixed-budget auxiliary distance comparison on the frozen 47/11 split."""
from pathlib import Path
from collections import OrderedDict
import time,hashlib,gc
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from weld_pipeline.server_legacy_runtime import install
install()
from weld_pipeline.server_reference_distance_model import ReferenceDistanceModel,bounded_log_distance_loss
from weld_pipeline.sjg_material_model_859 import SurfaceMaterialModel
from weld_pipeline.sjg_ptv3_801 import voxelize
from weld_pipeline.sjg_local_runtime_904 import read,write,digest,check_resources
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'server_rebuild/expansion_v1';OUT=BASE/'reference_distance_balanced_loss_v1'
SEED=1031;POINTS=4096;EPOCHS=4


def data(key):
    xyz=np.load(BASE/'inputs'/key/'xyz_mm.npy',mmap_mode='r')
    fields=BASE/'reference_distance_fields_v1'/key
    return xyz,np.load(fields/'lower_mm.npy',mmap_mode='r'),np.load(fields/'upper_mm.npy',mmap_mode='r')


def prepare(cases):
    records=[]
    for case in cases:
        check_resources();key=case['key'];xyz,lo,hi=data(key);r=read(BASE/'reference_distance_fields_v1'/key/'receipt.json')
        assert digest(BASE/'inputs'/key/'xyz_mm.npy')==r['input_sha256']
        assert r['target_sha256']==case['target_sha256']
        assert digest(BASE/'reference_distance_fields_v1'/key/'lower_mm.npy')==r['lower_sha256']
        assert digest(BASE/'reference_distance_fields_v1'/key/'upper_mm.npy')==r['upper_sha256']
        rng=np.random.default_rng(int(hashlib.sha256(('aux_crops|'+key).encode()).hexdigest()[:16],16))
        tree=cKDTree(xyz)
        # Evaluation anchors selected solely from XYZ indices before any target-based sampling.
        evaluation=rng.integers(len(xyz),size=8)
        ev=np.asarray(tree.query(xyz[evaluation],k=min(POINTS,len(xyz)),workers=2)[1],np.int64)
        payload={'evaluation':ev}
        if case['role']=='train':
            uniform=rng.integers(len(xyz),size=8);near=np.flatnonzero(hi<=3);assert len(near)
            anchors=np.concatenate([uniform,rng.choice(near,size=8,replace=True)])
            payload['training']=np.asarray(tree.query(xyz[anchors],k=min(POINTS,len(xyz)),workers=2)[1],np.int64)
        path=OUT/'crops'/f'{key}.npz';np.savez_compressed(path,**payload)
        records.append(dict(key=key,role=case['role'],crop_sha256=digest(path),field_receipt_sha256=digest(BASE/'reference_distance_fields_v1'/key/'receipt.json')))
        del tree,xyz,lo,hi;gc.collect()
        print('CROPS',key,len(records),flush=True)
    write(OUT/'crop_receipt.json',dict(cases=records,evaluation='8 uniform input-selected spatial crops per family; original orientation; no validation-target sampling',training='8 uniform plus8 reference-near anchors per TRAIN family,4096 nearest raw points, rigid rotation each epoch',scope='Sampled local auxiliary validation only; neither full-cloud inference nor weld extraction'))


def evaluate(model,cases):
    model.eval();rows=[]
    with torch.no_grad():
        for case in cases:
            key=case['key'];xyz,lo,hi=data(key);crops=np.load(OUT/'crops'/f'{key}.npz')['evaluation'];losses=[];total=near_count=near_hits=far_count=far_false=0;absolute=0.
            for ids in crops:
                out=model(voxelize(xyz[ids]));lower=torch.as_tensor(lo[ids],device='cuda');upper=torch.as_tensor(hi[ids],device='cuda');pred=out['log_distance'];losses.append(float(bounded_log_distance_loss(pred,lower,upper)))
                mm=torch.expm1(pred);absolute+=float((torch.relu(lower-mm)+torch.relu(mm-upper)).sum());total+=len(ids)
                near=upper<=3;far=lower>=6;near_count+=int(near.sum());near_hits+=int(((mm<=3)&near).sum());far_count+=int(far.sum());far_false+=int(((mm<=3)&far).sum())
            rows.append(dict(key=key,role=case['role'],mean_bounded_log_loss=float(np.mean(losses)),mean_interval_distance_mm=absolute/total,point_observations=total,near_count=near_count,near_hits=near_hits,far_count=far_count,far_false=far_false))
    summary={}
    for role in ('train','development_validation'):
        rr=[r for r in rows if r['role']==role];n=sum(r['near_count'] for r in rr);f=sum(r['far_count'] for r in rr)
        summary[role]=dict(families=len(rr),macro_bounded_log_loss=float(np.mean([r['mean_bounded_log_loss'] for r in rr])),macro_interval_distance_mm=float(np.mean([r['mean_interval_distance_mm'] for r in rr])),near_recall=sum(r['near_hits'] for r in rr)/n if n else None,far_false_near_rate=sum(r['far_false'] for r in rr)/f if f else None)
    return dict(summary=summary,cases=rows,scope='Uniform local crops, repeated point observations possible; proximity thresholds diagnostic only, not physical-weld classification or original D3 acceptance')


def balanced_distance_loss(pred,lower,upper):
    error=torch.relu(torch.log1p(lower)-pred).square()+torch.relu(pred-torch.log1p(upper)).square()
    masks=(upper<=3,(upper>3)&(lower<6),lower>=6)
    return torch.stack([error[m].mean() for m in masks if m.any()]).mean()


def fit(mode,state,cases):
    torch.manual_seed(SEED);np.random.seed(SEED);model=ReferenceDistanceModel();missing=model.load_state_dict(state,strict=False);assert len(missing.missing_keys)==4 and not missing.unexpected_keys
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('distance_head.') or (mode=='finetune_distilled' and name.startswith('backbone.')))
    model=model.cuda().eval();teacher=None
    if mode=='finetune_distilled':
        teacher=SurfaceMaterialModel().cuda().eval();teacher.load_state_dict(state)
        for p in teacher.parameters():p.requires_grad_(False)
    parameters=[dict(params=list(model.distance_head.parameters()),lr=1e-3)]
    if teacher is not None:parameters.append(dict(params=list(model.backbone.parameters()),lr=1e-5))
    optimizer=torch.optim.AdamW(parameters,weight_decay=1e-4)
    folder=OUT/mode;folder.mkdir();write(folder/'initial_evaluation.json',evaluate(model,cases));rng=np.random.default_rng(SEED);train=[r for r in cases if r['role']=='train'];schedule=[(c['key'],i) for c in train for i in range(16)];history=[];step=0;start=time.perf_counter();cache=OrderedDict()
    for epoch in range(EPOCHS):
        for index in rng.permutation(len(schedule)):
            key,crop=schedule[index];check_resources()
            if key not in cache:
                cache[key]=(*data(key),np.load(OUT/'crops'/f'{key}.npz')['training'])
                if len(cache)>2:cache.popitem(last=False)
            cache.move_to_end(key);xyz,lo,hi,ids_all=cache[key];ids=ids_all[crop]
            rotation=Rotation.random(random_state=rng).as_matrix();points=np.asarray(xyz[ids],float);points=(points-points.mean(0))@rotation.T;block=voxelize(points)
            lower=torch.as_tensor(lo[ids],device='cuda');upper=torch.as_tensor(hi[ids],device='cuda');optimizer.zero_grad(set_to_none=True);output=model(block);distance=balanced_distance_loss(output['log_distance'],lower,upper);preserve=distance*0
            if teacher is not None:
                with torch.no_grad():reference=teacher(block)
                terms=[]
                for name,value in zip(('embedding','normal','boundary','material'),reference):
                    terms.append((output[name]-value).square().mean()/(value.square().mean().clamp_min(1e-4)))
                preserve=torch.stack(terms).mean()
            loss=distance+.1*preserve;assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10.,error_if_nonfinite=True);optimizer.step();step+=1
            if step%100==0:
                row=dict(step=step,epoch=epoch+1,distance_loss=float(distance.detach()),structural_loss=float(preserve.detach()),gradient_norm=float(norm));history.append(row);print(mode,row,flush=True)
        torch.save(dict(model=model.state_dict(),mode=mode,epoch=epoch+1,steps=step,protocol_sha256=digest(OUT/'protocol.json')),folder/'last.pt')
    final=evaluate(model,cases);write(folder/'evaluation.json',final)
    if mode=='frozen':
        for name,value in state.items():assert torch.equal(model.state_dict()[name].cpu(),value),'Frozen original state changed: '+name
    write(folder/'receipt.json',dict(status='AUXILIARY_FIT_COMPLETE',mode=mode,steps=step,epochs=EPOCHS,seconds=time.perf_counter()-start,checkpoint_sha256=digest(folder/'last.pt'),evaluation_sha256=digest(folder/'evaluation.json'),protocol_sha256=digest(OUT/'protocol.json'),history=history,protected_DEV_reads=0,scope='Distance auxiliary only; not promoted into full extraction'))
    print('COMPLETE',mode,final['summary'],flush=True);del model,teacher,optimizer,cache;gc.collect();torch.cuda.empty_cache()


def main():
    torch.set_num_threads(2);check_resources();audit=BASE/'reference_distance_fields_audit_v1/report.json';assert read(audit)['status']=='PASS';split=BASE/'reference_distance_auxiliary_protocol_v1/split.json';cases=read(split)['cases'];warm=ROOT/'weld_pipeline/reports/goal_contract_training_20261005/sjg_pilot_776/run162';checkpoint=digest(warm/'last.pt');assert checkpoint==read(warm/'receipt.json')['checkpoint_sha256'];OUT.mkdir(exist_ok=False);(OUT/'crops').symlink_to(BASE/'reference_distance_comparison_v1/crops',target_is_directory=True)
    write(OUT/'protocol.json',dict(seed=SEED,epochs=EPOCHS,steps_per_mode=47*16*EPOCHS,points_per_crop=POINTS,head_lr=.001,backbone_lr=.00001,structural_distillation_weight=.1,modes=['finetune_distilled'],training_loss='Equal mean of present near<=3mm, intermediate, far>=6mm bounded-log loss groups per crop; evaluation unchanged unweighted',crop_source_receipt_sha256=digest(BASE/'reference_distance_comparison_v1/crop_receipt.json'),split_sha256=digest(split),audit_sha256=digest(audit),checkpoint_sha256=checkpoint,script_sha256=digest(__file__),model_module_sha256=digest(ROOT/'weld_pipeline/server_reference_distance_model.py'),selection='Fixed final step, no validation checkpoint/threshold selection',mode='eval mode throughout both fits (no stochastic dropout); gradients enabled for fitting',comparison='Single loss-balancing control against prior finetune_distilled; same original initialization, seed, crop IDs, rotations, steps, rates and distillation. No validation selection.',scope='Reference distance auxiliary, not full weld detector/structure/attributes/frames',protected_DEV_reads=0))
    write(OUT/'crop_receipt.json',read(BASE/'reference_distance_comparison_v1/crop_receipt.json'));state=torch.load(warm/'last.pt',map_location='cpu',weights_only=False)['model']
    for mode in ('finetune_distilled',):fit(mode,state,cases)
    write(OUT/'receipt.json',dict(status='COMPARISON_COMPLETE',protocol_sha256=digest(OUT/'protocol.json'),results={mode:read(OUT/mode/'evaluation.json')['summary'] for mode in ('finetune_distilled',)},promoted=False))

if __name__=='__main__':main()
