"""Registered small-case PTv3 surface-learning diagnostic, not weld acceptance."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_ptv3_801 import CONFIG,build_backbone,voxelize,cuda_input

DATA=OUT/'run22'


class SurfaceModel(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=build_backbone()
        self.embedding=nn.Linear(32,16);self.normal=nn.Linear(32,3);self.boundary=nn.Linear(32,1)
    def forward(self,block):
        feature=self.backbone(cuda_input(block)).feat
        inverse=torch.from_numpy(block['inverse']).cuda()
        return self.embedding(feature)[inverse],F.normalize(self.normal(feature),dim=-1)[inverse],self.boundary(feature).squeeze(-1)[inverse]


def spatial_partition(xyz,maximum=4096):
    pending=[np.arange(len(xyz),dtype=np.int64)];parts=[]
    while pending:
        ids=pending.pop()
        if len(ids)<=maximum:parts.append(ids);continue
        axis=int(np.argmax(np.ptp(xyz[ids],axis=0)));order=np.argsort(xyz[ids,axis],kind='stable')
        half=len(ids)//2;pending.extend([ids[order[:half]],ids[order[half:]]])
    assert np.array_equal(np.sort(np.concatenate(parts)),np.arange(len(xyz),dtype=np.int64))
    return parts


def prepare_blocks(xyz,labels,dest):
    tree=cKDTree(xyz);blocks=[];trace={}
    for bi,core in enumerate(spatial_partition(xyz)):
        near=np.asarray(tree.query(xyz[core].mean(0),k=min(8192,len(xyz)))[1],np.int64).reshape(-1)
        context=near[~np.isin(near,core)][:8192-len(core)]
        ids=np.r_[core,context].astype(np.int64);block=voxelize(xyz[ids])
        y={k:labels[k][ids].copy() for k in ('instance','normal','boundary_target')}
        # Losses cover each original point exactly once per full epoch.
        y['core']=np.arange(len(ids))<len(core)
        # One voxel cannot encode two different known surface instances.
        known=y['instance']>=0;bins=block['inverse'][known];values=y['instance'][known]
        low=np.full(len(block['coord']),np.iinfo(np.int32).max,np.int32);high=np.full(len(low),-1,np.int32)
        np.minimum.at(low,bins,values);np.maximum.at(high,bins,values)
        conflict=low<high;y['instance'][conflict[block['inverse']]]=-1
        blocks.append((block,y));trace[f'ids_{bi}']=ids;trace[f'core_count_{bi}']=np.array(len(core),np.int64)
    np.savez_compressed(dest/'input_blocks.npz',**trace)
    return blocks


def losses(output,y,pos_weight):
    emb,normal,boundary=output
    labels=torch.from_numpy(y['instance'].astype(np.int64)).cuda()
    core=torch.from_numpy(y['core']).cuda();known=(labels>=0)&core
    zero=emb.sum()*0
    pull=push=normloss=zero
    if known.any():
        idx=torch.nonzero(known,as_tuple=True)[0]
        # Bound clustering loss memory while retaining normal/boundary losses
        # on every eligible core point.
        if len(idx)>2048:idx=idx[torch.linspace(0,len(idx)-1,2048,device='cuda').long()]
        features=emb[idx];lab=labels[idx];_,inverse,counts=torch.unique(lab,return_inverse=True,return_counts=True)
        centers=torch.zeros((len(counts),16),device='cuda').index_add_(0,inverse,features)/counts[:,None]
        distance=torch.linalg.norm(features-centers[inverse],dim=1)
        penalties=F.relu(distance-.2).square()
        pull=(torch.zeros(len(counts),device='cuda').index_add_(0,inverse,penalties)/counts).mean()
        if len(counts)>1:
            pair=torch.cdist(centers,centers)
            push=F.relu(1.-pair[~torch.eye(len(counts),device='cuda',dtype=torch.bool)]).square().mean()
        truth=torch.from_numpy(y['normal']).cuda()
        normloss=(1.-(normal[known]*truth[known]).sum(-1).abs()).mean()
        pull=pull+.001*centers.norm(dim=1).mean()
    by=torch.from_numpy(y['boundary_target'].astype(np.float32)).cuda();bm=(by>=0)&core
    boundaryloss=F.binary_cross_entropy_with_logits(boundary[bm],by[bm],pos_weight=pos_weight) if bm.any() else zero
    total=pull+push+normloss+.5*boundaryloss
    return total,dict(total=float(total.detach()),pull=float(pull.detach()),push=float(push.detach()),normal=float(normloss.detach()),boundary=float(boundaryloss.detach()))


def evaluate(model,blocks,pos_weight):
    model.eval();metrics=[];angles=[];tp=fp=fn=tn=0
    with torch.no_grad():
        for block,y in blocks:
            output=model(block);_,stats=losses(output,y,pos_weight);metrics.append(stats)
            core=y['core'];mask=core&(y['instance']>=0)
            n=output[1].cpu().numpy();truth=y['normal']
            angles.extend(np.rad2deg(np.arccos(np.clip(np.abs((n[mask]*truth[mask]).sum(1)),0,1))).tolist())
            pred=output[2].sigmoid().cpu().numpy()>=.5;known=core&(y['boundary_target']>=0);positive=y['boundary_target']==1
            tp+=int((pred&known&positive).sum());fp+=int((pred&known&~positive).sum())
            fn+=int((~pred&known&positive).sum());tn+=int((~pred&known&~positive).sum())
    return dict(loss={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
        unoriented_normal_p50_deg=float(np.median(angles)) if angles else None,
        unoriented_normal_p95_deg=float(np.quantile(angles,.95)) if angles else None,
        boundary=dict(tp=tp,fp=fp,fn=fn,tn=tn),scope='SOURCE21_TRAIN_DIAGNOSTIC_ONLY',
        no_weld_extraction_or_directed_normal_acceptance=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(804);np.random.seed(804)
    protocol=dict(stage='PTV3_SINGLE_TRAIN_CASE_SURFACE_LEARNING',source=21,train_sources=[21],
        full_train_cohort_not_yet_run=[4,10,25,36,42],valid_sources_not_trained=[64],arc_diagnostic=[24],
        epochs=20,update_limit=2000,seed=804,learning_rate=3e-4,config=CONFIG,
        upstream_commit=read(ROOT/'training/third_party/ptv3_801/provenance.json')['commit'],
        inference_inputs='XYZ only',offline_supervision='planar finite STEP reference; ambiguous points masked',
        losses=['surface_instance_pull_push','unoriented_normal','boundary'],
        task_scope='surface front-end diagnostic; not a full weld model',full_contract_denominator=496,
        contract_pass=False,start_utc=utc(),source_sha256=digest(Path(__file__)))
    write(dest/'registered_protocol.json',protocol);updates=completed=0
    try:
        assert read(DATA/'receipt.json')['status']=='OFFLINE_PLANAR_POINT_TEACHERS_COMPLETE'
        freeze=next(v for v in read(DATA/'all_inputs_frozen.json')['sources'] if v['source']==21)
        label=next(v for v in read(DATA/'label_summary.json') if v['source']==21)
        assert digest(DATA/freeze['xyz_file'])==freeze['xyz_sha256']
        assert digest(DATA/label['teacher_file'])==label['teacher_sha256']
        xyz=np.load(DATA/freeze['xyz_file'])
        with np.load(DATA/label['teacher_file'],allow_pickle=False) as z: labels={k:z[k] for k in ('instance','normal','boundary_target')}
        blocks=prepare_blocks(xyz,labels,dest)
        protocol.update(blocks_per_epoch=len(blocks),original_points_per_full_epoch=len(xyz),
            block_mapping_sha256=digest(dest/'input_blocks.npz'),teacher_sha256=label['teacher_sha256'],
            input_sha256=freeze['xyz_sha256'],voxel_conflict_instance_unknown_points=sum(int(((y['instance']<0)&y['core']).sum()) for _,y in blocks))
        write(dest/'registered_protocol.json',protocol)
        check_resources();model=SurfaceModel().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        pos_weight=torch.tensor(min(20.,label['boundary_negative']/max(1,label['boundary_positive'])),device='cuda')
        before=evaluate(model,blocks,pos_weight);write(dest/'before.json',before)
        print('PTV3_SURFACE_START points',len(xyz),'blocks_per_epoch',len(blocks),'epochs',20,'surface diagnostic, not weld accuracy',flush=True)
        with (dest/'epochs.jsonl').open('w',encoding='utf-8') as log:
            for epoch in range(1,21):
                start=time.perf_counter();model.train();stats=[]
                for batch,index in enumerate(np.random.default_rng(804+epoch).permutation(len(blocks)),1):
                    check_resources();block,y=blocks[int(index)];optimizer.zero_grad(set_to_none=True)
                    loss,values=losses(model(block),y,pos_weight)
                    assert torch.isfinite(loss);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(norm)
                    optimizer.step();updates+=1;stats.append(values)
                    print('EPOCH',epoch,'/20 BATCH',batch,'/',len(blocks),'UPDATE',updates,'LOSS',round(values['total'],6),flush=True)
                    if updates>=2000:break
                full=batch==len(blocks)
                if full:completed+=1
                evaluation=evaluate(model,blocks,pos_weight)
                row=dict(epoch=epoch,complete_epoch=full,updates=updates,train_loss=float(np.mean([x['total'] for x in stats])),
                    evaluation=evaluation,seconds=time.perf_counter()-start)
                log.write(json.dumps(row)+'\n');log.flush();write(dest/'latest.json',row)
                with (dest/'last.pt').open('wb') as handle:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    epoch=epoch,updates=updates,protocol=protocol),handle)
                print('EPOCH_COMPLETE',epoch,'surface_loss',evaluation['loss']['total'],'normal_p95',evaluation['unoriented_normal_p95_deg'],flush=True)
                if updates>=2000:break
        after=evaluate(model,blocks,pos_weight);write(dest/'after.json',after)
        write(dest/'receipt.json',dict(status='PTV3_SINGLE_CASE_SURFACE_TRAIN_COMPLETE',completed_epochs=completed,actual_updates=updates,
            before=before,after=after,end_utc=utc(),checkpoint_sha256=digest(dest/'last.pt'),
            gpu_peak_MiB=torch.cuda.max_memory_allocated()/2**20,full_weld_model=False,contract_pass=False,
            development_holdout_evaluation_not_run=True))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),actual_updates=updates,completed_epochs=completed,error=traceback.format_exc()));raise


if __name__=='__main__':main()
