"""Training inputs and geometry-only whole-cloud inference for the pilot."""
import json
import random
import time
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from training.models.spunet import SpUNetBase

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'training/data/pilot_v1'


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.set_num_threads(4)
    # SpConv/CUDA atomic kernels are not guaranteed bitwise deterministic.
    torch.backends.cudnn.benchmark = False


@lru_cache(maxsize=3)
def geometry(name):
    p = DATA/name
    raw = np.load(p/'xyz_world_mm.npy',mmap_mode='r')
    first = np.load(p/'voxel_indices.npy',mmap_mode='r')
    xyz = np.asarray(raw[first])
    grid = np.load(p/'voxel_grid.npy',mmap_mode='r')
    return xyz, grid, cKDTree(xyz)


@lru_cache(maxsize=3)
def supervision(name):
    p=DATA/name
    labels=np.load(p/'voxel_labels.npy',mmap_mode='r')
    distance=np.load(p/'distance_mm.npy',mmap_mode='r')[np.load(p/'voxel_indices.npy',mmap_mode='r')]
    return labels, np.flatnonzero(labels==1), np.flatnonzero((labels==0)&(distance<30))


def make_input(xyz, grid, ix):
    q = xyz[ix]
    feat=((q-q.mean(axis=0))/1000).astype(np.float32)
    g=np.asarray(grid[ix],dtype=np.int32)
    g=g-g.min(axis=0)+16
    return {'grid_coord':torch.from_numpy(g).cuda(), 'feat':torch.from_numpy(feat).cuda(),
        'offset':torch.tensor([len(ix)],dtype=torch.long,device='cuda')}


def training_crop(name, rng, points):
    xyz, grid, tree = geometry(name)
    labels, positive, hard = supervision(name)
    r=rng.random()
    pool=positive if r<.5 and len(positive) else hard if r<.75 and len(hard) else None
    center=xyz[int(rng.choice(pool)) if pool is not None else int(rng.integers(len(xyz)))]
    _, ix = tree.query(center,k=min(points,len(xyz)),workers=1)
    ix=np.asarray(ix)
    return make_input(xyz,grid,ix),torch.from_numpy(np.array(labels[ix],dtype=np.int64)).cuda(),ix


def segmentation_loss(logits, target, foreground_weight):
    valid=target>=0
    if not bool(valid.any()): raise ValueError('Crop has no audited labels')
    x, y = logits.float()[valid], target[valid]
    ce=F.cross_entropy(x,y,weight=torch.tensor([1.,foreground_weight],device=x.device))
    prob=x.softmax(dim=1)[:,1]
    truth=(y==1).float()
    dice=1-(2*(prob*truth).sum()+1)/(prob.sum()+truth.sum()+1)
    return ce+dice


def scores(target, probability, threshold=.5, weights=None):
    valid=target>=0
    y=target[valid]==1; pred=probability[valid]>=threshold
    w=np.ones(len(y),dtype=np.int64) if weights is None else weights[valid]
    tp=int(w[y&pred].sum()); fp=int(w[~y&pred].sum()); fn=int(w[y&~pred].sum()); tn=int(w[~y&~pred].sum())
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,
        'f1':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1),'ignored':int((~valid).sum())}


def spatial_cores(xyz, max_core):
    """Disjoint spatial leaves cover all input voxels without any truth labels."""
    todo=[np.arange(len(xyz))]
    while todo:
        ix=todo.pop()
        if len(ix)<=max_core:
            yield ix; continue
        q=xyz[ix]
        axis=int(np.argmax(np.ptp(q,axis=0)))
        order=np.argsort(q[:,axis],kind='stable')
        mid=len(order)//2
        todo.extend([ix[order[:mid]],ix[order[mid:]]])


@torch.no_grad()
def predict_cloud(model, name, max_points=32768):
    """Input is XYZ/grid only; supervision is never read by this function."""
    model.eval()
    xyz, grid, tree=geometry(name)
    prob=np.full(len(xyz),np.nan,dtype=np.float32)
    count=0
    torch.cuda.synchronize(); start=time.perf_counter()
    for core in spatial_cores(xyz,max_points//2):
        _, context=tree.query(xyz[core].mean(axis=0),k=min(max_points,len(xyz)),workers=1)
        outside=np.asarray(context)[~np.isin(context,core)]
        ix=np.concatenate([core,outside[:max_points-len(core)]])
        # The installed Windows spconv wheel lacks a suitable FP16 inference
        # kernel on this driver. FP32 eval uses the same weights and architecture.
        logits=model(make_input(xyz,grid,ix))
        p=logits.float().softmax(dim=1)[:len(core),1].cpu().numpy()
        if not np.isfinite(p).all(): raise ValueError('Non-finite prediction')
        prob[core]=p; count+=1
    torch.cuda.synchronize()
    assert np.isfinite(prob).all(), 'Incomplete geometric coverage'
    return prob, {'blocks':count,'voxel_points':len(xyz),'seconds_model_and_tiling':time.perf_counter()-start,
        'timing_scope':'cached voxel geometry through predicted probabilities; not raw PCD to CSV acceptance time'}


def evaluate(model, names, points=32768, output=None):
    records=[]
    for name in names:
        probability,timing=predict_cloud(model,name,points)
        # The evaluator reads labels only after independent prediction is complete.
        raw_labels=np.load(DATA/name/'labels.npy',mmap_mode='r')
        inverse=np.load(DATA/name/'inverse.npy',mmap_mode='r')
        record={'name':name,**scores(raw_labels,probability[inverse]),**timing}
        records.append(record)
        if output:
            output=Path(output); output.mkdir(parents=True,exist_ok=True)
            np.save(output/f'{name}_voxel_probability.npy',probability)
        print(f"EVAL {name}: F1={record['f1']:.4f} P={record['precision']:.4f} R={record['recall']:.4f}; {record['seconds_model_and_tiling']:.1f}s",flush=True)
    total={key:sum(r[key] for r in records) for key in ['tp','fp','fn','tn']}
    tp,fp,fn=total['tp'],total['fp'],total['fn']
    result={'scope':'point-level weld-neighborhood segmentation; not weld instance/endpoint acceptance',
        'threshold':.5,'groups':records,'micro':{**total,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
            'f1':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1)},
        'macro_f1':float(np.mean([r['f1'] for r in records]))}
    return result


def create_model():
    model=SpUNetBase(in_channels=3,num_classes=2).cuda()
    # Sparse batches are small; the upstream .01 running-stat update converges
    # too slowly for the pilot. Architecture and trainable parameters stay intact.
    for layer in model.modules():
        if isinstance(layer,torch.nn.BatchNorm1d): layer.momentum=.1
    return model
