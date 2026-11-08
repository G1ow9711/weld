"""Dense whole-interval instances on native curves; no point-mask crossings."""
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import curveonly_01a1ac7c_sequence_v171 as previous

OBSERVATIONS=111
PERMUTATION=np.r_[previous.PERMUTATION,108,107,109,110].astype(np.int64)
OUTPUT_PERMUTATION=np.array([0,2,1,3],np.int64)
GRID_MM=previous.GRID_MM
BATCH_SIZE=previous.BATCH_SIZE
NMS_IOU=.5
assert np.array_equal(PERMUTATION[PERMUTATION],np.arange(OBSERVATIONS))


def candidate_rows(features,packet):
    rows=previous.candidate_rows(features,packet)
    for row in rows:
        L=row['length_mm'];s=np.arange(len(row['grid']))*row['grid_step_mm']
        left=s/L if not row['closed']else np.zeros(len(s))
        right=1-s/L if not row['closed']else np.zeros(len(s))
        extra=np.column_stack([left,right,np.full(len(s),row['closed']),np.full(len(s),np.log1p(L))])
        row['grid']=np.column_stack([row['grid'],extra]).astype(np.float32)
    return rows


def targets(row,chosen):
    """Exact interval identity and endpoints; observation absence is not a gap."""
    s=row['station'];L=row['length_mm'];n=len(s)
    result=dict(labels=np.zeros(n,np.float32),left_fraction=np.zeros(n),right_fraction=np.zeros(n),
        quality=np.zeros(n),instance=np.full(n,-1,np.int64))
    best=np.full(n,-1.);coverage=[];overlap=np.zeros(n,int)
    for ii,t in enumerate(chosen):
        a,b=float(t['start_mm']),float(t['end_mm']);assert b>a
        full=row['closed']and abs((b-a)-L)<=previous.reference.PROJECTION_NUMERIC_MM
        if full:
            take=np.ones(n,bool);left=right=np.full(n,L/2);quality=np.ones(n)
        else:
            take=np.zeros(n,bool);left=np.zeros(n);right=np.zeros(n)
            for turn in ((-1,0,1)if row['closed']else(0,)):
                q=s+turn*L;inside=(q>=a)&(q<=b)
                take|=inside;left[inside]=q[inside]-a;right[inside]=b-q[inside]
            quality=np.sqrt(np.minimum(left,right)/np.maximum(np.maximum(left,right),1e-30))
        overlap+=take
        replace=take&(quality>best)
        result['labels'][replace]=1;result['left_fraction'][replace]=left[replace]/L
        result['right_fraction'][replace]=right[replace]/L;result['quality'][replace]=quality[replace]
        result['instance'][replace]=ii;best[replace]=quality[replace]
        coverage.append(dict(instance=ii,truth=t['truth'],start_mm=a,end_mm=b,closed_full=bool(full),
            anchor_count=int(take.sum())))
    for q in coverage:q['assigned_anchor_count']=int((result['instance']==q['instance']).sum())
    assert all(q['anchor_count']>0 and q['assigned_anchor_count']>0 for q in coverage),('INSTANCE_WITHOUT_INPUT_ANCHOR',coverage)
    assert (result['left_fraction']>=0).all()and(result['right_fraction']>=0).all()
    result.update(coverage=coverage,overlap_anchor_count=int((overlap>1).sum()))
    return result


def interval_iou(a,b,other_a,other_b,length,closed):
    other_a=np.asarray(other_a,float);other_b=np.asarray(other_b,float)
    if closed:
        a,b=a%length,a%length+min(b-a,length)
        width=np.minimum(other_b-other_a,length);other_a=other_a%length;other_b=other_a+width
        overlap=np.zeros(len(other_a))
        for turn in (-1,0,1):
            overlap+=np.maximum(0,np.minimum(b,other_b+turn*length)-np.maximum(a,other_a+turn*length))
        overlap=np.minimum(overlap,np.minimum(b-a,width))
    else:
        width=other_b-other_a;overlap=np.maximum(0,np.minimum(b,other_b)-np.maximum(a,other_a))
    return overlap/np.maximum((b-a)+width-overlap,1e-30)


def intervals(row,foreground,left_fraction,right_fraction,quality):
    """Positive anchors propose complete spans; suppress competing whole spans."""
    s=row['station'];L=row['length_mm'];closed=row['closed']
    p=np.asarray(foreground,float);left=np.asarray(left_fraction,float)*L;right=np.asarray(right_fraction,float)*L
    q=np.asarray(quality,float);assert p.shape==left.shape==right.shape==q.shape==s.shape
    assert np.isfinite(np.r_[p,left,right,q]).all()
    left=np.maximum(left,0);right=np.maximum(right,0)
    if closed:
        total=left+right;factor=np.minimum(1.,L/np.maximum(total,1e-30));left*=factor;right*=factor
    else:left=np.minimum(left,s);right=np.minimum(right,L-s)
    start=s-left;end=s+right
    valid=(p>.5)&(end>start)
    ids=np.flatnonzero(valid)
    if not len(ids):return [],dict(input_anchors=len(s),positive_anchors=0,kept=0,no_positive_support=True)
    score=np.sqrt(np.clip(p*q,0,1))
    order=ids[np.lexsort((ids,-score[ids]))];kept=[]
    while len(order):
        i=int(order[0]);kept.append(i);rest=order[1:]
        if not len(rest):break
        iou=interval_iou(float(start[i]),float(end[i]),start[rest],end[rest],L,closed)
        order=rest[iou<NMS_IOU]
    runs=[]
    for i in kept:
        a,b=float(start[i]),float(end[i])
        if closed:width=b-a;a%=L;b=a+width
        runs.append(dict(start_mm=a,end_mm=b,confidence=float(score[i]),anchor=int(i),foreground=float(p[i])))
    return runs,dict(input_anchors=len(s),positive_anchors=int(len(ids)),kept=len(runs),
        whole_interval_nms_iou=NMS_IOU,no_minimum_weld_or_gap_length=True,no_count_template=True)


def proposals(packet,rows,predictions):
    output=[];states=[]
    for row,prediction in zip(rows,predictions):
        runs,record=intervals(row,*prediction);ci=row['candidate'];points=np.asarray(packet['candidates'][ci]['points'])
        states.append(dict(candidate=ci,intervals=runs,native_reference=row['native_reference'],**record))
        for j,q in enumerate(runs):
            output.append(dict(candidate=ci,points=previous.geometry.native_slice(points,q['start_mm'],q['end_mm']).tolist(),
                weld_type='horizontal',confidence=q['confidence'],interval=dict(start_mm=q['start_mm'],end_mm=q['end_mm'],index=j,predicted_count=len(runs))))
    return output,states


def fit_range_normalizer(rows):
    low=np.full(OBSERVATIONS,np.inf);high=np.full(OBSERVATIONS,-np.inf)
    for row in rows:low=np.minimum(low,row['grid'].min(0));high=np.maximum(high,row['grid'].max(0))
    low=np.minimum(low,low[PERMUTATION]);high=np.maximum(high,high[PERMUTATION])
    mean=((low+high)/2).astype(np.float32);scale=((high-low)/2).astype(np.float32)
    scale=np.where(scale>1e-12,scale,1.).astype(np.float32)
    assert np.array_equal(mean,mean[PERMUTATION])and np.array_equal(scale,scale[PERMUTATION])
    return mean,scale


def pack(rows,mean,std,device):
    size=np.asarray([len(q['grid'])for q in rows],np.int64);width=int(size.max())
    source=np.zeros((len(rows),OBSERVATIONS,width),np.float32);n=max(len(q['station'])for q in rows)
    arrays=dict(x=source,size=size,closed=np.asarray([q['closed']for q in rows],bool),
        left=np.zeros((len(rows),n),np.int64),right=np.zeros((len(rows),n),np.int64),alpha=np.zeros((len(rows),n),np.float32))
    for key in ('labels','left_fraction','right_fraction','quality','weight','positive_weight'):arrays[key]=np.zeros((len(rows),n),np.float32)
    for i,row in enumerate(rows):
        source[i,:,:size[i]]=((row['grid']-mean)/std).T;k=len(row['station'])
        for key in arrays:
            if key in row and key not in ('x','size','closed'):arrays[key][i,:k]=row[key]
    return {key:torch.as_tensor(value,device=device)for key,value in arrays.items()}


def anchor_outputs(grid,batch):
    left=batch['left'][:,None,:].expand(-1,grid.shape[1],-1);right=batch['right'][:,None,:].expand_as(left)
    alpha=batch['alpha'][:,None,:]
    return grid.gather(2,left)*(1-alpha)+grid.gather(2,right)*alpha


class IntervalNetwork(nn.Module):
    def __init__(self):
        super().__init__();c=previous.CHANNELS
        self.input=nn.Conv1d(OBSERVATIONS,c,1)
        self.blocks=nn.ModuleList([previous.BoundaryDepthwise(d)for d in previous.DILATIONS])
        self.head=nn.Sequential(nn.Conv1d(3*c,c,1),nn.SiLU(),nn.Conv1d(c,4,1))
        self.register_buffer('reverse_channels',torch.as_tensor(PERMUTATION.copy(),dtype=torch.long))
        self.register_buffer('reverse_outputs',torch.as_tensor(OUTPUT_PERMUTATION.copy(),dtype=torch.long))

    def forward(self,x,size,closed):
        mask=(torch.arange(x.shape[-1],device=x.device)[None,None,:]<size[:,None,None]).to(x.dtype)
        z=F.silu(self.input(x))*mask
        for block in self.blocks:z=block(z,size,closed,mask)
        average=z.sum(2,keepdim=True)/size[:,None,None]
        maximum=z.masked_fill(mask==0,-torch.inf).amax(2,keepdim=True)
        return self.head(torch.cat([z,average.expand_as(z),maximum.expand_as(z)],dim=1))*mask

    def both_directions(self,batch):
        x,size,closed=batch['x'],batch['size'],batch['closed'];index=previous.reverse_index(size,closed,x.shape[-1])
        reverse=x.gather(2,index[:,None,:].expand(-1,x.shape[1],-1))[:,self.reverse_channels,:]
        a=self.forward(x,size,closed);b=self.forward(reverse,size,closed)
        b=b.gather(2,index[:,None,:].expand(-1,b.shape[1],-1))[:,self.reverse_outputs,:]
        return a,b


def predict_fields(network,rows,mean,std,device='cpu'):
    if str(device)=='cpu'and torch.get_num_threads()!=2:torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);network.eval();result=[None]*len(rows)
    with torch.no_grad():
        for ids in previous.batches(rows):
            batch=pack([rows[i]for i in ids],mean,std,device);a,b=network.both_directions(batch)
            values=torch.sigmoid(anchor_outputs((a+b)*.5,batch)).cpu().numpy()
            for j,i in enumerate(ids):result[i]=values[j,:,:len(rows[i]['station'])].astype(float)
    return result


class IntervalCurveSet:
    def __init__(self,network,mean,std):
        self.network=network.cpu().eval();self.mean=np.asarray(mean,np.float32);self.std=np.asarray(std,np.float32)
        self.last_states=self.last_timing=self.last_fields=None

    def predict(self,features,packet):
        start=time.perf_counter();rows=candidate_rows(features,packet)
        fields=predict_fields(self.network,rows,self.mean,self.std);output,states=proposals(packet,rows,fields)
        self.last_fields=fields;self.last_states=states
        self.last_timing=dict(cpu_seconds=time.perf_counter()-start,parameter_count=sum(p.numel()for p in self.network.parameters()),
            complete_interval_instances=True,no_crossing_CTMC_or_count_template=True,no_minimum_weld_or_gap_length=True)
        return output


input_curves=previous.input_curves
arclength=previous.arclength
NAMES=previous.NAMES


def decode(features,packet,model,threshold=.5):
    assert isinstance(model,IntervalCurveSet)
    return previous.original_runtime.decode(features,packet,model,threshold)
