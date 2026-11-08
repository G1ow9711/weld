"""Small native-curve sequence model, with open/periodic physical boundaries."""
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import curveonly_01a1ac7c_section_local_v135 as section
from . import curveonly_01a1ac7c_native_reference_v154 as reference
from . import curveonly_01a1ac7c_structured_intervals_v115 as sequence
from . import curveonly_01a1ac7c_set_v18 as geometry
from . import curveonly_01a1ac7c_local_runtime_v112 as original_runtime

CHANNELS=48
KERNEL=5
DILATIONS=(1,2,4,8,16,32,64)
GRID_MM=2.
OBSERVATIONS=107
BATCH_SIZE=32
NAMES=section.NAMES
PERMUTATION=np.arange(OBSERVATIONS)
for a,b in ((12,14),(23,24),(27,28),(29,30)):
    i,j=section.legacy.RAW_COLUMNS.index(a),section.legacy.RAW_COLUMNS.index(b)
    PERMUTATION[[i,j]]=PERMUTATION[[j,i]]
assert np.array_equal(PERMUTATION[PERMUTATION],np.arange(OBSERVATIONS))


def candidate_rows(features,packet,labels=None,weight=None):
    """Resample observations only; supervision stays on every original station."""
    obs=section.observations(features,packet);result=[]
    for ci,c in enumerate(packet['candidates']):
        lo,hi=packet['offsets'][ci:ci+2];s,receipt=reference.parameters(c);length=float(s[-1])
        p=np.asarray(c['points']);closed=bool(np.linalg.norm(p[0]-p[-1])<=1e-6)
        n=max(3,int(math.ceil(length/GRID_MM)))if closed else max(2,int(math.ceil(length/GRID_MM))+1)
        step=length/(n if closed else n-1);grid=np.arange(n)*step
        values=np.column_stack([np.interp(grid,s,obs[lo:hi,j])for j in range(OBSERVATIONS)]).astype(np.float32)
        position=s/step;position[0]=0.;position[-1]=n if closed else n-1
        left=np.floor(position).astype(np.int64);alpha=(position-left).astype(np.float32)
        if closed:left%=n;right=(left+1)%n
        else:left=np.clip(left,0,n-1);right=np.minimum(left+1,n-1)
        row=dict(candidate=ci,lo=lo,hi=hi,grid=values,grid_step_mm=step,closed=closed,
            station=s,length_mm=length,left=left,right=right,alpha=alpha,native_reference=receipt)
        if labels is not None:
            assert weight is not None
            row.update(labels=np.asarray(labels[lo:hi],np.float32),weight=np.asarray(weight[lo:hi],np.float32))
        result.append(row)
    return result


def batches(rows,rng=None):
    groups={}
    order=np.arange(len(rows))if rng is None else rng.permutation(len(rows))
    for i in order:groups.setdefault(int(math.ceil(math.log2(len(rows[i]['grid'])))),[]).append(int(i))
    result=[]
    for key in sorted(groups):
        ids=groups[key]
        result.extend(ids[i:i+BATCH_SIZE]for i in range(0,len(ids),BATCH_SIZE))
    if rng is not None:rng.shuffle(result)
    return result


def pack(rows,mean,std,device):
    size=np.asarray([len(q['grid'])for q in rows],np.int64);width=int(size.max())
    source=np.zeros((len(rows),OBSERVATIONS,width),np.float32)
    original=max(len(q['station'])for q in rows)
    left=np.zeros((len(rows),original),np.int64);right=left.copy()
    alpha=np.zeros((len(rows),original),np.float32);weight=alpha.copy();labels=alpha.copy()
    for i,row in enumerate(rows):
        source[i,:,:size[i]]=((row['grid']-mean)/std).T
        n=len(row['station']);left[i,:n]=row['left'];right[i,:n]=row['right'];alpha[i,:n]=row['alpha']
        if 'labels'in row:labels[i,:n]=row['labels'];weight[i,:n]=row['weight']
    arrays=dict(x=source,size=size,closed=np.asarray([q['closed']for q in rows],bool),
        left=left,right=right,alpha=alpha,weight=weight,labels=labels)
    return {key:torch.as_tensor(value,device=device)for key,value in arrays.items()}


def reverse_index(size,closed,width):
    index=torch.arange(width,device=size.device)[None,:]
    periodic=(-index)%size[:,None]
    opened=torch.clamp(size[:,None]-1-index,min=0)
    return torch.where(closed[:,None],periodic,opened)


def original_logits(grid,batch):
    return grid.gather(1,batch['left'])*(1-batch['alpha'])+grid.gather(1,batch['right'])*batch['alpha']


class BoundaryDepthwise(nn.Module):
    def __init__(self,dilation):
        super().__init__();self.dilation=dilation
        self.spatial=nn.Conv1d(CHANNELS,CHANNELS,KERNEL,dilation=dilation,groups=CHANNELS)
        self.mix=nn.Conv1d(CHANNELS,CHANNELS,1)

    def forward(self,x,size,closed,mask):
        padding=(KERNEL//2)*self.dilation
        index=torch.arange(-padding,x.shape[-1]+padding,device=x.device)[None,:]
        periodic=index%size[:,None]
        opened=torch.minimum(index.clamp(min=0),size[:,None]-1)
        index=torch.where(closed[:,None],periodic,opened)
        padded=x.gather(2,index[:,None,:].expand(-1,x.shape[1],-1))
        delta=self.mix(F.silu(self.spatial(padded)))
        return (x+.1*delta)*mask


class CurveNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Conv1d(OBSERVATIONS,CHANNELS,1)
        self.blocks=nn.ModuleList([BoundaryDepthwise(d)for d in DILATIONS])
        self.head=nn.Sequential(nn.Conv1d(3*CHANNELS,CHANNELS,1),nn.SiLU(),nn.Conv1d(CHANNELS,1,1))
        self.register_buffer('reverse_channels',torch.as_tensor(PERMUTATION.copy(),dtype=torch.long))

    def forward(self,x,size,closed):
        mask=(torch.arange(x.shape[-1],device=x.device)[None,None,:]<size[:,None,None]).to(x.dtype)
        z=F.silu(self.input(x))*mask
        for block in self.blocks:z=block(z,size,closed,mask)
        average=z.sum(2,keepdim=True)/size[:,None,None]
        maximum=z.masked_fill(mask==0,-torch.inf).amax(2,keepdim=True)
        joined=torch.cat([z,average.expand_as(z),maximum.expand_as(z)],dim=1)
        return self.head(joined)[:,0]*mask[:,0]

    def both_directions(self,batch):
        x,size,closed=batch['x'],batch['size'],batch['closed']
        index=reverse_index(size,closed,x.shape[-1])
        reverse=x.gather(2,index[:,None,:].expand(-1,x.shape[1],-1))[:,self.reverse_channels,:]
        # Each direction uses the same geometry boundary convention and weights.
        a=self.forward(x,size,closed)
        b=self.forward(reverse,size,closed).gather(1,index)
        return a,b


def probabilities(network,rows,mean,std,device='cpu'):
    if str(device)=='cpu'and torch.get_num_threads()!=2:torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    result=[None]*len(rows);network.eval()
    with torch.no_grad():
        for ids in batches(rows):
            selected=[rows[i]for i in ids];batch=pack(selected,mean,std,device)
            a,b=network.both_directions(batch)
            logits=original_logits((a+b)*.5,batch)
            positive=torch.sigmoid(logits).cpu().numpy()
            for j,i in enumerate(ids):
                values=positive[j,:len(rows[i]['station'])].astype(float)
                result[i]=np.column_stack([1-values,values])
    return result


class SequenceCurveSet:
    def __init__(self,network,mean,std,prior):
        self.network=network.cpu().eval();self.mean=np.asarray(mean,np.float32);self.std=np.asarray(std,np.float32)
        self.prior=dict(prior);self.classes_=np.array([0,1],np.int64)
        self.last_states=self.last_timing=self.last_probability=None

    def predict(self,features,packet):
        start=time.perf_counter();rows=candidate_rows(features,packet)
        probability=probabilities(self.network,rows,self.mean,self.std)
        proposals=[];states=[]
        for row,pp in zip(rows,probability):
            ci=row['candidate'];points=np.asarray(packet['candidates'][ci]['points'])
            runs,labels,objective=sequence.intervals(row['station'],pp,row['closed'],row['length_mm'],self.prior)
            states.append(dict(candidate=ci,intervals=runs,native_reference=row['native_reference'],objective=objective,
                changed_station_count=int((labels!=pp.argmax(1)).sum())))
            for j,q in enumerate(runs):
                proposals.append(dict(candidate=ci,points=geometry.native_slice(points,q['start_mm'],q['end_mm']).tolist(),
                    weld_type='horizontal',confidence=q['confidence'],
                    interval=dict(start_mm=q['start_mm'],end_mm=q['end_mm'],index=j,predicted_count=len(runs))))
        self.last_states=states;self.last_probability=np.concatenate(probability)if probability else np.empty((0,2))
        self.last_timing=dict(cpu_seconds=time.perf_counter()-start,input_observations=OBSERVATIONS,
            parameter_count=sum(p.numel()for p in self.network.parameters()),native_grid_max_mm=GRID_MM,
            native_grid_rows=sum(len(q['grid'])for q in rows),original_stations=len(features),whole_curve_context=True,
            reverse_symmetric_logits=True,no_count_templates_minimum_length_or_gap_fill=True)
        return proposals


input_curves=section.input_curves
arclength=section.arclength


def decode(features,packet,model,threshold=.5):
    assert isinstance(model,SequenceCurveSet)
    return original_runtime.decode(features,packet,model,threshold)
