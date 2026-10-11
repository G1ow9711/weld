"""Dense nearby-event offset supervision and point-only voting inference."""
import numpy as np
import torch
from torch import nn

from weld_pipeline.canonical_horizontal_patch_heatmaps import restore_probability_order
from weld_pipeline.event_gap_decoder import event_peaks
from weld_pipeline.joint_relation_heatmaps import JointRelationHeatmaps, LOCAL_COUNT

VERSION='joint_event_votes_v1'
RADIUS_MM=64.
SIGMA_MM=1.5
TAIL_MM=9.
VOTE_SCALE=4.


def dense_targets(target,step):
    target=np.asarray(target,np.float32)
    result=np.full((len(target),7),-1.,np.float32)
    result[:,0]=target[:,0]
    locations=np.arange(len(target),dtype=float)*step
    for channel in range(3):
        known=target[:,channel+1]>=0
        events=np.asarray([p*step for p,_ in event_peaks(np.maximum(target[:,channel+1],0),step,.5)])
        result[known,channel+1]=0.;result[known,channel+4]=0.
        if len(events):
            distances=events[None,:]-locations[:,None]
            nearest=np.argmin(np.abs(distances),axis=1)
            offsets=distances[np.arange(len(target)),nearest]
            supported=known&(np.abs(offsets)<=RADIUS_MM+1e-6)
            result[supported,channel+1]=1.
            result[supported,channel+4]=offsets[supported]/RADIUS_MM
    return result


def vote_heatmaps(foreground,support,offsets,step):
    foreground=np.asarray(foreground,float)
    support=np.asarray(support,float);offsets=np.asarray(offsets,float)
    if support.shape!=(len(foreground),3) or offsets.shape!=support.shape:
        raise ValueError('Invalid event-vote arrays')
    if not np.isfinite(foreground).all() or not np.isfinite(support).all() or not np.isfinite(offsets).all():
        raise ValueError('Finite event votes required')
    result=np.zeros((len(foreground),7),np.float32);result[:,0]=np.clip(foreground,0.,1.)
    locations=np.arange(len(foreground),dtype=float)*step
    for channel in range(3):
        mass=np.zeros(len(foreground),float)
        centers=locations+np.clip(offsets[:,channel],-1.,1.)*RADIUS_MM
        for center,weight in zip(centers,np.clip(support[:,channel],0.,1.)):
            if weight<=0 or center<-TAIL_MM or center>locations[-1]+TAIL_MM:continue
            lo=max(0,int(np.ceil((center-TAIL_MM)/step)))
            hi=min(len(mass),int(np.floor((center+TAIL_MM)/step))+1)
            if hi>lo:mass[lo:hi]+=weight*np.exp(-.5*((locations[lo:hi]-center)/SIGMA_MM)**2)
        result[:,channel+1]=-np.expm1(-mass/VOTE_SCALE)
    return result


class JointEventVotes(JointRelationHeatmaps):
    def __init__(self):
        super().__init__(messages=False,width=128)
        self.output=nn.Conv1d(self.width,7,1)


@torch.no_grad()
def infer_votes(model,data,sequences,chunk_size=1024,halo=64,batch_size=8):
    model.eval();device=next(model.parameters()).device
    context=model.encode_graph(*data['torch_graph'])
    raw={entry['sequence']:np.zeros((entry['stop']-entry['start'],7),np.float32)
         for entry in data['horizontal']}
    blocks=[]
    for entry in data['horizontal']:
        length=entry['stop']-entry['start']
        for start in range(0,length,chunk_size):blocks.append((entry,start,min(length,start+chunk_size)))
    for begin in range(0,len(blocks),batch_size):
        items=blocks[begin:begin+batch_size];size=max(b-a+2*halo for _,a,b in items)
        x=np.zeros((len(items),LOCAL_COUNT,size),np.float32);indices=[]
        for bi,(entry,a,b) in enumerate(items):
            length=entry['stop']-entry['start'];lo=a-halo
            left,right=max(0,lo),min(length,b+halo)
            x[bi,:,left-lo:right-lo]=data['features'][entry['start']+left:entry['start']+right].T
            indices.append(entry['node'])
        prediction=model.point_forward(torch.from_numpy(x).to(device),context[torch.tensor(indices,device=device)]).cpu().numpy()
        for bi,(entry,a,b) in enumerate(items):raw[entry['sequence']][a:b]=prediction[bi,:,halo:halo+b-a].T
    result=[np.zeros((len(s['xyz']),7),np.float32) for s in sequences]
    for entry in data['horizontal']:
        index=entry['sequence'];r=raw[index]
        sigmoid=lambda v:1./(1.+np.exp(-np.clip(v,-60.,60.)))
        p=vote_heatmaps(sigmoid(r[:,0]),sigmoid(r[:,1:4]),r[:,4:7],sequences[index]['sample_step_mm'])
        result[index]=restore_probability_order(p,entry['transform'])
    return result
