"""Local point heatmaps conditioned on point-derived contact relation hypotheses."""
import numpy as np
import torch
from torch import nn

from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame, restore_probability_order
from weld_pipeline.point_plate_context import contact_descriptor
from weld_pipeline.typed_contact_model import ChannelNorm, Residual

VERSION='joint_relation_heatmaps_v1'
LOCAL_CHANNELS=np.r_[0:49,55:67,70:84]
LOCAL_COUNT=75
NODE_COUNT=233
EDGE_COUNT=8
RELATIONS=('same_two_contact_planes','near_shared_plane','near_joint')


def segment_distance(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    u,v=a[1]-a[0],b[1]-b[0]
    la,lb=np.linalg.norm(u),np.linalg.norm(v)
    u/=la;v/=lb
    result=[]
    for q,line,tangent,length in [(a[0],b,v,lb),(a[1],b,v,lb),(b[0],a,u,la),(b[1],a,u,la)]:
        p=line[0]+np.clip((q-line[0])@tangent,0.,length)*tangent
        result.append(float(np.linalg.norm(q-p)))
    cosine=float(u@v);denominator=1.-cosine*cosine
    if denominator>1e-8:
        delta=b[0]-a[0]
        s=(delta@u-cosine*(delta@v))/denominator
        t=(cosine*(delta@u)-delta@v)/denominator
        if 0<=s<=la and 0<=t<=lb:
            result.append(float(np.linalg.norm(a[0]+s*u-b[0]-t*v)))
    return min(result)


def make_graph(xyz,sequences,structural,process,profile):
    selected=[index for index,s in enumerate(sequences) if s.get('curve_kind')!='arc']
    nodes,records,canonical=[],[],{}
    for index in selected:
        sequence,_,transform=canonical_frame(sequences[index])
        canonical[index]=(sequence,transform)
        mid=(sequence['xyz'][0].astype(float)+sequence['xyz'][-1].astype(float))/2.
        normal=sequence['features'][len(sequence['features'])//2,60:66].astype(float).reshape(2,3)
        direction=np.asarray(sequence['direction'],float)
        length=sequence['sample_step_mm']*(len(sequence['xyz'])-1)
        contact=np.r_[contact_descriptor(sequence)]
        standing=normal[np.abs(normal[:,2])<.4]
        finite=np.zeros(39,np.float32)
        if len(standing):
            finite,_=profile.observe(mid,standing[0])
        kind=[float(sequence['type']=='horizontal'),float(sequence['type']=='vertical')]
        features=np.r_[contact,structural[index],process[index],length/1000.,direction,normal.ravel(),kind,finite].astype(np.float32)
        if features.shape!=(NODE_COUNT,) or not np.isfinite(features).all():
            raise ValueError('Invalid PCD contact node features')
        nodes.append(features)
        records.append(dict(sequence=index,mid=mid,direction=direction,normals=normal,length=length,
                            endpoints=sequence['xyz'][[0,-1]].astype(float)))
    edge_index,edge_type,edge_features=[],[],[]
    for i,a in enumerate(records):
        for j in range(i+1,len(records)):
            b=records[j];delta=b['mid']-a['mid']
            similarity=np.abs(a['normals']@b['normals'].T)
            matching=(similarity>.9999)&(np.abs(a['normals']@delta)[:,None]<=.3)
            shared=int(matching.any(1).sum())
            distance=segment_distance(a['endpoints'],b['endpoints'])
            same=bool(shared==2 and matching.any(0).all())
            if same and distance<=80.:
                relation=0
            elif shared and distance<=80.:
                relation=1
            elif distance<=6.:
                relation=2
            else:
                continue
            for source,target in [(i,j),(j,i)]:
                s,t=records[source],records[target];d=t['mid']-s['mid']
                local=np.stack((s['direction'],*s['normals']))@d/1000.
                feature=np.r_[distance/80.,abs(s['direction']@t['direction']),shared/2.,
                              min(s['length'],t['length'])/max(s['length'],t['length']),
                              np.clip(local,-20.,20.),float(same)].astype(np.float32)
                edge_index.append((source,target));edge_type.append(relation);edge_features.append(feature)
    return dict(nodes=np.asarray(nodes,np.float32),
        edge_index=np.asarray(edge_index,np.int64).reshape(-1,2).T,
        edge_type=np.asarray(edge_type,np.int64),edge_features=np.asarray(edge_features,np.float32).reshape(-1,EDGE_COUNT),
        sequence_indices=np.asarray(selected,np.int64),canonical=canonical,
        materials_independently_verified=False)


class JointRelationHeatmaps(nn.Module):
    def __init__(self,messages,width=128):
        super().__init__();self.messages=bool(messages);self.width=width
        self.node_encoder=nn.Sequential(nn.Linear(NODE_COUNT,width),nn.LayerNorm(width),nn.GELU())
        self.self_layers=nn.ModuleList([nn.Linear(width,width) for _ in range(3)])
        self.relations=nn.ModuleList([nn.ModuleList([nn.Linear(width,width,bias=False) for _ in RELATIONS]) for _ in range(3)])
        self.edge_layers=nn.ModuleList([nn.Linear(EDGE_COUNT,width,bias=False) for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.condition=nn.Linear(width,2*width)
        nn.init.zeros_(self.condition.weight);nn.init.zeros_(self.condition.bias)
        self.point_encoder=nn.Sequential(nn.Conv1d(LOCAL_COUNT,width,1),ChannelNorm(width),nn.GELU())
        self.point_layers=nn.Sequential(*[Residual(width,d) for d in (1,4,16)])
        self.output=nn.Conv1d(width,4,1)

    def encode_graph(self,nodes,edges,edge_type,edge_features):
        h=self.node_encoder(nodes)
        source,target=edges
        for layer in range(3):
            update=self.self_layers[layer](h)
            if self.messages and len(edge_type):
                aggregation=torch.zeros_like(h)
                degree=torch.zeros((len(h),1),device=h.device,dtype=h.dtype)
                for relation in range(len(RELATIONS)):
                    selected=edge_type==relation
                    if not bool(selected.any()):
                        continue
                    message=self.relations[layer][relation](h[source[selected]])+self.edge_layers[layer](edge_features[selected])
                    aggregation.index_add_(0,target[selected],message)
                    degree.index_add_(0,target[selected],torch.ones((int(selected.sum()),1),device=h.device,dtype=h.dtype))
                update=update+aggregation/degree.clamp_min(1.)
            h=self.norms[layer](h+torch.nn.functional.gelu(update))
        return h

    def point_forward(self,point_features,node_context):
        x=self.point_encoder(point_features)
        scale,bias=self.condition(node_context).chunk(2,dim=1)
        x=x*(1.+.1*scale.tanh()[:,:,None])+.1*bias[:,:,None]
        return self.output(self.point_layers(x))

    def forward(self,point_features,node_indices,nodes,edges,edge_type,edge_features):
        h=self.encode_graph(nodes,edges,edge_type,edge_features)
        return self.point_forward(point_features,h[node_indices])


@torch.no_grad()
def infer_cached(model,data,sequences,chunk_size=1024,halo=64,batch_size=8):
    model.eval();device=next(model.parameters()).device
    graph=data['torch_graph'];context=model.encode_graph(*graph)
    result=[np.zeros((len(sequence['features']),7),np.float32) for sequence in sequences]
    blocks=[]
    for entry in data['entries']:
        if entry['type']!='horizontal':
            continue
        length=entry['stop']-entry['start']
        for start in range(0,length,chunk_size):
            stop=min(length,start+chunk_size)
            blocks.append((entry,start,stop))
    for offset in range(0,len(blocks),batch_size):
        items=blocks[offset:offset+batch_size]
        size=max(stop-start+2*halo for _,start,stop in items)
        batch=np.zeros((len(items),LOCAL_COUNT,size),np.float32);indices=[]
        for bi,(entry,start,stop) in enumerate(items):
            length=entry['stop']-entry['start'];lo=start-halo
            a,b=max(0,lo),min(length,stop+halo)
            batch[bi,:,a-lo:b-lo]=data['features'][entry['start']+a:entry['start']+b].T
            indices.append(entry['node'])
        predicted=model.point_forward(torch.from_numpy(batch).to(device),context[torch.tensor(indices,device=device)]).sigmoid().cpu().numpy()
        for bi,(entry,start,stop) in enumerate(items):
            canonical=result[entry['sequence']]
            canonical[start:stop,:4]=predicted[bi,:,halo:halo+stop-start].T
    for entry in data['entries']:
        if entry['type']=='horizontal':
            result[entry['sequence']]=restore_probability_order(result[entry['sequence']],entry['transform'])
    return result
