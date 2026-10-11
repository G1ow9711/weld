"""Two separately saved models, with no trainable feature sharing."""
import torch
from torch.nn import functional as F
from weld_pipeline.sjg_contact_tcn_1052 import ContactTCN

def semantic(raw):
    flags=raw[:,:2];state=raw[:,2:18].reshape(-1,8,2);numeric=F.softplus(raw[:,18:])
    return dict(process_flags=flags,process_state=state,process_numeric=numeric,
        process_code=torch.cat((flags.sigmoid(),state.softmax(-1).flatten(1),numeric),1))

class ProcessModel(torch.nn.Module):
    def __init__(self,features=669):
        super().__init__();self.encoder=torch.nn.Linear(features,64);self.head=torch.nn.Linear(64,26)
    def forward(self,x):return semantic(self.head(F.gelu(self.encoder(x))))

class ExtractionModel(ContactTCN):
    def __init__(self,features=671):
        super().__init__(features,64);self.boundary=torch.nn.Linear(64,8);self.fusion=torch.nn.Linear(26,64,bias=False)
        torch.nn.init.normal_(self.boundary.weight,std=.01);torch.nn.init.zeros_(self.boundary.bias);torch.nn.init.zeros_(self.fusion.weight)
    def forward(self,x,index,process_code):
        h=F.silu(self.input(x))
        for k in range(4):
            ids=index[k];valid=ids>=0;safe=ids.clamp_min(0)
            v=h*self.kernel[k,1]+h[safe[:,0]]*valid[:,0,None]*self.kernel[k,0]+h[safe[:,1]]*valid[:,1,None]*self.kernel[k,2]
            h=self.norm[k](h+F.silu(self.mix[k](v)))
        h=h+self.fusion(process_code.detach());c=self.classify(h);fractions=self.position(h).softmax(-1)
        return dict(occupancy=c[:,:3],event=c[:,3:11].reshape(-1,2,4),direction=c[:,11],closure=c[:,12],
            weld_type=c[:,13:15],positions=fractions.cumsum(-1)[:,:2],boundary=self.boundary(h).reshape(-1,2,4))

class ProcessBags:
    """Original-record macro objective, geometric witnesses do not add weight."""
    def __init__(self,records,device):
        self.records=[];self.selected={}
        for key in sorted({r['source'] for r in records}):
            ids=sorted({f for r in records if r['source']==key for c in r['children'] for f,d in c})
            self.selected[key]=ids;mapping={f:i for i,f in enumerate(ids)}
            for r in records:
                if r['source']!=key:continue
                children=[(torch.tensor([mapping[f] for f,d in c],device=device),torch.tensor([d for f,d in c],device=device)) for c in r['children']]
                self.records.append((r,children))
    def loss(self,outputs):
        groups={};bool_correct=[];state_correct=[];errors=[]
        def add(key,v):groups.setdefault(key,[]).append(v)
        reverse=[1,0,3,2,5,4,7,6]
        for r,children in self.records:
            out=outputs[r['source']];key=r['source']
            # Each child's geometric alternatives average first. Parent Any is OR.
            child_flags=torch.stack([out['process_flags'][ids].sigmoid().mean(0) for ids,d in children])
            prob=1-torch.prod(1-child_flags,dim=0);prob=prob.clamp(1e-7,1-1e-7)
            for k,valid in enumerate(r['flag_mask']):
                if valid:
                    t=r['flags'][k];add((key,'bool',k,t),F.binary_cross_entropy(prob[k],prob.new_tensor(float(t))))
                    bool_correct.append((prob[k]>=.5).float()==t)
            for k,state in enumerate(r['state']):
                if state<0:continue
                chosen=children if k<2 else [children[0] if k%2==0 else children[-1]]
                ps=[];ns=[]
                for ids,direction in chosen:
                    columns=torch.where(direction,k,reverse[k]);ps.append(out['process_state'][ids,columns].softmax(-1).mean(0));ns.append(out['process_numeric'][ids,columns].mean())
                p=torch.stack(ps).mean(0).clamp_min(1e-7);n=torch.stack(ns).mean()
                add((key,'state',k,state),-p[state].log());state_correct.append(p.argmax()==state)
                if r['numeric_mask'][k]:
                    delta=n-r['numeric'][k];add((key,'numeric',k,0),F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=1/50));errors.append(delta.abs()*50)
        # Equal sources; equal fields; equal known states within each field.
        fields={}
        for (source,kind,field,state),values in groups.items():fields.setdefault((source,kind,field),[]).append(torch.stack(values).mean())
        sources={}
        for (source,kind,field),values in fields.items():sources.setdefault(source,[]).append(torch.stack(values).mean())
        loss=torch.stack([torch.stack(v).mean() for v in sources.values()]).mean()
        metrics=dict(loss=float(loss.detach()),known_bool=len(bool_correct),bool_accuracy=float(torch.stack(bool_correct).float().mean()),
            known_states=len(state_correct),state_accuracy=float(torch.stack(state_correct).float().mean()),exact_numeric=len(errors),numeric_MAE_mm=float(torch.stack(errors).mean().detach()))
        return loss,metrics
