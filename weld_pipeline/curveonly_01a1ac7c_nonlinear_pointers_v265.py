"""Nonlinear residual point/pair scoring; original134observations and span grammar."""
import math
import numpy as np
import torch
from torch import nn
from . import curveonly_01a1ac7c_direction_spans_v247 as previous
from .curveonly_01a1ac7c_corner_publication_v256 import decode

CHANNELS = previous.CHANNELS
OBSERVATIONS, NAMES, PERMUTATION = previous.OBSERVATIONS, previous.NAMES, previous.PERMUTATION
codec, base, normalize, arclength = previous.codec, previous.base, previous.normalize, previous.arclength
candidate_rows, input_curves, pack, sequence = previous.candidate_rows, previous.input_curves, previous.pack, previous.sequence


class OrderedSpanNetwork(previous.OrderedSpanNetwork):
    def __init__(self):
        super().__init__()
        self.start_residual = nn.Linear(CHANNELS,1)
        self.end_residual = nn.Linear(CHANNELS,1)
        self.distance_projection = nn.Linear(2,CHANNELS,bias=False)
        for layer in (self.start_residual,self.end_residual):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def start_logits(self,context,encoded):
        query = self.start_query(context)
        if context.ndim == 3:
            original = torch.einsum('btc,bmc->btm',query,encoded['skey'])/math.sqrt(CHANNELS)
            nonlinear = torch.tanh(query[:,:,None,:]+encoded['skey'][:,None,:,:])
        else:
            original = torch.einsum('bc,bmc->bm',query,encoded['skey'])/math.sqrt(CHANNELS)
            nonlinear = torch.tanh(query[:,None,:]+encoded['skey'])
        return original+self.start_residual(nonlinear)[...,0]

    def end_logits(self,context,start_feature,start_position,encoded):
        query = self.end_query(torch.cat([context,start_feature,start_position],-1))
        if context.ndim == 3:
            original = torch.einsum('btc,bmc->btm',query,encoded['ekey'])/math.sqrt(CHANNELS)
            delta = encoded['position'][:,None,:]-start_position
            nonlinear = query[:,:,None,:]+encoded['ekey'][:,None,:,:]
        else:
            original = torch.einsum('bc,bmc->bm',query,encoded['ekey'])/math.sqrt(CHANNELS)
            delta = encoded['position']-start_position
            nonlinear = query[:,None,:]+encoded['ekey']
        distance = torch.stack([delta,delta.square()],-1)
        return original+self.end_residual(torch.tanh(nonlinear+self.distance_projection(distance)))[...,0]

    def attend(self, encoded, hidden, prefix, previous_start, starts, ends, span_mask):
        pooled = encoded['pooled'][:,None,:].expand(-1,hidden.shape[1],-1)
        context = torch.cat([hidden,pooled,prefix],2)
        start_feature = self.gather(encoded['virtual'],starts)
        end_feature = self.gather(encoded['virtual'],ends)
        start_position = encoded['position'].gather(1,starts)[:,:,None]
        end_position = encoded['position'].gather(1,ends)[:,:,None]
        start_logits = self.start_logits(context,encoded)
        end_logits = self.end_logits(context,start_feature,start_position,encoded)
        index = encoded['index'][:,None,:]
        smask = (index > previous_start[:,:,None]) & (index <= encoded['start_cap'][:,None,None])
        emask = (index >= starts[:,:,None]) & (index < encoded['cap'][:,None,None])
        emask &= (~encoded['closed'][:,None,None]) | (index <= starts[:,:,None]+encoded['size'][:,None,None])
        # Non-span slots are never scored, but retain a finite softmax for gradients.
        smask = torch.where(span_mask[:,:,None],smask,index==0)
        emask = torch.where(span_mask[:,:,None],emask,index==0)
        start_logits = start_logits.masked_fill(~smask,-1e9)
        end_logits = end_logits.masked_fill(~emask,-1e9)
        offset = .5*torch.tanh(self.offset_head(torch.cat([context,start_feature,end_feature,start_position,end_position],2)))
        return dict(continue_logit=self.continue_head(context)[:,:,0],
            full_logit=self.full_head(context)[:,:,0],start=start_logits,end=end_logits,offset=offset)

    def teacher(self,batch):
        encoded = self.encode(batch['x'],batch['size'],batch['closed'])
        starts,ends = batch['starts'],batch['ends']
        first = self.gather(encoded['virtual'],starts[:,:-1])
        last = self.gather(encoded['virtual'],ends[:,:-1])
        token = self.token(torch.cat([first,last,batch['coordinates'][:,:-1]],2))
        if token.shape[1]:
            after,_ = self.recurrent(token,encoded['initial'][None,:,:])
            hidden = torch.cat([encoded['initial'][:,None,:],after],1)
        else:
            hidden = encoded['initial'][:,None,:]
        return self.attend(encoded,hidden,batch['prefix'],batch['previous_start'],starts,ends,batch['span_weight']>0)

    def generate(self,encoded,rows):
        device = encoded['virtual'].device
        batch_size = len(rows)
        hidden = encoded['initial'];prefix = torch.zeros((batch_size,4),device=device)
        previous_start = torch.full((batch_size,),-1,dtype=torch.long,device=device)
        active = torch.ones(batch_size,dtype=torch.bool,device=device)
        ranges,steps = [[] for _ in rows],[[] for _ in rows]
        diagnostics = [np.zeros((3,len(row['grid'])),np.float32) for row in rows]
        limit = int(encoded['start_cap'].max())+2
        for turn in range(limit):
            possible = previous_start < encoded['start_cap']
            active &= possible
            if not bool(active.any()):break
            context = torch.cat([hidden,encoded['pooled'],prefix],1)
            probability = torch.sigmoid(self.continue_head(context)[:,0])
            cycle = torch.sigmoid(self.full_head(context)[:,0])
            continuing = active & (probability>.5)
            full = continuing & encoded['closed'] & (turn==0) & (cycle>.5)
            for j in torch.nonzero(active).flatten().tolist():
                steps[j].append(dict(step=turn,continue_probability=float(probability[j]),
                    full_cycle_probability=float(cycle[j]),continue_selected=bool(continuing[j]),
                    termination='LEARNED_STOP' if not bool(continuing[j]) else 'FULL_CYCLE' if bool(full[j]) else None))
            for j in torch.nonzero(full).flatten().tolist():
                ranges[j].append(dict(a=0.,b=rows[j]['length_mm'],full=True,confidence=float(probability[j])))
                diagnostics[j][0]=float(probability[j])
            active = continuing & ~full
            if not bool(active.any()):break
            logits = self.start_logits(context,encoded)
            valid = (encoded['index']>previous_start[:,None]) & (encoded['index']<=encoded['start_cap'][:,None])
            valid = torch.where(active[:,None],valid,encoded['index']==0)
            logits = logits.masked_fill(~valid,-1e9)
            start = logits.argmax(1)
            zs = encoded['virtual'][torch.arange(batch_size,device=device),start]
            ps = encoded['position'].gather(1,start[:,None])
            end_logits = self.end_logits(context,zs,ps,encoded)
            valid = (encoded['index']>=start[:,None]) & (encoded['index']<encoded['cap'][:,None])
            valid &= (~encoded['closed'][:,None]) | (encoded['index']<=start[:,None]+encoded['size'][:,None])
            valid = torch.where(active[:,None],valid,encoded['index']==0)
            end_logits = end_logits.masked_fill(~valid,-1e9)
            end = end_logits.argmax(1)
            ze = encoded['virtual'][torch.arange(batch_size,device=device),end]
            pe = encoded['position'].gather(1,end[:,None])
            offset = .5*torch.tanh(self.offset_head(torch.cat([context,zs,ze,ps,pe],1)))
            same = start==end
            ordered,_ = offset.sort(1)
            offset = torch.where(same[:,None],ordered,offset)
            coordinates = torch.zeros((batch_size,4),device=device)
            sp = logits.softmax(1).cpu().numpy();ep = end_logits.softmax(1).cpu().numpy()
            for j in torch.nonzero(active).flatten().tolist():
                row=rows[j];n=len(row['grid']);length=row['length_mm'];step=row['grid_step_mm']
                a=(int(start[j])+float(offset[j,0]))*step
                b=(int(end[j])+float(offset[j,1]))*step
                if row['closed']:
                    width=min(length,max(0.,b-a));a%=length;b=a+width
                else:a=max(0.,min(length,a));b=max(0.,min(length,b))
                steps[j][-1].update(start_cell=int(start[j]),end_cell=int(end[j]),
                    start_peak_probability=float(sp[j,int(start[j])]),end_peak_probability=float(ep[j,int(end[j])]),
                    predicted_start_mm=a,predicted_end_mm=b,offsets=offset[j].cpu().tolist())
                for virtual in range(int(encoded['cap'][j])):
                    diagnostics[j][1,virtual % n]=max(diagnostics[j][1,virtual % n],sp[j,virtual])
                    diagnostics[j][2,virtual % n]=max(diagnostics[j][2,virtual % n],ep[j,virtual])
                if b>a:
                    ranges[j].append(dict(a=a,b=b,full=False,confidence=float(probability[j])))
                    station=np.arange(n)*step
                    take=np.zeros(n,bool)
                    for shift in ((-1,0,1) if row['closed'] else (0,)):
                        take |= (station+shift*length>=a)&(station+shift*length<=b)
                    diagnostics[j][0,take]=np.maximum(diagnostics[j][0,take],float(probability[j]))
                coordinates[j]=torch.tensor([a/length,b/length,(b-a)/length,0.],device=device)
            token=self.token(torch.cat([zs,ze,coordinates],1))[:,None,:]
            after,_=self.recurrent(token,hidden[None,:,:]);hidden=after[:,0,:]
            prefix=coordinates;previous_start=torch.where(active,start,previous_start)
        else:
            assert not bool(active.any()),'Strict start-pointer progress failed to terminate'
        return ranges,steps,diagnostics


class OrderedSpanSet(previous.OrderedSpanSet):
    pass
