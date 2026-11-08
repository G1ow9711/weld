"""Ordered span pointers with learned stopping, continuous offsets and native cycles.

No fixed weld-count classes, foreground threshold, minimum weld/gap, or label input.
The existing finite geometry, primitive identity and export policy are reused.
"""
import math
import time
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import curveonly_01a1ac7c_periodic_profile_v227 as previous

codec, base = previous.codec, previous.base
NAMES, OBSERVATIONS = previous.NAMES, 111
CHANNELS = 48
PERMUTATION = codec.previous.PERMUTATION
TOL = codec.reference.PROJECTION_NUMERIC_MM
candidate_rows, input_curves, arclength = previous.candidate_rows, previous.input_curves, previous.arclength
normalize = previous.normalize


def sequence(row, choices, reverse=False):
    """All original ranges, with an unwrapped second period for closed spans."""
    length, step, n = row['length_mm'], row['grid_step_mm'], len(row['grid'])
    result = []
    for choice in choices:
        a, b = float(choice['start_mm']), float(choice['end_mm'])
        width = b-a
        assert width > 0
        full = bool(row['closed'] and abs(width-length) <= TOL)
        if reverse:
            a, b = length-b, length-a
        if full:
            a, b = 0., length
        elif row['closed']:
            a %= length
            b = a+width
        else:
            assert a >= -TOL and b <= length+TOL
            a, b = max(0., a), min(length, b)
        cells = np.floor(np.array([a, b])/step+.5).astype(np.int64)
        offsets = np.array([a, b])/step-cells
        assert np.abs(offsets).max() <= .5+1e-7
        cap = 2*n if row['closed'] else n-1
        assert 0 <= cells[0] <= (n if row['closed'] else n-1)
        assert cells[0] <= cells[1] <= cap
        result.append(dict(truth=choice['truth'], a=float(a), b=float(b),
            start=int(cells[0]), end=int(cells[1]), offsets=offsets.tolist(), full=full))
    result.sort(key=lambda q:(q['a'], q['b'], q['truth']))
    if any(q['full'] for q in result):
        assert len(result) == 1
    for left, right in zip(result, result[1:]):
        assert right['start'] > left['start'], 'Ordered-span same-start-cell capacity conflict'
    return result


def intervals_to_codes(row, intervals):
    codes = []
    for item in intervals:
        if item.get('full'):
            code = dict(start_index=0, end_index=0, start_offset=0., end_offset=0., full_cycle=True)
        else:
            code = codec.encode(row, item['a'], item['b'])
        codes.append(dict(code, confidence=float(item.get('confidence', 1.))))
    return codes


def reverse_ranges(row, ranges):
    result = []
    length = row['length_mm']
    for item in ranges:
        a, b = length-item['b'], length-item['a']
        if row['closed'] and not item.get('full'):
            width = b-a;a %= length;b = a+width
        result.append(dict(item, a=float(a), b=float(b)))
    return result


def pack(rows, mean, scale, device, reverse=False):
    size = np.array([len(q['grid']) for q in rows], np.int64)
    width = int(size.max())
    x = np.zeros((len(rows), OBSERVATIONS, width), np.float32)
    sequences = [q['reverse_sequence'] if reverse else q['sequence'] for q in rows]
    steps = max((len(q) for q in sequences), default=0)+1
    arrays = dict(size=size, closed=np.array([q['closed'] for q in rows], bool),
        starts=np.zeros((len(rows), steps), np.int64), ends=np.zeros((len(rows), steps), np.int64),
        offsets=np.zeros((len(rows), steps, 2), np.float32),
        coordinates=np.zeros((len(rows), steps, 4), np.float32),
        prefix=np.zeros((len(rows), steps, 4), np.float32),
        previous_start=np.full((len(rows), steps), -1, np.int64),
        continue_target=np.zeros((len(rows), steps), np.float32),
        continue_weight=np.zeros((len(rows), steps), np.float32),
        span_weight=np.zeros((len(rows), steps), np.float32),
        full_target=np.zeros((len(rows), steps), np.float32),
        full_weight=np.zeros((len(rows), steps), np.float32))
    for i, (row, seq) in enumerate(zip(rows, sequences)):
        grid = (row['grid']-mean)/scale
        if reverse:
            order = (-np.arange(size[i])) % size[i] if row['closed'] else np.arange(size[i])[::-1]
            grid = grid[order][:, PERMUTATION]
        x[i, :, :size[i]] = grid.T
        for k, item in enumerate(seq):
            arrays['starts'][i,k], arrays['ends'][i,k] = item['start'], item['end']
            arrays['offsets'][i,k] = item['offsets']
            coordinate = np.array([item['a'], item['b'], item['b']-item['a'], 0.])/row['length_mm']
            coordinate[3] = item['full']
            arrays['coordinates'][i,k] = coordinate
            arrays['prefix'][i,k+1] = coordinate
            arrays['previous_start'][i,k+1] = item['start']
            weight = row['instance_weights'][item['truth']]
            arrays['continue_target'][i,k] = 1.
            arrays['continue_weight'][i,k] = row.get('continue_factor',.5)*weight
            if not item['full']:
                arrays['span_weight'][i,k] = weight
            if row['closed'] and k == 0:
                arrays['full_target'][i,k] = item['full']
                arrays['full_weight'][i,k] = sum(row['instance_weights'][q['truth']] for q in seq)
        arrays['continue_weight'][i,len(seq)] = row.get('continue_factor',.5)*row.get('stop_weight',0.)
    arrays['x'] = x
    return {k:torch.as_tensor(v, device=device) for k,v in arrays.items()}


class OrderedSpanNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        c = CHANNELS
        self.input = nn.Conv1d(OBSERVATIONS, c, 1)
        self.blocks = nn.ModuleList([base.BoundaryDepthwise(d) for d in base.DILATIONS])
        self.norm = nn.LayerNorm(c)
        self.initial = nn.Linear(2*c, c)
        self.token = nn.Sequential(nn.Linear(2*c+4, c), nn.SiLU())
        self.recurrent = nn.GRU(c, c, batch_first=True)
        context = 3*c+4
        self.continue_head = nn.Sequential(nn.Linear(context,c),nn.SiLU(),nn.Linear(c,1))
        self.full_head = nn.Sequential(nn.Linear(context,c),nn.SiLU(),nn.Linear(c,1))
        self.start_key = nn.Linear(c+2,c)
        self.end_key = nn.Linear(c+2,c)
        self.start_query = nn.Linear(context,c)
        self.end_query = nn.Linear(context+c+1,c)
        self.offset_head = nn.Sequential(nn.Linear(context+2*c+2,c),nn.SiLU(),nn.Linear(c,2))

    def encode(self, x, size, closed):
        mask = (torch.arange(x.shape[-1],device=x.device)[None,None,:] < size[:,None,None]).to(x.dtype)
        z = F.silu(self.input(x))*mask
        for block in self.blocks:
            z = block(z,size,closed,mask)
        z = self.norm(z.transpose(1,2)).transpose(1,2)*mask
        average = z.sum(2)/size[:,None]
        maximum = z.masked_fill(mask==0,-torch.inf).amax(2)
        pooled = torch.cat([average,maximum],1)
        cap = torch.where(closed,2*size+1,size)
        width = int(cap.max())
        index = torch.arange(width,device=x.device)[None,:].expand(len(x),-1)
        gather = torch.where(closed[:,None],index % size[:,None],torch.minimum(index,size[:,None]-1))
        virtual = z.gather(2,gather[:,None,:].expand(-1,CHANNELS,-1)).transpose(1,2)
        denominator = torch.where(closed,size,size-1).to(x.dtype)
        position = index/denominator[:,None]
        features = torch.cat([virtual,position[:,:,None],position.square()[:,:,None]],2)
        return dict(virtual=virtual,pooled=pooled,initial=torch.tanh(self.initial(pooled)),
            skey=self.start_key(features),ekey=self.end_key(features),position=position,
            index=index,cap=cap,start_cap=torch.where(closed,size,size-1),size=size,closed=closed)

    @staticmethod
    def gather(virtual,index):
        return virtual.gather(1,index[:,:,None].expand(-1,-1,CHANNELS))

    def attend(self, encoded, hidden, prefix, previous_start, starts, ends, span_mask):
        pooled = encoded['pooled'][:,None,:].expand(-1,hidden.shape[1],-1)
        context = torch.cat([hidden,pooled,prefix],2)
        start_feature = self.gather(encoded['virtual'],starts)
        end_feature = self.gather(encoded['virtual'],ends)
        start_position = encoded['position'].gather(1,starts)[:,:,None]
        end_position = encoded['position'].gather(1,ends)[:,:,None]
        start_logits = torch.einsum('btc,bmc->btm',self.start_query(context),encoded['skey'])/math.sqrt(CHANNELS)
        end_query = self.end_query(torch.cat([context,start_feature,start_position],2))
        end_logits = torch.einsum('btc,bmc->btm',end_query,encoded['ekey'])/math.sqrt(CHANNELS)
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
            logits = torch.einsum('bc,bmc->bm',self.start_query(context),encoded['skey'])/math.sqrt(CHANNELS)
            valid = (encoded['index']>previous_start[:,None]) & (encoded['index']<=encoded['start_cap'][:,None])
            valid = torch.where(active[:,None],valid,encoded['index']==0)
            logits = logits.masked_fill(~valid,-1e9)
            start = logits.argmax(1)
            zs = encoded['virtual'][torch.arange(batch_size,device=device),start]
            ps = encoded['position'].gather(1,start[:,None])
            query = self.end_query(torch.cat([context,zs,ps],1))
            end_logits = torch.einsum('bc,bmc->bm',query,encoded['ekey'])/math.sqrt(CHANNELS)
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


def predict(network,rows,mean,scale):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);network.cpu().eval()
    results=[None]*len(rows);states=[None]*len(rows);diagnostics=[None]*len(rows);full=[None]*len(rows)
    with torch.no_grad():
        for ids in base.batches(rows):
            selected=[dict(rows[i],sequence=[],reverse_sequence=[]) for i in ids]
            directions=[]
            for reverse in (False,True):
                batch=pack(selected,mean,scale,'cpu',reverse)
                encoded=network.encode(batch['x'],batch['size'],batch['closed'])
                directions.append(network.generate(encoded,selected))
            for j,i in enumerate(ids):
                forward,reverse=directions
                ranges=forward[0][j]+reverse_ranges(rows[i],reverse[0][j])
                codes=intervals_to_codes(rows[i],ranges)
                results[i]=codes
                order=(-np.arange(len(rows[i]['grid']))) % len(rows[i]['grid']) if rows[i]['closed'] else np.arange(len(rows[i]['grid']))[::-1]
                backward=reverse[2][j][:,order][[0,2,1]]
                diagnostics[i]=np.maximum(forward[2][j],backward)
                full[i]=max((q[0]['full_cycle_probability'] if q else 0.) for q in (forward[1][j],reverse[1][j]))
                states[i]=dict(candidate=rows[i]['candidate'],codes=codes,
                    forward_steps=forward[1][j],reverse_steps=reverse[1][j],
                    no_fixed_count_classes_or_foreground_event_gate=True,
                    direction_proposals_combined_before_frozen_source_selection=True)
    return results,states,diagnostics,full


class OrderedSpanSet:
    def __init__(self,network,mean,scale):
        self.network=network.cpu().eval();self.mean=np.asarray(mean,np.float32);self.scale=np.asarray(scale,np.float32)
        self.last_states=self.last_fields=self.last_full=self.last_timing=None
        self.last_representation_normalization=None

    def predict(self,features,packet):
        start=time.perf_counter();rows=candidate_rows(features,packet)
        codes,states,fields,full=predict(self.network,rows,self.mean,self.scale)
        raw=codec.proposals(packet,rows,codes)
        raw,receipt=normalize(raw,packet)
        self.last_states,self.last_fields,self.last_full=states,fields,full
        self.last_representation_normalization=receipt
        self.last_timing=dict(cpu_seconds=time.perf_counter()-start,
            parameter_count=sum(q.numel() for q in self.network.parameters()),
            paired_autoregressive_span_pointers_with_learned_stopping=True,
            diagnostic_fields=['published_candidate_support_confidence','maximum_executed_start_pointer_attention','maximum_executed_end_pointer_attention'],
            diagnostics_are_not_R32_independent_foreground_or_event_probabilities=True)
        return raw


def decode(features,packet,model,threshold=.5):
    return previous.decode(features,packet,model,threshold)
