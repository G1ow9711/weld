"""Persistent normalized query positions and query self-attention, Pro278.

Only 256 query objects interact; every memory/profile remains in cross-attention.
No runtime count oracle, query orthogonality target, candidate pruning, or NMS.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F
from weld_pipeline.sjg_parent_set_network_1075 import ParentSetModel


class PersistentQueryModel(ParentSetModel):
    def __init__(self,**kwargs):
        kwargs.setdefault('bin_channels',247)
        super().__init__(**kwargs)
        self.query_attention=nn.ModuleList([nn.MultiheadAttention(self.width,4,dropout=0.) for _ in self.cross])
        for layer in self.query_attention:
            nn.init.zeros_(layer.out_proj.weight);nn.init.zeros_(layer.out_proj.bias)
        self.mechanism_enabled=True
        self.capture_mechanism=False

    def decode_queries(self,objects,chunk=2048):
        if not self.mechanism_enabled:return super().decode_queries(objects,chunk)
        position=F.layer_norm(self.parent_query,(self.width,))
        if self.capture_mechanism:
            position.retain_grad();self.captured_position=position
            self.captured_layers=[];self.captured_attention=[]
        query=self.parent_query
        for cross,self_attention in zip(self.cross,self.query_attention):
            value=F.layer_norm(query,(self.width,))
            update,_=self_attention(value+position,value+position,value,need_weights=False)
            query=query+update
            if self.capture_mechanism:
                # Diagnose the complete memory attention without storing a
                # gradient graph or modifying any denominator.
                with torch.no_grad():
                    q=cross.query(query+position).reshape(-1,4,self.width//4).transpose(0,1)
                    k=cross.key(objects).reshape(-1,4,self.width//4).transpose(0,1)
                    logits=q@k.transpose(-1,-2)/(self.width//4)**.5
                    probability=logits.softmax(-1)
                    mean=probability.mean(1,keepdim=True)
                    self.captured_attention.append(dict(objects=len(objects),
                        row_sum_max_error=float((probability.sum(-1)-1).abs().max()),
                        mean_entropy=float(-(probability*probability.clamp_min(1e-30).log()).sum(-1).mean()),
                        query_distribution_mean_squared_difference=float((probability-mean).square().mean())))
                    del q,k,logits,probability,mean
            query=cross(query,objects,chunk,query_position=position)
            if self.capture_mechanism:self.captured_layers.append(query.detach().clone())
        return query


def warm_start(checkpoint,device):
    # Keep training-only loss/scipy imports out of pure inference.
    from weld_pipeline.sjg_pair_set_revision_1094 import widen_bins
    model=PersistentQueryModel().to(device)
    state=copy.deepcopy(checkpoint['state_dict'])
    state['bin_encoder.weight']=widen_bins(state['bin_encoder.weight'])
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert not unexpected and missing and all(k.startswith('query_attention.') for k in missing)
    old=[p for name,p in model.named_parameters() if not name.startswith('query_attention.')]
    new=list(model.query_attention.parameters())
    optimizer=torch.optim.AdamW(old,lr=.0003,weight_decay=.0001)
    opt=copy.deepcopy(checkpoint['optimizer_state'])
    for entry in opt['state'].values():
        for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if name in entry:entry[name]=widen_bins(entry[name])
    optimizer.load_state_dict(opt)
    optimizer.add_param_group(dict(params=new,lr=.0003,weight_decay=.0001))
    return model,optimizer
