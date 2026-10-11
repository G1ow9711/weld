"""All281 fields from pure-source context. No GT is read by forward()."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .mixed_source_graph_heads_279 import MixedSourceGraphHeads, readonly_tensor
from .typed_prediction_codec_281 import CATEGORIES, NUMERIC_NAMES, RELATIONS, DOMAIN_KINDS


class FullyTypedGraphHeads(MixedSourceGraphHeads):
    def __init__(self,hidden=64):
        super().__init__(hidden)
        self.numeric=nn.Linear(hidden,6*len(NUMERIC_NAMES))
        self.category=nn.ModuleDict({name:nn.Linear(hidden,6*len(values)) for name,values in CATEGORIES.items()})
        self.domain_query=nn.Linear(hidden,6*len(DOMAIN_KINDS)*hidden)
        self.relation_embedding=nn.Embedding(len(RELATIONS),hidden)
        self.relation_order=nn.Sequential(nn.Linear(3*hidden,hidden),nn.SiLU(),nn.Linear(hidden,2))
        self.domain_set_order=nn.Sequential(nn.Linear(2*hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
        self.gap_class=nn.Sequential(nn.Linear(3*hidden,hidden),nn.SiLU(),nn.Linear(hidden,2))

    def forward(self,inputs,*,observed_domain_mask=None):
        # Same source-derived encoders and every legacy/carrier domain. A mean
        # of every raw triangle is already independently checked in the input.
        a=self.anchor_encoder(inputs["anchors"])
        context=self.context_encoder(inputs["source"]).expand(len(a),-1)
        surface=self.surface_encoder(inputs["surfaces"])+self.domain_type.weight[0]
        carrier=self.carrier_encoder(inputs["carriers"])+self.domain_type.weight[1]
        domains=torch.cat((surface,carrier),dim=0)
        if not len(domains):raise ValueError("Original domain pool must remain nonempty")
        logits=self.context_attention(a)@domains.T/math.sqrt(self.hidden)
        if observed_domain_mask is not None:
            if observed_domain_mask.shape!=logits.shape or observed_domain_mask.dtype!=torch.bool or (~observed_domain_mask.any(dim=1)).any():
                raise ValueError("Only nonempty source-derived incidence masks are accepted")
            logits=logits.masked_fill(~observed_domain_mask,-torch.inf)
        attended=logits.softmax(dim=1)@domains
        hidden=self.fusion(torch.cat((a,context,attended),dim=1))
        raw=self.raw_triangle_context_encoder(inputs["raw_triangle_context"]).expand(len(a),-1)
        hidden=self.raw_context_fusion(torch.cat((hidden,raw),dim=1))
        return dict(presence_logits=self.presence(hidden),numeric=self.numeric(hidden).reshape(-1,6,len(NUMERIC_NAMES)),
            categories={name:layer(hidden).reshape(-1,6,len(CATEGORIES[name])) for name,layer in self.category.items()},
            domain_queries=self.domain_query(hidden).reshape(-1,6,len(DOMAIN_KINDS),self.hidden),
            domain_embeddings=domains,anchor_embeddings=hidden)

    def typed_relation_predictions(self,left,right,left_roles,right_roles,kinds):
        left=left+self.role_embedding(left_roles);right=right+self.role_embedding(right_roles)
        values=torch.cat((left,right,self.relation_embedding(kinds)),dim=-1)
        return dict(logits=self.relation_score(values).squeeze(-1),order_and_group=self.relation_order(values))

    def gap_logits(self,parent,left,right):
        return self.gap_class(torch.cat((parent,left,right),dim=-1))

    def selected_domain_embeddings(self,indices,legacy,catalog):
        if indices.dtype!=torch.long or indices.ndim!=1 or ((indices<0)|(indices>=len(legacy)+catalog.count)).any():
            raise ValueError("Known loss pointer outside complete source-domain pool")
        result=legacy.new_zeros((len(indices),self.hidden))
        mask=indices<len(legacy)
        if mask.any():result[mask]=legacy[indices[mask]]
        if (~mask).any():
            rows=(indices[~mask]-len(legacy)).detach().numpy()
            result[~mask]=self.raw_embeddings(torch.from_numpy(np.array(catalog.features[rows],dtype=np.float32,copy=True)))
        return result

    def _raw_set_negative_sum(self,queries,features):
        logits=queries@self.raw_embeddings(features).T/math.sqrt(self.hidden)
        return F.softplus(logits).to(torch.float64).sum(dim=1)

    def full_domain_set_loss(self,queries,legacy,catalog,positive_query_rows,positive_domains,positive_orders,*,chunk_size=1024,replay=True):
        """Balanced multi-label loss includes EVERY candidate, not selected negatives."""
        total=len(legacy)+catalog.count
        if queries.device.type!="cpu" or legacy.device.type!="cpu":raise ValueError("Measured CPU full-pool replay required")
        if len(queries)==0:return queries.sum()*0,dict(full_denominator=True,candidate_count=total,raw_rows_consumed=0,queries=0)
        if positive_query_rows.dtype!=torch.long or positive_query_rows.shape!=positive_domains.shape or positive_orders.shape!=positive_domains.shape:
            raise ValueError("Every positive domain membership and order must be explicit")
        if ((positive_query_rows<0)|(positive_query_rows>=len(queries))).any():raise ValueError("Undefined membership query")
        pairs=list(zip(positive_query_rows.tolist(),positive_domains.tolist()))
        if len(pairs)!=len(set(pairs)):raise ValueError("Duplicate domain membership")
        counts=torch.bincount(positive_query_rows,minlength=len(queries))
        if (counts==0).any():raise ValueError("Parent allowed domains cannot be empty")
        sums=F.softplus(queries@legacy.T/math.sqrt(self.hidden)).to(torch.float64).sum(dim=1)
        visited=0
        for first,last,features in catalog.blocks(chunk_size):
            block=readonly_tensor(features)
            addition=(checkpoint(self._raw_set_negative_sum,queries,block,preserve_rng_state=False)
                if replay and torch.is_grad_enabled() and queries.requires_grad else self._raw_set_negative_sum(queries,block))
            sums=sums+addition;visited+=last-first
        if visited!=catalog.count:raise ValueError("Membership loss must consume every original raw candidate")
        selected=self.selected_domain_embeddings(positive_domains,legacy,catalog)
        positives=(queries[positive_query_rows]*selected).sum(dim=1)/math.sqrt(self.hidden)
        removed=torch.zeros(len(queries),dtype=torch.float64)
        removed.index_add_(0,positive_query_rows,F.softplus(positives).to(torch.float64))
        positive_sum=torch.zeros_like(removed)
        positive_sum.index_add_(0,positive_query_rows,F.softplus(-positives).to(torch.float64))
        negative_counts=total-counts
        negative=torch.where(negative_counts>0,(sums-removed)/negative_counts.clamp(min=1),torch.zeros_like(sums))
        membership=(positive_sum/counts+negative).mean()
        predicted_order=self.domain_set_order(torch.cat((queries[positive_query_rows],selected),dim=1)).squeeze(-1)
        order=F.smooth_l1_loss(predicted_order,positive_orders)
        loss=membership+order
        if not torch.isfinite(loss):raise ValueError("Nonfinite complete membership/order loss")
        return loss,dict(full_denominator=True,candidate_count=total,raw_rows_consumed=visited,queries=len(queries),positive_memberships=len(pairs))


def typed_supervised_loss(output,occupancy,rows,roles,categories,numeric,numeric_known):
    """Known role/category/scalar supervision only. No material-zero invention."""
    logits=output["presence_logits"]
    if occupancy.shape!=logits.shape or not torch.all((occupancy>=-1)&(occupancy<=1)):
        raise ValueError("Whole original role mask required")
    metrics={};presence=logits.sum()*0
    for value in (0,1):
        mask=occupancy==value
        if mask.any():presence=presence+F.binary_cross_entropy_with_logits(logits[mask],torch.full_like(logits[mask],float(value)))
    metrics["presence"]=presence
    predicted=output["numeric"][rows,roles]
    if predicted.shape!=numeric.shape or numeric_known.shape!=numeric.shape:raise ValueError("All281 scalar masks required")
    metrics["numeric"]=F.smooth_l1_loss(predicted[numeric_known],numeric[numeric_known]) if numeric_known.any() else predicted.sum()*0
    if categories.shape!=(len(rows),len(CATEGORIES)):raise ValueError("All281 category columns required")
    category=logits.sum()*0
    for column,(name,values) in enumerate(CATEGORIES.items()):
        targets=categories[:,column];known=targets>=0
        if known.any():
            if (targets[known]>=len(values)).any():raise ValueError("Unknown categorical value")
            category=category+F.cross_entropy(output["categories"][name][rows[known],roles[known]],targets[known])
    metrics["categories"]=category
    return sum(metrics.values()),metrics
