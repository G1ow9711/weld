"""Query-conditioned input-event endpoint scores with explicit supervision masks."""
import torch
from torch import nn
class BoundaryEventSelector(nn.Module):
 def __init__(self):
  super().__init__()
  def encoder():return nn.Sequential(nn.Linear(457,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
  self.query=encoder();self.event=encoder();self.score=nn.Sequential(nn.Linear(259,128),nn.ReLU(),nn.Linear(128,2))
 def forward(self,query_features,event_features,query_ratio,event_ratio,length_mm):
  q=self.query(query_features)[:,None,:].expand(-1,event_features.shape[1],-1);e=self.event(event_features);delta=event_ratio-query_ratio[:,None];relative=torch.stack((delta,delta*length_mm[:,None]/1000,length_mm[:,None].expand_as(delta)/1000),-1)
  return self.score(torch.cat((q,e,relative),-1))

def masked_endpoint_event_loss(logits,legal,positive):
 """Positive sets are acceptable events; unknown endpoint heads have no loss."""
 assert logits.shape==legal.shape==positive.shape and legal.dtype==positive.dtype==torch.bool
 assert not torch.any(positive&~legal),'A target event must be an input-legal candidate'
 terms=[]
 for side in (0,1):
  rows=positive[:,:,side].any(1)
  if not rows.any():continue
  values=logits[rows,:,side].masked_fill(~legal[rows,:,side],-torch.inf);lp=values.log_softmax(1);good=lp.masked_fill(~positive[rows,:,side],-torch.inf);terms.append(-good.logsumexp(1))
 if not terms:return logits[torch.isfinite(logits)].sum()*0
 return torch.cat(terms).mean()
