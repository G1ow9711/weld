"""Input-only complete profile tokens and query-conditioned attention model."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .sjg_local_runtime_904 import read,digest

class ProfileSequences:
 def __init__(self,source,key):
  source=Path(source);folder=source/('run344' if key in ('25','36') else 'run329')/('source_'+key)
  receipt=read(folder/'receipt.json');assert digest(folder/'features.npy')==receipt['features_sha256']
  self.raw=np.load(folder/'features.npy',mmap_mode='r')
  binding=next(v for v in read(source/'run285/all_geometry_frozen_before_labels.json')['cases'] if v['source']==key)
  geometry=source/'run285'/binding['geometry_file'];assert digest(geometry)==binding['geometry_sha256']
  with np.load(geometry) as z:self.offsets=z['offsets'];self.spans=z['spans']
  alias=source/'run288'/('source_'+key+'_alias_map.npz')
  with np.load(alias) as z:self.canonical=z['canonical_fragments'];self.representative=z['fragment_representative']
  self.bindings=dict(features_sha256=receipt['features_sha256'],geometry_sha256=digest(geometry),alias_sha256=digest(alias))
 def tokens(self,pid):
  a,b=self.offsets[pid:pid+2];ids=np.arange(a,b);rep=self.representative[ids];rows=np.searchsorted(self.canonical,rep);assert np.array_equal(self.canonical[rows],rep)
  span=self.spans[a:b,:2];length=float(span[-1,1]);assert length>0 and np.all(span[:,1]>span[:,0]);assert np.allclose(span[1:,0],span[:-1,1],atol=1e-8,rtol=0)
  raw=np.asarray(self.raw[np.column_stack((rows*2,rows*2+1))],np.float64)
  tokens=np.column_stack((raw.mean(1),raw.min(1),raw.max(1),span/length,np.full(len(span),np.log1p(length)))).astype(np.float32)
  assert tokens.shape==(b-a,459) and np.isfinite(tokens).all()
  return tokens,length
 def close(self):self.raw._mmap.close()

class OrderedContextModel(nn.Module):
 def __init__(self):
  super().__init__()
  self.local=nn.Sequential(nn.Linear(457,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,4))
  self.query=nn.Sequential(nn.Linear(457,128),nn.ReLU())
  self.tokens=nn.Sequential(nn.Linear(459,128),nn.ReLU())
  self.attention=nn.MultiheadAttention(128,4,batch_first=True)
  self.residual=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,4))
  nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
 def forward(self,x,q,tokens,valid):
  h=self.query(x);t=self.tokens(tokens);context,_=self.attention(h[:,None,:],t,t,key_padding_mask=~valid,need_weights=False)
  y=self.local(x)+self.residual(torch.cat((h,context[:,0]),1));d=torch.sigmoid(y[:,:2]);return torch.stack((q*(1-d[:,0]),q+(1-q)*d[:,1]),1),y[:,2:]
