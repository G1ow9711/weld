"""Fixed input fragment queries without any stored model prediction dependency."""
import numpy as np
from .server_ordered_profile_context import ProfileSequences
from .sjg_local_runtime_904 import read
GRID=np.array([.125,.375,.625,.875])
class FixedQueries(ProfileSequences):
 def __init__(self,source,key):
  super().__init__(source,key);self.pid=np.searchsorted(self.offsets,self.canonical,side='right')-1;self.length=self.spans[self.offsets[self.pid+1]-1,1];self.profiles=read(source/'run185'/('source_'+key)/'profiles.json');self.n=len(self.canonical)*4
 def features(self,ids):
  ids=np.asarray(ids,dtype=np.int64);f=ids//4;qi=ids%4;raw=np.asarray(self.raw[np.column_stack((f*2,f*2+1))],np.float64);span=self.spans[self.canonical[f],:2];q=(span[:,0]+GRID[qi]*(span[:,1]-span[:,0]))/self.length[f];x=np.column_stack((raw.mean(1),raw.min(1),raw.max(1),q)).astype(np.float32);assert np.isfinite(x).all();return x
