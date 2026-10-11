"""Same input-only selector features plus actual predicted-endpoint contexts."""
import numpy as np
from tools.query_reference_utility_runtime import Inputs as BaseInputs,BASE
from weld_pipeline.sjg_local_runtime_904 import digest
class Inputs(BaseInputs):
 def __init__(self,key):
  super().__init__(key)
  src=BASE/'frozen_features/legacy_runtime'/('source_'+key);alias=src/'run288'/('source_'+key+'_alias_map.npz');assert digest(alias)==self.receipt['alias_sha256']
  with np.load(alias) as z:representative=z['fragment_representative']
  self.rows=np.searchsorted(self.canonical,representative);assert np.array_equal(self.canonical[self.rows],representative)
  # Disjoint numerical coordinate ranges allow vectorized search within each
  # actual finite profile. These offsets are indexing only, not world geometry.
  lengths=self.spans[self.offsets[1:]-1,1];self.origins=np.r_[0.,np.cumsum(lengths[:-1]+1.)];pids=np.repeat(np.arange(len(lengths)),np.diff(self.offsets));self.global_right=self.spans[:,1]+self.origins[pids];assert np.all(np.diff(self.global_right)>=0)
 def features(self,ids):
  ids=np.asarray(ids);base=super().features(ids);pid=self.pid[ids//4];length=self.length[ids//4];bounds=self.ratio[ids]*length[:,None];extras=[]
  for end in (0,1):
   positions=bounds[:,end]+self.origins[pid];frag=np.searchsorted(self.global_right,positions,side='right' if end==0 else 'left');frag=np.maximum(self.offsets[pid],np.minimum(frag,self.offsets[pid+1]-1));row=self.rows[frag];xx=np.asarray(self.x[np.column_stack((2*row,2*row+1))],np.float64);span=self.spans[frag,:2];width=span[:,1]-span[:,0];fraction=(bounds[:,end]-span[:,0])/np.maximum(width,1e-12);extras.append(np.column_stack((xx.mean(1),xx.min(1),xx.max(1),fraction,np.log1p(width))))
  out=np.column_stack((base,*extras)).astype(np.float32);assert out.shape==(len(ids),1378) and np.isfinite(out).all();return out
