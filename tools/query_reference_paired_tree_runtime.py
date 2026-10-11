"""Input-only quality features for fragment/member/query K8 proposal layout."""
import numpy as np
from tools.query_reference_utility_runtime import Inputs as BaseInputs,BASE
from weld_pipeline.sjg_local_runtime_904 import read,digest
class Inputs(BaseInputs):
 def __init__(self,key):
  self.base=BaseInputs(key);self.folder=BASE/'paired_tree_runtime_v1'/key;r=read(self.folder/'receipt.json');assert r['source_geometry_receipt_sha256']==digest(self.base.folder/'receipt.json');self.K=r['K'];assert self.K==8
  for n,sha in r['files'].items():assert digest(self.folder/n)==sha
  self.pid=np.load(self.folder/'profile_ids.npy');self.canonical=np.load(self.folder/'canonical_fragments.npy');assert np.array_equal(self.pid,np.repeat(self.base.pid,self.K));assert np.array_equal(self.canonical,np.repeat(self.base.canonical,self.K));self.length=np.repeat(self.base.length,self.K);self.profiles=self.base.profiles;self.ratio=np.load(self.folder/'interval_ratios.npy').reshape(-1,2);self.role=np.load(self.folder/'weld_type.npy').ravel();self.valid=np.load(self.folder/'valid.npy').ravel();self.receipt=r
 def features(self,ids):
  ids=np.asarray(ids);original=(ids//(4*self.K))*4+ids%4;local=self.base.features(original)[:,:457];bounds=self.ratio[ids];length=self.length[ids//4];out=np.column_stack((local,bounds,self.role[ids],np.log1p(length),np.log1p((bounds[:,1]-bounds[:,0])*length))).astype(np.float32);assert out.shape==(len(ids),462);return out
