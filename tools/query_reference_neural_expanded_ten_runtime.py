"""Input-only filter view of the frozen local neural full-query proposals."""
import numpy as np
from tools.query_reference_utility_runtime import Inputs as BaseInputs,BASE
from weld_pipeline.sjg_local_runtime_904 import read,digest
class Inputs(BaseInputs):
 def __init__(self,key):
  super().__init__(key)
  source=self.folder;self.folder=BASE/'positive_query_neural_expanded_ten'/key;r=read(self.folder/'receipt.json');assert r['source_geometry_receipt_sha256']==digest(source/'receipt.json')
  for name,sha in r['files'].items():assert digest(self.folder/name)==sha
  assert np.array_equal(self.pid,np.load(self.folder/'profile_ids.npy'));assert np.array_equal(self.canonical,np.load(self.folder/'canonical_fragments.npy'))
  self.ratio=np.load(self.folder/'interval_ratios.npy').reshape(-1,2);self.role=np.load(self.folder/'weld_type.npy').ravel();self.valid=np.load(self.folder/'valid.npy').ravel();self.receipt=r
