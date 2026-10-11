from pathlib import Path
import numpy as np
from weld_pipeline.server_whole_parent_features import encode_parents
from weld_pipeline.server_fixed_fragment_queries import FixedQueries
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 key='new_ac2008a03607';folder=BASE/'unified_parent_scoring_v1'/('source_'+key);r=read(folder/'receipt.json')
 assert digest(folder/'parents.json')==r['files']['parents.json'];assert digest(folder/'features.npy')==r['files']['features.npy']
 inp=FixedQueries(BASE/'frozen_features/legacy_runtime'/('source_'+key),key)
 result=encode_parents(inp,read(folder/'parents.json')['parents']);expected=np.load(folder/'features.npy')
 assert np.array_equal(expected,result['features'])
 assert np.array_equal(np.load(folder/'child_features.npy'),result['child_features'])
 assert read(folder/'parent_children.json')==result['parent_children']
 empty=encode_parents(inp,[]);assert empty['features'].shape==(0,1458)
 out=BASE/'whole_parent_feature_parity_v1';out.mkdir(exist_ok=False);write(out/'receipt.json',dict(status='PASS',parents=len(expected),unique_children=len(result['child_features']),parent_and_child_features_bitwise_equal=True,parent_child_indices_equal=True,empty_pool_supported=True,source_input_receipt_sha256=digest(folder/'receipt.json'),module_sha256=digest('weld_pipeline/server_whole_parent_features.py'),GT_reads=0));inp.close();print('PASS',len(expected),flush=True)
if __name__=='__main__':main()
