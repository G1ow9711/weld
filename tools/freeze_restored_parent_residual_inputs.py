"""Freeze real original parent features for an isolated additive experiment.

No targets, historical predictions, STEP, candidate pruning, or fitting are used.
The original 160-column parent schema remains distinct from interval features.
"""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'server_rebuild/best_baseline_restore'
DEST=ROOT/'server_rebuild/parent_residual_v1'
KEYS=('21','BM79A_native_local','10','25','36')

def main():
 DEST.mkdir(exist_ok=False);cases=[]
 for key in KEYS:
  source=BASE/'legacy_runtime'/('source_'+key);folder=source/'run379'/('test_'+key)
  pred=folder/'decoded.json';data=read(pred);x=np.load(folder/'unary.npy',mmap_mode='r')
  assert x.shape==(len(data['parents']),160) and np.isfinite(x).all()
  assert len(read(folder/'parent_metadata.json'))==len(data['parents'])
  base=source/'run446'/('test_'+key);receipt=read(base/'prediction_receipt.json')
  assert receipt['GT_reads']==receipt['STEP_reads']==0
  assert digest(base/'final_decoded.json')==receipt['decoded_sha256']
  paths=dict(parent_features=folder/'unary.npy',parents=pred,metadata=folder/'parent_metadata.json',precision_scores=folder/'precision_scores.npz',baseline=base/'final_decoded.json',baseline_receipt=base/'prediction_receipt.json')
  cases.append(dict(source=key,role='TRAIN',parent_candidates=len(data['parents']),child_candidates=sum(p['child_count'] for p in data['parents']),columns=160,files={name:dict(path=str(path.relative_to(ROOT)),sha256=digest(path)) for name,path in paths.items()}))
  x._mmap.close()
 write(DEST/'inputs_frozen.json',dict(schema='ORIGINAL379_PARENT_UNARY160_WITH_FROZEN446_BASELINE_V1',cases=cases,GT_reads=0,STEP_reads=0,DEV_reads=0,fit_calls=0,pruning_applied=False,interval_features_not_fabricated=True,model_changed=False))
 print('Frozen',sum(x['parent_candidates'] for x in cases),'real parent candidates across',len(cases),'TRAIN cases')

if __name__=='__main__':main()
