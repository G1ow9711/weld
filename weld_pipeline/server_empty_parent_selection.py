"""Explicit empty-candidate selection, preserving the complete invalid ledger.

The original selector cannot call sklearn on an empty 160-column input. No
scores or candidates are invented; all frozen weight bindings are still checked.
"""
import time
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.sjg_atomic_checkpoint_1100 import atomic_binary_write

def select_if_empty(dest,key):
 start=time.perf_counter();folder=dest/('test_'+key);original=read(folder/'decoded.json')
 if original['parents']:return False
 u=np.load(folder/'unary.npy');assert u.shape==(0,160)
 assert not read(folder/'parent_metadata.json')
 assert not read(folder/'compact_parents.json')['parents']
 points=np.load(folder/'parent_points.npy',mmap_mode='r');assert len(points)==0;points._mmap.close()
 training=read(dest/'selector_training.json')
 for name,sha in training['frozen_files'].items():assert digest(dest/name)==sha
 assert digest(dest/'precision_model.pkl')==training['precision_model_sha256']
 result=dict(original);result['parents']=[];result['total_prediction_children']=sum(v['counted_false_children'] for v in original['invalid_predictions']);result.update(runtime_model='FROZEN376_WITH_GEOMETRIC_PRECISION_RESIDUAL',GT_reads=0)
 write(folder/'final_decoded.json',result)
 atomic_binary_write(folder/'precision_scores.npz',lambda f:np.savez(f,probability=np.empty(0),deduction=np.empty(0)))
 write(folder/'final_prediction.json',dict(status='LABEL_FREE_PRECISION_PREDICTION_FROZEN',source=key,selected_candidate_indices=[],candidate_count=0,decoded_sha256=digest(folder/'final_decoded.json'),selector_sha256=training['selector_sha256'],precision_model_sha256=training['precision_model_sha256'],GT_reads=0,STEP_reads=0,invalid_ledger_unchanged=True,inference_classifier_evaluations=0,empty_input_early_stop=True,empty_input_adapter_sha256=digest(__file__),seconds=time.perf_counter()-start))
 return True
