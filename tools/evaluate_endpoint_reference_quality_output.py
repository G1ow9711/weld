"""Freeze input-only quality selection on HELD, then original complete D3."""
from pathlib import Path
import pickle
import numpy as np
from threadpoolctl import threadpool_limits
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.server_compact_line_predictions import decode_compact_lines
from weld_pipeline.server_array_cached_d3_evaluation import score_native_parents_cached
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'
def main():
 modeldir=BASE/'endpoint_reference_quality_model_v1';r=read(modeldir/'receipt.json');assert digest(modeldir/'model.pkl')==r['model_sha256'];assert digest('weld_pipeline/server_endpoint_candidate_quality_features.py')==r['feature_module_sha256']
 with (modeldir/'model.pkl').open('rb') as f:model=pickle.load(f)
 root=BASE/'endpoint_reference_quality_full_output_v1';root.mkdir(exist_ok=False)
 keys=['new_cfc732e5f068','new_ceeae49d6777']
 # Freeze both cases before any target read. No threshold search or geometric NMS.
 for key in keys:
  feature=BASE/'endpoint_candidate_quality_features_v1'/key;fr=read(feature/'receipt.json');assert digest(feature/'features.npy')==fr['features_sha256'];src=BASE/'reference_detector_balanced_endpoint_full_output_v1'/key;freeze=read(src/'prediction_freeze.json');assert digest(src/'compact_predictions.npz')==freeze['files']['compact_predictions.npz'];assert digest('weld_pipeline/server_compact_line_predictions.py')==freeze['compact_decoder_sha256'];x=np.load(feature/'features.npy');z=np.load(src/'compact_predictions.npz');assert len(x)==len(z['profile_ids']);scores=model.predict_proba(x)[:,1];keep=scores>=.5;out=root/key;out.mkdir();np.save(out/'scores.npy',scores);np.savez(out/'compact_predictions.npz',**{name:z[name][keep] for name in z.files});write(out/'prediction_freeze.json',dict(key=key,threshold=.5,selected=int(keep.sum()),model_sha256=r['model_sha256'],feature_receipt_sha256=digest(feature/'receipt.json'),source_prediction_freeze_sha256=digest(src/'prediction_freeze.json'),compact_sha256=digest(out/'compact_predictions.npz'),scores_sha256=digest(out/'scores.npy'),GT_reads=0,DEV_reads=0,duplicate_suppression=False))
 rows=[]
 for key in keys:
  out=root/key;decoded=decode_compact_lines(np.load(out/'compact_predictions.npz'));target=read(BASE/'labels'/key/'targets.json');score=score_native_parents_cached(decoded,target,[False]*len(target['parents']));write(out/'new_score.json',score);row=dict(key=key,**score['categories']['overall']);rows.append(row);print(row,flush=True)
 write(root/'receipt.json',dict(status='COMPLETE_NOT_PROMOTED',rows=rows,threshold_tuned=False,fit_calls=0,DEV_reads=0,source_sha256=digest(__file__),scope='Full original D3 on selected singleton candidates; missing geometry, arcs, parent assembly and attributes unresolved'))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
